import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

import torch_neuronx
from tests.pytorch_tests.clone_tests import clone_pytorch_tests
from tests.utils.neuron_test_utils import track_neuron_ops

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

pytorch_tests_dir = Path("tests/pytorch_tests")


def _apply_opinfo_patches():
    """Patch OpInfo to filter complex dtypes for Neuron."""
    import torch
    from torch.testing._internal.common_device_type import DeviceTypeTestBase
    from torch.testing._internal.opinfo.core import OpInfo

    # Skip if already patched
    if getattr(OpInfo.supported_dtypes, "_neuron_patched", False):
        return

    complex_dtypes = frozenset({torch.complex32, torch.complex64, torch.complex128})

    orig_supported_dtypes = OpInfo.supported_dtypes
    orig_supported_backward_dtypes = OpInfo.supported_backward_dtypes

    def patched_supported_dtypes(self, device_type):
        result = orig_supported_dtypes(self, device_type)
        if device_type in ("privateuse1", "neuron"):
            return result - complex_dtypes
        return result

    def patched_supported_backward_dtypes(self, device_type):
        result = orig_supported_backward_dtypes(self, device_type)
        if device_type in ("privateuse1", "neuron"):
            return result - complex_dtypes
        return result

    patched_supported_dtypes._neuron_patched = True
    OpInfo.supported_dtypes = patched_supported_dtypes
    OpInfo.supported_backward_dtypes = patched_supported_backward_dtypes

    orig_get_dtypes = DeviceTypeTestBase._get_dtypes

    @classmethod
    def patched_get_dtypes(cls, test):
        result = orig_get_dtypes.__func__(cls, test)
        if result is None or cls.device_type not in ("privateuse1", "neuron"):
            return result
        filtered = tuple(
            d
            for d in result
            if not (
                d in complex_dtypes
                or (isinstance(d, tuple) and any(x in complex_dtypes for x in d))
            )
        )
        return filtered or None

    DeviceTypeTestBase._get_dtypes = patched_get_dtypes


def _apply_gradcheck_patches():
    """Patch gradcheck/gradgradcheck with float32-appropriate eps for Neuron.

    Neuron doesn't natively support float64 — it downcasts to float32, computes,
    then upcasts back to float64, so we use torch.finfo(torch.float32).eps as the
    finite-difference step size.
    """
    import torch
    from torch.testing._internal import common_utils

    _f32_eps = torch.finfo(torch.float32).eps

    orig_gradcheck = common_utils.gradcheck
    orig_gradgradcheck = common_utils.gradgradcheck

    def patched_gradcheck(*args, **kwargs):
        kwargs.setdefault("eps", _f32_eps)
        return orig_gradcheck(*args, **kwargs)

    def patched_gradgradcheck(*args, **kwargs):
        kwargs.setdefault("eps", _f32_eps)
        return orig_gradgradcheck(*args, **kwargs)

    common_utils.gradcheck = patched_gradcheck
    common_utils.gradgradcheck = patched_gradgradcheck


@dataclass
class PytorchTestConfig:
    """Configuration for a single test file/class combination."""

    test_module: object
    test_class_name: str
    test_methods: list[str]
    xfail: list[dict[str, dict[str, str]]]
    skip_tests: list[dict[str, dict[str, str]]]


def _normalize_test_list_format(
    test_config, config_type: str = "test"
) -> list[dict[str, dict[str, str]]]:
    """
    Function to normalize test configuration (xfail, skip_tests, etc.) to a standard format.
    """
    if not test_config:
        return []

    normalized_list = []

    for item in test_config:
        if "tests" in item and "reason" in item:
            reason_dict = {"reason": item["reason"]}
            for test_name in item["tests"]:
                normalized_list.append({test_name: reason_dict})
        elif "test_list_filename" in item and "reason" in item:
            reason_dict = {"reason": item["reason"]}
            with open(pytorch_tests_dir / item["test_list_filename"]) as f:
                parsed_json = json.load(f)
            test_list = parsed_json.get("failed_tests", [])
            for test_name in test_list:
                normalized_list.append({test_name: reason_dict})
        else:
            for test_name, test_config_dict in item.items():
                if isinstance(test_config_dict, dict) and "reason" in test_config_dict:
                    normalized_list.append({test_name: test_config_dict})

    return normalized_list


def _normalize_xfail_format(xfail_config) -> list[dict[str, dict[str, str]]]:
    return _normalize_test_list_format(xfail_config, "xfail")


def _normalize_skip_tests_format(skip_tests_config) -> list[dict[str, dict[str, str]]]:
    return _normalize_test_list_format(skip_tests_config, "skip")


def _get_test_methods_from_class(test_class) -> list[str]:
    return [
        method
        for method in dir(test_class)
        if method.startswith("test_") and callable(getattr(test_class, method))
    ]


def _filter_test_methods(
    test_methods: list[str],
    specified_methods: list[str] | None = None,
    sanity_methods: list[str] | None = None,
    sanity_pt_distributed_tests: bool = False,
) -> list[str]:
    if specified_methods is not None:
        test_methods = [method for method in test_methods if method in specified_methods]

    if sanity_pt_distributed_tests:
        if sanity_methods is None:
            logger.info(
                "sanity_pt_distributed_tests flag is enabled, while there are no "
                "sanity_methods methods so skipping run."
            )
            return []
        else:
            test_methods = [method for method in test_methods if method in sanity_methods]

    return test_methods


def _create_test_config(
    test_module: object,
    class_name: str,
    test_methods: list[str],
    config: dict,
    collect_skipped_tests: bool = False,
) -> PytorchTestConfig:
    normalized_xfail = _normalize_xfail_format(config.get("xfail", []))

    # Handle folder-level xfail
    if config.get("folder_xfail_all", False):
        folder_xfail_reason = config.get("folder_xfail_reason", "Folder-level xfail for all tests")
        folder_xfail = _create_xfail_for_all_tests(test_methods, folder_xfail_reason)
        normalized_xfail.extend(folder_xfail)

    normalized_skip_tests = _normalize_skip_tests_format(config.get("skip_tests", []))

    # Handle skip_class - mark all tests in class as skipped
    if config.get("skip_class"):
        skip_reason = config.get("skip_reason", "Class skipped")
        logger.info(f"Skipping entire class {class_name}: {skip_reason}")
        class_skip_tests = [{method: {"reason": skip_reason}} for method in test_methods]
        normalized_skip_tests.extend(class_skip_tests)

    # Handle skip_test_patterns - skip tests matching substring patterns
    skip_patterns = config.get("skip_test_patterns", [])
    if skip_patterns:
        pattern_reason = config.get("skip_test_patterns_reason", "Matched skip pattern")
        for method in test_methods:
            if any(p in method for p in skip_patterns):
                normalized_skip_tests.append({method: {"reason": pattern_reason}})

    if normalized_skip_tests:
        skip_test_names = set()
        default_skip_reason = "No reason provided, should provide valid reason to skip test"
        for skip_item in normalized_skip_tests:
            for test_name, skip_config in skip_item.items():
                skip_test_names.add(test_name)
                logger.debug(
                    f"Skipping test {test_name}: {skip_config.get('reason', default_skip_reason)}"
                )

        # Remove skipped tests from the methods list unless collect_skipped_tests is True
        # When True, tests remain in list and pytest.mark.skip is applied in test wrapper
        # For skip_class, always keep tests to show them as skipped
        if not collect_skipped_tests and not config.get("skip_class") and not skip_patterns:
            test_methods = [method for method in test_methods if method not in skip_test_names]
            logger.info(
                f"After applying skip_tests filter, {len(test_methods)} test methods remain"
            )

    return PytorchTestConfig(
        test_module=test_module,
        test_class_name=class_name,
        test_methods=test_methods,
        xfail=normalized_xfail,
        skip_tests=normalized_skip_tests,
    )


def _get_test_classes_from_module(test_module) -> list[str]:
    test_classes = []
    for name in dir(test_module):
        if name.startswith("_"):
            # Skip non-test classes
            continue
        obj = getattr(test_module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "__module__")
            and obj.__module__ == test_module.__name__
        ):
            has_test_methods = any(
                method.startswith("test_") and callable(getattr(obj, method)) for method in dir(obj)
            )
            if has_test_methods:
                test_classes.append(name)
    return test_classes


def _apply_class_name_suffix(
    test_module: object, class_name: str, config: dict
) -> tuple[str, type]:
    """Apply device-specific suffix to test class name and return the class."""
    class_name_suffix_options = config.get("class_name_suffix_options")
    if not class_name_suffix_options:
        return class_name, getattr(test_module, class_name)

    for class_name_suffix in class_name_suffix_options:
        try:
            neuron_test_class_name = class_name + class_name_suffix
            pytorch_test_class = getattr(test_module, neuron_test_class_name)
            return neuron_test_class_name, pytorch_test_class
        except AttributeError:
            continue

    # Fallback to base class if no suffixed version exists
    logger.warning(f"No suffixed class found for {class_name}, using base class")
    return class_name, getattr(test_module, class_name)


def _process_test_class(
    test_module: object,
    class_name: str,
    specified_methods: list[str] | None,
    config: dict,
    sanity_pt_distributed_tests: bool = False,
    collect_skipped_tests: bool = False,
) -> PytorchTestConfig:
    # Apply suffix to class name and get class object
    neuron_test_class_name, pytorch_test_class = _apply_class_name_suffix(
        test_module, class_name, config
    )
    # Get all test methods
    test_methods = _get_test_methods_from_class(pytorch_test_class)
    logger.info(
        f"Found {len(test_methods)} test methods in {neuron_test_class_name}: {test_methods}"
    )

    # Get sanity_methods from config
    sanity_methods = config.get("sanity_methods")

    # Filter to specified methods if provided, or sanity_methods
    #   if sanity_pt_distributed_tests is True
    filtered_methods = _filter_test_methods(
        test_methods,
        specified_methods,
        sanity_methods,
        sanity_pt_distributed_tests,
    )
    logger.info(f"After initial filtering: {len(filtered_methods)} test methods")

    return _create_test_config(
        test_module,
        neuron_test_class_name,
        filtered_methods,
        config,
        collect_skipped_tests,
    )


def _process_all_classes_in_module(
    test_module: object,
    specified_methods: list[str] | None,
    config: dict,
    sanity_pt_distributed_tests: bool = False,
    collect_skipped_tests: bool = False,
) -> list[PytorchTestConfig]:
    test_configs = []
    test_classes = _get_test_classes_from_module(test_module)
    skip_classes = config.get("skip_classes", [])

    logger.info(f"Found {len(test_classes)} test classes in module: {test_classes}")

    for class_name in test_classes:
        if any(skip in class_name for skip in skip_classes):
            logger.info(f"Skipping class {class_name} due to skip_classes config")
            continue
        try:
            # Use the same _process_test_class function
            pytorch_test_config = _process_test_class(
                test_module,
                class_name,
                specified_methods,
                config,
                sanity_pt_distributed_tests,
                collect_skipped_tests,
            )
            test_configs.append(pytorch_test_config)
        except AttributeError as e:
            logger.warning(f"Could not process class {class_name}: {e}")
            continue

    return test_configs


def _process_single_test_configuration(
    config: dict,
    test_dir: Path,
    sanity_pt_distributed_tests: bool = False,
    collect_skipped_tests: bool = False,
) -> list[PytorchTestConfig]:
    test_file_name = config["file"]
    test_class_name = config.get("class")
    specified_methods = config.get("methods")

    # Verify test file exists
    test_file = test_dir / test_file_name
    if not test_file.exists():
        raise FileNotFoundError(f"Expected test file not found: {test_file}")

    try:
        # Add test directory to sys.path for relative imports within PyTorch tests
        if str(test_dir) not in sys.path:
            sys.path.insert(0, str(test_dir))

        # Add distributed test directory to sys.path to resolve import errors
        # Cloned PyTorch test files use relative imports like "import test_c10d_common"
        distributed_dir = test_dir / "distributed"
        if distributed_dir.exists() and str(distributed_dir) not in sys.path:
            sys.path.insert(0, str(distributed_dir))

        # Import the test module
        import_path = "tests.pytorch_tests.test"
        relative_parts = ".".join(Path(test_file_name).with_suffix("").parts)
        module_import_path = f"{import_path}.{relative_parts}"

        test_module = importlib.import_module(module_import_path)

        if test_class_name is None:
            # Process all test classes in the module
            return _process_all_classes_in_module(
                test_module,
                specified_methods,
                config,
                sanity_pt_distributed_tests,
                collect_skipped_tests,
            )
        else:
            # Process single specified test class
            pytorch_test_config = _process_test_class(
                test_module,
                test_class_name,
                specified_methods,
                config,
                sanity_pt_distributed_tests,
                collect_skipped_tests,
            )
            return [pytorch_test_config]

    except ImportError as e:
        logger.exception(f"Failed to import PyTorch test module {module_import_path}: {e}")
        raise


def _get_all_test_files_in_folder(
    folder_path: str, test_dir: Path, exclude_subfolders: list[str] | None = None
) -> list[str]:
    folder_full_path = test_dir / folder_path
    if not folder_full_path.exists() or not folder_full_path.is_dir():
        logger.warning(f"Folder not found or not a directory: {folder_full_path}")
        return []

    test_files = []
    exclude_subfolders = exclude_subfolders or []

    for file_path in folder_full_path.rglob("test_*.py"):
        if file_path.is_file():
            # Check if file is in an excluded subdirectory
            relative_to_folder = file_path.relative_to(folder_full_path)
            if any(
                str(relative_to_folder).startswith(excl + "/")
                or str(relative_to_folder).startswith(excl + "\\")
                for excl in exclude_subfolders
            ):
                logger.debug(f"Excluding file {relative_to_folder} due to exclude_subfolders")
                continue

            relative_path = file_path.relative_to(test_dir)
            test_files.append(str(relative_path))

    logger.info(
        f"Found {len(test_files)} test files in folder {folder_path} "
        f"(excluded subfolders: {exclude_subfolders}): {test_files}"
    )
    return test_files


def _should_xfail_all_folder_tests(config: dict, args: argparse.Namespace) -> bool:
    if (
        args.test_file_name is not None
        or args.test_class_name is not None
        or args.test_method_name is not None
    ):
        return False

    return config.get("methods") is None and config.get("class") is None


def _create_xfail_for_all_tests(
    test_methods: list[str], reason: str = "Folder-level xfail for all tests"
) -> list[dict[str, dict[str, str]]]:
    xfail_list = []
    for method in test_methods:
        xfail_list.append({method: {"reason": reason}})
    return xfail_list


def _expand_folder_configurations(
    test_configurations: list[dict], test_dir: Path, args: argparse.Namespace
) -> list[dict]:
    expanded_configs = []

    for config in test_configurations:
        if "folder" in config:
            folder_path = config["folder"]
            exclude_subfolders = config.get("exclude_subfolders", [])
            test_files = _get_all_test_files_in_folder(folder_path, test_dir, exclude_subfolders)

            should_xfail_all = _should_xfail_all_folder_tests(config, args)

            for test_file in test_files:
                file_config = config.copy()
                file_config["file"] = test_file

                del file_config["folder"]
                # Remove exclude_subfolders from file config as it's no longer needed
                if "exclude_subfolders" in file_config:
                    del file_config["exclude_subfolders"]

                if should_xfail_all and not file_config.get("xfail"):
                    file_config["folder_xfail_all"] = True
                    file_config["folder_xfail_reason"] = config.get(
                        "xfail_reason", "Folder-level xfail for all tests"
                    )

                expanded_configs.append(file_config)
        else:
            expanded_configs.append(config)

    return expanded_configs


def _parse_test_configurations(
    args: argparse.Namespace, test_config_json: str
) -> list[dict[str, str]]:
    """Parse test configurations from either JSON file or command line arguments."""

    if args.test_file_name is None:
        # Load from JSON file
        logger.info(f"Reading test configuration from JSON file: {test_config_json}")

        json_file_path = Path(test_config_json)
        if not json_file_path.exists():
            raise FileNotFoundError(f"Test config JSON file not found: {json_file_path}")

        try:
            with open(json_file_path) as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {json_file_path}: {e}") from e

        return json_data.get("test_configurations", [])

    else:
        test_method_list = args.test_method_name.split(",") if args.test_method_name else None
        class_name_suffix_options = (
            args.class_name_suffix_options.split(",") if args.class_name_suffix_options else None
        )

        return [
            {
                "file": args.test_file_name,
                "class": args.test_class_name,
                "class_name_suffix_options": class_name_suffix_options,
                "methods": test_method_list,
                "xfail": [],
                "skip_tests": [],
            }
        ]


def _create_test_class_for_config(config: PytorchTestConfig) -> type:
    """Create a single test class for a PytorchTestConfig."""
    class_name = config.test_class_name
    test_module = config.test_module
    original_class = getattr(test_module, class_name)

    def setup_class(cls):
        cls.pytorch_test_class = original_class
        cls.tracking_dir = Path(f"op_tracking_{class_name}")
        cls.tracking_dir.mkdir(exist_ok=True)

    def _run_pytorch_test(self, method_name):
        test_instance = self.pytorch_test_class()
        if hasattr(test_instance, "setUp"):
            test_instance.setUp()
        with track_neuron_ops():
            try:
                getattr(test_instance, method_name)()
            except Exception as e:
                logger.error(f"{method_name} failed: {e}")
                raise
            finally:
                executed_ops = torch_neuronx.get_executed_ops()
                fallback_ops = torch_neuronx.get_fallback_ops()
                worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
                tracking_file = self.tracking_dir / f"ops_tracking_{worker_id}.json"
                data = defaultdict(list)
                if tracking_file.exists():
                    with open(tracking_file) as f:
                        data = json.load(f)
                data["test_method"].append(method_name)
                data["test_class"].append(class_name)
                data["executed_ops"].append(executed_ops)
                data["fallback_ops"].append(fallback_ops)
                with open(tracking_file, "w") as f:
                    json.dump(data, f)
                if hasattr(test_instance, "tearDown"):
                    test_instance.tearDown()

    # Create class with setup_class and _run_pytorch_test
    test_cls = type(
        class_name,
        (),
        {
            "setup_class": classmethod(setup_class),
            "_run_pytorch_test": _run_pytorch_test,
        },
    )

    # Add test methods
    for method_name in config.test_methods:

        def make_test(m_name):
            def test_method(self):
                self._run_pytorch_test(m_name)

            test_method.__name__ = m_name
            return test_method

        setattr(test_cls, method_name, make_test(method_name))

    # Apply xfail markers
    for xfail in config.xfail:
        xfail_kw = next(iter(xfail.keys()))
        for m_name in config.test_methods:
            if xfail_kw in m_name:
                method = getattr(test_cls, m_name)
                marked = pytest.mark.xfail(reason=xfail[xfail_kw]["reason"])(method)
                setattr(test_cls, m_name, marked)

    # Apply skip markers
    for skip in config.skip_tests:
        skip_kw = next(iter(skip.keys()))
        for m_name in config.test_methods:
            if skip_kw in m_name:
                method = getattr(test_cls, m_name)
                setattr(test_cls, m_name, pytest.mark.skip(reason=skip[skip_kw]["reason"])(method))

    return test_cls


def create_pytorch_test_wrappers(configs: list[PytorchTestConfig]) -> dict[str, type]:
    """
    Create wrapper classes for PyTorch tests, one per original test class.

    Args:
        configs: List of test configurations from setup_pytorch_tests
    Returns:
        Dict mapping class name to generated test class
    """
    return {config.test_class_name: _create_test_class_for_config(config) for config in configs}


def setup_pytorch_tests(
    spec_file="test_spec.json",
    sanity_pt_distributed_tests=False,
    class_name_filter=None,
    collect_skipped_tests=False,
):
    """Clone pytorch tests, import test module and instantiate the test class.

    Args:
        spec_file: JSON spec file name in tests/pytorch_tests/specs/
        sanity_pt_distributed_tests: Enable sanity mode for distributed tests
        class_name_filter: If provided, only process configs matching this class name.
        collect_skipped_tests: If true, collect skipped tests (appear as skipped in report).
    """
    parser = argparse.ArgumentParser(description="Run pytorch tests in torch_neuronx environment")
    parser.add_argument(
        "--test-file-name",
        type=str,
        default=None,
        help="PyTorch test file to import. Defaults to filename in test_spec.json",
    )
    parser.add_argument(
        "--test-class-name",
        type=str,
        default=None,
        help="PyTorch test class name to instantiate. Defaults to classes in test_spec.json",
    )
    parser.add_argument(
        "--class-name-suffix-options",
        type=str,
        default=None,
        help=(
            "Comma-separated suffixes to try appending to model class names. Required for "
            "certain classes to run on neuron device (e.g., 'PRIVATEUSE1,NEURON')"
        ),
    )
    parser.add_argument(
        "--test-method-name",
        type=str,
        default=None,
        help="Comma-delimited name of the test methods. If not specified, all methods will execute",
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default=None,
        help="PyTorch version to clone. If not specified, auto-detect from environment",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, _ = parser.parse_known_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    test_dir = pytorch_tests_dir / "test"
    spec_dir = pytorch_tests_dir / "specs"
    test_config_json = spec_dir / spec_file

    logger.info(f"Using spec file: {test_config_json}")

    # Check if test directory exists, if not raise an error
    if not test_dir.exists() or not any(test_dir.iterdir()):
        clone_pytorch_tests(args.torch_version)

    # Add pytorch tests to path
    sys.path.insert(0, str(pytorch_tests_dir))

    test_configurations = _parse_test_configurations(args, test_config_json)
    if not test_configurations:
        raise ValueError("No test configurations found")

    # Early filter by class name - filters JSON configs BEFORE importing test modules
    if class_name_filter:
        original_count = len(test_configurations)
        test_configurations = [
            cfg
            for cfg in test_configurations
            if cfg.get("class") and class_name_filter in cfg["class"]
        ]
        logger.info(
            f"Filtered by '{class_name_filter}': {original_count} -> {len(test_configurations)}"
        )
        if not test_configurations:
            raise ValueError(f"No test configurations found matching class '{class_name_filter}'")

    # Expand folder configurations into individual file configurations
    test_configurations = _expand_folder_configurations(test_configurations, test_dir, args)

    logger.info(f"Processing {len(test_configurations)} test configurations")

    logger.info(f"Sanity mode enabled: {sanity_pt_distributed_tests}")

    # Process all configurations
    test_configs = []
    for config in test_configurations:
        configs = _process_single_test_configuration(
            config,
            test_dir,
            sanity_pt_distributed_tests,
            collect_skipped_tests,
        )
        test_configs.extend(configs)

    if not test_configs:
        raise RuntimeError(
            "No valid test configurations found. Please check JSON file or command line arguments."
        )

    logger.info(f"Created {len(test_configs)} test configurations")
    return test_configs
