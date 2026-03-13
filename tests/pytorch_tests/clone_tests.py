import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_torch_version():
    """Get the current torch version from environment"""
    import torch

    version = torch.__version__
    # Remove any extra suffixes (e.g., "+cpu", "+cu118")
    version = version.split("+")[0]
    logger.info(f"Auto-detected torch version: {version}")
    return version


def _clone_pytorch_tests(torch_version, test_dir, pytorch_tests_dir):
    logger.info(f"Cloning PyTorch {torch_version} tests...")

    with tempfile.TemporaryDirectory(dir=pytorch_tests_dir, prefix="temp_pytorch_") as temp_dir:
        temp_path = Path(temp_dir)

        # Clone with sparse checkout
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                torch_version,
                "--depth",
                "1",
                "-c",
                "advice.detachedHead=false",
                "https://github.com/pytorch/pytorch.git",
                temp_path,
            ],
            check=True,
            capture_output=True,
        )

        # Configure sparse checkout for test directory only
        subprocess.run(["git", "sparse-checkout", "set", "test"], cwd=temp_path, check=True)

        # Move test directory to final location
        source_test_dir = temp_path / "test"
        if not source_test_dir.exists():
            raise RuntimeError(f"Expected test directory not found: {source_test_dir}")

        try:
            source_test_dir.rename(test_dir)
        except FileExistsError:
            # Another worker already renamed the temp directory, skip this
            print(f"Directory {test_dir} already exists, skipping.")


def clone_pytorch_tests(torch_version=None):
    """Clone PyTorch tests directory.

    Clones the test directory from PyTorch repository. If tests already exist,
    checks version compatibility and re-clones if there's a mismatch.

    Args:
        torch_version: PyTorch version to clone. Auto-detects if not provided.
    """
    if torch_version is None:
        torch_version = get_torch_version()

    # fetch pre-release branch
    if "2.10" in torch_version:
        torch_version = "release/2.10"
    # Convert released version to tag format
    elif not torch_version.startswith("v"):
        torch_version = f"v{torch_version}"

    script_path = Path(__file__).resolve()
    pytorch_tests_dir = script_path.parent
    test_dir = pytorch_tests_dir / "test"
    version_file = test_dir / ".pytorch_version"

    # Check if test_dir exists and has matching version
    if test_dir.exists():
        if version_file.exists():
            stored_version = version_file.read_text().strip()
            if stored_version == torch_version:
                logger.debug(f"Test directory exists with matching version {torch_version}")
                return
            logger.info(f"Version mismatch: stored={stored_version}, current={torch_version}")
        else:
            logger.info("No version file found in existing test directory")
        # Delete and re-clone
        logger.info("Removing outdated test directory...")
        shutil.rmtree(test_dir)

    # Clone tests
    logger.info(
        f"Test directory {test_dir} doesn't exist. Cloning PyTorch {torch_version} tests..."
    )
    try:
        _clone_pytorch_tests(torch_version, test_dir, pytorch_tests_dir)
    except Exception as e:
        logger.exception(f"Setup script failed: {e}")
        raise RuntimeError(f"Failed to setup PyTorch tests: {e}") from e

    if not test_dir.exists():
        raise RuntimeError(
            f"Setup script completed but test directory {test_dir} still doesn't exist"
        )

    # Write version marker
    version_file.write_text(torch_version)
    logger.info(f"PyTorch tests setup completed. Test directory: {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pytorch tests in torch_neuronx environment")
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

    clone_pytorch_tests(args.torch_version)
