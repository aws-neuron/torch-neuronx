import logging
import os
import shutil

import pytest
from infra.resource_manager import ResourceManager
from kaena_test_framework import kaena_constants as kconstants
from kaena_test_framework.logger import exception

KAIZEN_DOCKERFILE = os.path.join(os.path.dirname(__file__), "Dockerfile.kaizen")
ROOT = os.path.dirname(os.path.dirname(__file__))
REMOTE_EXEC_DIR = "torch_neuron_eager"
REMOTE_SRC_DIR = "torch_neuron"
REMOTE_TEST_DIR = "tests"
REMOTE_TOOLS_DIR = "tools"

# Output paths
WORKSPACE_DIR = f"/workspace/{REMOTE_EXEC_DIR}"
OUTPUTS_DIR = f"{WORKSPACE_DIR}/outputs"
REPORTS_DIR = f"{OUTPUTS_DIR}/test-reports"

logger = logging.getLogger(__name__)

# TODO: Modify this to be at max 2 hours after splitting of targets
# as these are used for CI tests
KAIZEN_TEST_TIMEOUT = "14400"


@pytest.fixture(scope="session", autouse=True)
def setup():
    # packing files into src folder and unpacking within dockerfile to reduce build context
    # using root as build context, kaizen uploads everything resulting in 1 hour build times
    # kaizen build context doesn't support single files
    shutil.copy(
        os.path.join(ROOT, "conftest.py"), os.path.join(ROOT, "torch_neuron", "conftest.py")
    )
    shutil.copy(os.path.join(ROOT, "setup.py"), os.path.join(ROOT, "torch_neuron", "setup.py"))
    shutil.copy(
        os.path.join(ROOT, "pyproject.toml"), os.path.join(ROOT, "torch_neuron", "pyproject.toml")
    )

    resource = ResourceManager().create(
        resource_type=kconstants.AWS_COMPUTE_RESOURCE_NEURON_SHARED_EKS_CLUSTER,
        node_count=1,
        instance_type="trn2.48xlarge",
        timeout=KAIZEN_TEST_TIMEOUT,
    )
    image_uri = resource.install(
        dockerfile=KAIZEN_DOCKERFILE,
        build_context=[
            f"{REMOTE_SRC_DIR}={ROOT}/{REMOTE_SRC_DIR}",
            f"{REMOTE_TEST_DIR}={ROOT}/{REMOTE_TEST_DIR}",
            f"{REMOTE_TOOLS_DIR}={ROOT}/{REMOTE_TOOLS_DIR}",
        ],
        skip_if_exists=True,
        extends_base_dlc=False,
        follow_symlinks=True,
    )
    resource.terminate()
    yield (image_uri,)
    # TODO: aggregate test artifacts here
    pass


@pytest.mark.test_unit_suites_kaizen
def test_unit_suites_kaizen(setup: tuple, request: pytest.FixtureRequest):
    (image_uri,) = setup
    try:
        test_suite_path = request.config.option.test_suite_path or ""
        num_workers = getattr(request.config.option, "num_workers", None)
        num_workers_arg = f"--num_workers={num_workers}" if num_workers else ""
        ignore_paths = getattr(request.config.option, "ignore_paths", None) or []
        ignore_args = " ".join([f"--ignore={path}" for path in ignore_paths])
        skip_distributed = getattr(request.config.option, "skip_distributed", False)
        skip_distributed_arg = "--skip-distributed" if skip_distributed else ""

        json_report = getattr(request.config.option, "json_report", False)
        json_report_arg = "--json-report" if json_report else ""

        skip_pt_test_distributed = getattr(request.config.option, "skip_pt_test_distributed", False)
        skip_pt_test_distributed_arg = (
            "--skip_pt_test_distributed" if skip_pt_test_distributed else ""
        )
        sanity_pt_distributed_tests = getattr(
            request.config.option, "sanity_pt_distributed_tests", False
        )
        sanity_pt_distributed_tests_arg = (
            "--sanity_pt_distributed_tests" if sanity_pt_distributed_tests else ""
        )

        test_cmd = (
            f"./tools/run-test-parallel {test_suite_path} {skip_distributed_arg} {ignore_args} "
            f" {skip_pt_test_distributed_arg} {sanity_pt_distributed_tests_arg} {num_workers_arg} "
            f"--coverage {json_report_arg} --report-dir={REPORTS_DIR}"
        ).strip()

        commands = [
            f"cd {REMOTE_EXEC_DIR}",
            "mkdir outputs",
            f"{test_cmd}; echo $? > test_exit_code",
            f"cp -r /workspace/{REMOTE_EXEC_DIR}/outputs /ustore/workload_output_rw/",
            "exit $(cat test_exit_code)",
        ]
        command = " && ".join(commands)

        resource = ResourceManager().create(
            resource_type=kconstants.AWS_COMPUTE_RESOURCE_NEURON_SHARED_EKS_CLUSTER,
            force_driver_reinstall=False,
            node_count=1,
            instance_type="trn2.48xlarge",
            image_uri=image_uri,
            timeout=KAIZEN_TEST_TIMEOUT,
        )
        workload_id = resource.execute(
            "unit_tests", [resource.create_task(command=command, timeout=KAIZEN_TEST_TIMEOUT)]
        )
    except Exception as e:
        workload_id = getattr(e, "workload_id", None)
        logger.error(f"Exception Type: {type(e).__name__}")
        logger.error(f"Exception Details: {e}")
        logger.error(f"Traceback: {e.__traceback__}")
        exception(e)
        raise e
    finally:
        resource.get_artifact(
            "unit_tests",
            "/outputs",
            "build/brazil-unit-tests/outputs",
            workload_id,
            "WORKLOAD_OUTPUT_RW",
            True,
        )
        pass
