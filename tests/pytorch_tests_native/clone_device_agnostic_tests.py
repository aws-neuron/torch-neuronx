"""Clone PyTorch tests with device-agnostic changes and Neuron patches."""

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

PYTORCH_FORK_REPO = "https://github.com/amazon-contributing/upstream-to-pytorch.git"
PYTORCH_FORK_BRANCH = "main"


def _clone_from_local(dest_dir: Path, source_path: Path):
    """Clone test/ and torch/testing/_internal/ from local pytorch repo."""
    # Clone test/
    test_dir = dest_dir / "test"
    src_test = source_path / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    shutil.copytree(src_test, test_dir)
    logger.info(f"Cloned test/ from {source_path}")

    # Remove PyTorch's conftest.py
    pt_conftest = test_dir / "conftest.py"
    if pt_conftest.exists():
        pt_conftest.unlink()
        logger.info("Removed PyTorch test/conftest.py")

    # Clone torch/testing/_internal/
    src = source_path / "torch" / "testing" / "_internal"
    dest = dest_dir / "torch" / "testing" / "_internal"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    logger.info(f"Cloned torch/testing/_internal/ to {dest}")


def _clone_from_remote(dest_dir: Path, repo: str, branch: str):
    """Clone test/ and torch/testing/_internal/ from remote repo."""
    test_dir = dest_dir / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    with tempfile.TemporaryDirectory(prefix="temp_pytorch_") as temp_dir:
        temp_path = Path(temp_dir)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                branch,
                "--depth",
                "1",
                "-c",
                "advice.detachedHead=false",
                repo,
                temp_path,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "test", "torch/testing/_internal"],
            cwd=temp_path,
            check=True,
        )

        # Copy test/
        shutil.copytree(temp_path / "test", test_dir)
        logger.info(f"Cloned test/ from {repo}")

        # Remove PyTorch's conftest.py
        pt_conftest = test_dir / "conftest.py"
        if pt_conftest.exists():
            pt_conftest.unlink()
            logger.info("Removed PyTorch test/conftest.py")

        # Copy torch/testing/_internal/
        dest = dest_dir / "torch" / "testing" / "_internal"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(temp_path / "torch" / "testing" / "_internal", dest)
        logger.info(f"Cloned torch/testing/_internal/ to {dest}")


def _apply_neuron_patch(dest_dir: Path):
    """Inject Neuron setup code at top of all test files."""
    test_dir = dest_dir / "test"
    if not test_dir.exists():
        return

    patch = """\
# Neuron test setup - apply overlay and register backend
import os as _os
_overlay = _os.environ.get("TORCH_TESTING_INTERNAL_PATH")
if _overlay:
    import torch.testing._internal as _internal
    if _overlay not in _internal.__path__:
        _internal.__path__.insert(0, _overlay)
    import torch_neuronx  # noqa: F401

"""
    count = 0
    for py_file in test_dir.rglob("*.py"):
        content = py_file.read_text()
        if "torch.testing._internal" not in content:
            continue
        # Insert after first line (comment/shebang)
        lines = content.split("\n", 1)
        new_content = lines[0] + "\n\n" + patch + (lines[1] if len(lines) > 1 else "")
        py_file.write_text(new_content)
        count += 1

    logger.info(f"Patched {count} test files with Neuron setup")


def clone(source_repo: str | None = None):
    """Clone PyTorch tests with device-agnostic changes.

    Args:
        source_repo: Local path to pytorch fork. If None, clones from remote fork.
    """
    dest_dir = Path(__file__).resolve().parent

    if source_repo:
        source_path = Path(source_repo)
        if not source_path.exists():
            raise RuntimeError(f"Source repo not found: {source_path}")
        _clone_from_local(dest_dir, source_path)
    else:
        _clone_from_remote(dest_dir, PYTORCH_FORK_REPO, PYTORCH_FORK_BRANCH)

    _apply_neuron_patch(dest_dir)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone PyTorch tests with device-agnostic changes")
    parser.add_argument("--source-repo", type=str, help="Local path to pytorch fork")
    args = parser.parse_args()
    clone(source_repo=args.source_repo)
