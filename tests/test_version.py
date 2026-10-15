import os
import shutil
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("pytorch_version", "expected_version"),
    [
        ("2.12.0", "2.12.3.0.123+abcdef.dev"),
        ("2.13.0", "2.13.3.0.123+abcdef.dev"),
        ("2.14.0", "2.14.3.0.123+abcdef.dev"),
    ],
)
def test_package_metadata_uses_selected_pytorch_version(
    tmp_path, pytorch_version, expected_version
):
    package_dir = tmp_path / "torch_neuronx"
    package_dir.mkdir()
    shutil.copy(ROOT_DIR / "setup.py", tmp_path / "setup.py")
    shutil.copy(ROOT_DIR / "torch_neuronx" / "_version.py", package_dir / "_version.py")
    (package_dir / "_build_config.py").write_text('BUILD_TYPE = "DEV"\n')
    (package_dir / "__init__.py").touch()

    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    (torch_dir / "__init__.py").write_text(f'__version__ = "{pytorch_version}"\n')

    egg_base = tmp_path / "metadata"
    egg_base.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "GIT_SHA": "abcdef",
            "PYTHONPATH": str(tmp_path),
            "TORCH_NEURONX_BUILD_PYTORCH_VERSION": pytorch_version,
            "VERSION_BUILD": "123",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            str(egg_base),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata_path = next(egg_base.glob("*.egg-info/PKG-INFO"))
    metadata = Parser().parsestr(metadata_path.read_text())
    assert metadata["Version"] == expected_version
    assert f"torch=={pytorch_version}" in metadata.get_all("Requires-Dist")
