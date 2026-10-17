from pathlib import Path

import pytest
from agent_skills_test_support import write_valid_bundle


@pytest.fixture
def valid_bundle(tmp_path: Path) -> Path:
    write_valid_bundle(tmp_path)
    return tmp_path
