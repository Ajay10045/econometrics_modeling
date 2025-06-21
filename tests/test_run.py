"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
from pathlib import Path
import importlib
import pytest

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

if importlib.util.find_spec("julia") is None:
    pytest.skip("Julia is not installed", allow_module_level=True)


class TestKedroRun:
    def test_kedro_run(self):
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            assert session.run() is not None
