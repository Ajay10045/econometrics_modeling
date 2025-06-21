import importlib
import pytest

spec = importlib.util.find_spec("pandas")
if spec is None:
    pytest.skip("pandas is not installed", allow_module_level=True)
try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception as exc:  # pragma: no cover - skip if broken install
    pytest.skip(f"pandas cannot be imported: {exc}", allow_module_level=True)

from econometrics_modelling.pipelines.mixed_modelling.nodes import (
    prepare_data_for_MM,
    prepare_formula_for_MM,
)


def test_prepare_data_for_MM_adds_dummy():
    df = pd.DataFrame({"ppg": ["A"], "x": [1]})
    res = prepare_data_for_MM(df, ["ppg"], "ppg")
    assert "dummy" in res["ppg"].values
    assert len(res) == 2  # noqa: PLR2004


def test_prepare_formula_for_MM():
    params = {
        "hierarchy_levels": ["ppg"],
        "model_specification": {
            "dependent_variable": "log_vol",
            "main_effects": ["log_price"],
            "random_effects": {
                "correlated": [
                    {"measure": "log_price", "by_level": "ppg", "with_intercept": True}
                ]
            },
        },
    }
    formula = prepare_formula_for_MM(params)
    assert "log_vol" in formula
    assert "log_price" in formula
    assert "(1+log_price|ppg)" in formula

