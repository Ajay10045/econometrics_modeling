import pandas as pd

from econometrics_modelling.pipelines.mixed_modelling.nodes import (
    prepare_data_for_MM,
    prepare_formula_for_MM,
)


def test_prepare_data_for_MM_adds_dummy():
    df = pd.DataFrame({"ppg": ["A"], "x": [1]})
    res = prepare_data_for_MM(df, ["ppg"], "ppg")
    assert "dummy" in res["ppg"].values
    assert len(res) == 2


def test_prepare_formula_for_MM():
    params = {
        "target": "log_vol",
        "lvl1": "ppg",
        "fe_lvl1_var": ["log_price"],
        "re_lvl1_var": ["log_price"],
    }
    formula = prepare_formula_for_MM(params)
    assert "log_vol" in formula
    assert "log_price*ppg" in formula
    assert "(1 + log_price|ppg)" in formula

