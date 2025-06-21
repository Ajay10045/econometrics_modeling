"""Utilities for fitting mixed models using Julia."""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_data_for_MM(
    data: pd.DataFrame,
    categorical_columns: Iterable[str],
    level_1: str,
) -> pd.DataFrame:
    """Ensure categorical variables have at least two levels.

    If a column only contains a single level Julia's ``MixedModels`` throws an
    error. This helper appends a dummy row with value ``"dummy"`` for such
    columns.
    """
    df = data.copy()
    need_dummy = False
    dummy = {}
    for col in categorical_columns:
        if df[col].nunique() <= 1:
            need_dummy = True
            dummy[col] = "dummy"
        else:
            dummy[col] = df[col].iloc[0]
    if need_dummy:
        for col in df.columns:
            if col not in dummy:
                dummy[col] = df[col].iloc[0]
        df = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)
    return df


def prepare_formula_for_MM(params: dict) -> str:
    """Create a formula string for ``MixedModels.jl`` based on parameters."""
    target = params.get("target", "log_total_volume")
    fixed_effects: list[str] = []
    for i in range(1, 5):
        level_name = params.get(f"lvl{i}")
        vars_ = params.get(f"fe_lvl{i}_var", [])
        for var in vars_:
            if level_name:
                fixed_effects.append(f"{var}*{level_name}")
            else:
                fixed_effects.append(var)
    fe = " + ".join(fixed_effects)

    random_parts: list[str] = []
    for i in range(1, 5):
        level_name = params.get(f"lvl{i}")
        vars_ = params.get(f"re_lvl{i}_var", [])
        if level_name:
            term = "1"
            if vars_:
                term += " + " + " + ".join(vars_)
            random_parts.append(f"({term}|{level_name})")
    re = " + ".join(random_parts)
    if fe and re:
        formula = f"{target} ~ {fe} + {re}"
    elif fe:
        formula = f"{target} ~ {fe}"
    else:
        formula = f"{target} ~ {re}"
    return formula


def get_random_effects_stats(
    random_dt: pd.DataFrame, var: Iterable, dof: Iterable, grouping_columns: Iterable[str]
) -> pd.DataFrame:
    """Attach variance and degrees of freedom to random effects table."""
    random_dt = random_dt.copy()
    random_dt["var"] = list(var)
    random_dt["dof"] = list(dof)
    for col in grouping_columns:
        if col not in random_dt:
            random_dt[col] = None
    return random_dt


def get_fixed_effects_stats(
    fixed_dt: pd.DataFrame, formula: str, categ_cols: Iterable[str]
) -> pd.DataFrame:
    """Add a simple significance flag to fixed effects table."""
    fixed_dt = fixed_dt.copy()
    fixed_dt["significant"] = fixed_dt["p_value"] < 0.05
    fixed_dt["formula"] = formula
    return fixed_dt


def fit_and_predict(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Fit a mixed model in Julia and return predictions and statistics."""
    from julia.api import Julia
    from julia import Main
    level_1 = params.get("lvl1")
    grouping_columns = [params.get(f"lvl{i}") for i in range(1, 5) if params.get(f"lvl{i}")]

    data_to_fit = prepare_data_for_MM(data, grouping_columns, level_1)
    formula = prepare_formula_for_MM(params)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_filepath = os.path.join(tmpdir, "data.csv")
        data_to_fit.to_csv(data_filepath, index=False)

        jl = Julia(compiled_modules=False)
        Main.include(os.path.join("src", "models", "mixed_model.jl"))
        Main.fm = formula
        result = Main.mixed_model_fn(data_filepath, Main.fm)

    (
        residuals,
        pred,
        rand_eff,
        effect,
        estimate,
        stderr,
        z_value,
        p_value,
        var,
        dof,
    ) = result
    logger.info("Mixed model built successfully")

    data_to_fit["pred"] = list(pred)
    data_to_fit["resid"] = list(residuals)
    results = data_to_fit[data_to_fit[level_1] != "dummy"].reset_index(drop=True)

    random_dt = pd.DataFrame(rand_eff).sort_values(by=[level_1], key=lambda col: col.str.lower())
    random_dt = get_random_effects_stats(random_dt, var, dof, grouping_columns)

    fixed_dt = pd.DataFrame(
        zip(effect, estimate, stderr, z_value, p_value),
        columns=["effect", "estimate", "stderr", "z_value", "p_value"],
    )
    categ_cols = data_to_fit.select_dtypes(include=["object", "category"]).columns.tolist()
    fixed_dt = get_fixed_effects_stats(fixed_dt, formula, categ_cols)

    Main.GC.gc()
    del jl
    del Main
    gc.collect()

    return results, fixed_dt, random_dt, formula
