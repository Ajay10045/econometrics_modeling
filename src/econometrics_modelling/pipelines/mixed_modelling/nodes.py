import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_data_for_MM(
    data: pd.DataFrame, categorical_columns: Iterable[str], level_1: str
) -> pd.DataFrame:
    """Ensure categorical variables have at least two levels."""

    df = data.copy()
    need_dummy = False
    dummy: dict[str, object] = {}
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
    """Construct a mixed model formula from a hierarchical specification."""

    spec = params
    target = spec.get("model_specification", {}).get(
        "dependent_variable", "y"
    )

    fixed_terms: list[str] = []
    fixed_terms.extend(spec.get("model_specification", {}).get("main_effects", []))

    for interaction in (
        spec.get("model_specification", {})
        .get("fixed_effects", {})
        .get("interactions", [])
    ):
        fixed_terms.append(f"{interaction['measure']}:{interaction['with_level']}")

    random_terms: list[str] = []
    uncorrelated = (
        spec.get("model_specification", {})
        .get("random_effects", {})
        .get("uncorrelated", {})
    )
    for lvl in uncorrelated.get("intercepts", []):
        random_terms.append(f"(1|{lvl})")
    for slope in uncorrelated.get("slopes", []):
        random_terms.append(f"(0+{slope['measure']}|{slope['by_level']})")

    for corr in (
        spec.get("model_specification", {})
        .get("random_effects", {})
        .get("correlated", [])
    ):
        intercept = "1+" if corr.get("with_intercept", False) else "0+"
        random_terms.append(f"({intercept}{corr['measure']}|{corr['by_level']})")

    fe = " + ".join(fixed_terms)
    re = " + ".join(random_terms)

    if fe and re:
        return f"{target} ~ {fe} + {re}"
    if fe:
        return f"{target} ~ {fe}"
    return f"{target} ~ {re}"


def mixed_modeling_node(feature_engineered_data: pd.DataFrame, params: dict):
    try:
        from julia.api import Julia
        Julia(compiled_modules=False)  # 🚨 Must be done BEFORE importing from julia
        from julia import Main
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia: {e}")

    df = feature_engineered_data.copy()
    formula = params.get("formula") or prepare_formula_for_MM(params)

    logger.info("Model formula: %s", formula)

    # Save to temp file
    data_path = Path("data/08_model_input/feature_data.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    # Load Julia environment and call model
    Main.include("src/econometrics_modelling/pipelines/mixed_modelling/mixed_model.jl")  # or wherever mixed_model_fn is
    group_vars = list(params.get("hierarchy_levels", []))
    results = Main.mixed_model_fn(str(data_path), formula, group_vars)

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
    ) = results

    df["pred"] = pred
    df["resid"] = residuals

    results_df = df.copy()
    results_df = results_df[results_df[params["hierarchy_levels"][-1]] != "dummy"].reset_index(drop=True)

    fixed_dt = pd.DataFrame(
        zip(effect, estimate, stderr, z_value, p_value),
        columns=["term", "estimate", "stderr", "z_value", "p_value"]
    )
    return fixed_dt
