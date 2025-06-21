import logging
from collections.abc import Iterable

import pandas as pd
import statsmodels.formula.api as smf

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


def mixed_modeling_node(feature_engineered_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Fit a mixed effects model and return coefficient estimates."""

    df = feature_engineered_data.copy()
    formula = params.get("formula")
    if not formula:
        formula = prepare_formula_for_MM(params)

    hierarchy = params.get("hierarchy_levels", [params.get("group_column", "ppg")])
    group_col = params.get("group_column", hierarchy[-1])
    re_formula = params.get("re_formula", "1")
    vc_formula = params.get("vc_formula", {"retailer": "0 + C(retailer_id)"})

    grouping_columns = hierarchy
    df = prepare_data_for_MM(df, grouping_columns, group_col)

    logger.info("Model formula: %s", formula)
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df[group_col],
            re_formula=re_formula,
            vc_formula=vc_formula,
        )
        result = model.fit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Model fitting failed: %s", exc)
        return pd.DataFrame(columns=["term", "estimate"])

    coef = pd.DataFrame(result.params, columns=["estimate"])
    coef.index.name = "term"
    return coef.reset_index()

