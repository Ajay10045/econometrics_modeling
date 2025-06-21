import logging
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


def mixed_modeling_node(feature_engineered_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Fit a mixed effects model and return coefficient estimates."""
    df = feature_engineered_data.copy()
    formula = params["formula"]
    group_col = params.get("group_column", "ppg_id")
    re_formula = params.get("re_formula", "1")
    vc_formula = params.get("vc_formula", {"retailer": "0 + C(retailer_id)"})

    logger.info("Model formula: %s", formula)
    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df[group_col],
        re_formula=re_formula,
        vc_formula=vc_formula,
    )
    result = model.fit()

    coef = pd.DataFrame(result.params, columns=["estimate"])
    coef.index.name = "term"
    return coef.reset_index()
