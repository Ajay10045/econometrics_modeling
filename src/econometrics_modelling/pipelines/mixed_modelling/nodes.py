import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def build_mixed_model_formula(params: dict) -> str:
    fixed_effects = []

    for i in range(1, 5):
        level_var = params.get(f'mixed_modeling.fe_lvl{i}_var', [])
        level_name = params.get(f'mixed_modeling.lvl{i}', '')
        for var in level_var:
            if level_name:
                fixed_effects.append(f'{var}*{level_name}')
            else:
                fixed_effects.append(var)

    fixed_effects_str = ' + '.join(fixed_effects)

    random_effects = ' + '.join([f'(1|{lvl})' for lvl in ['ppg_id', 'retailer_id']])

    formula = f'log_total_volume ~ {fixed_effects_str} + {random_effects}'
    return formula


def mixed_modeling_node(feature_engineered_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = feature_engineered_data.copy()

    #formula = build_mixed_model_formula(params)
    formula = 'log_total_volume ~ log_avg_price + log_promo_acv_tpr + trend + holiday_flag + log_cpi'
    print(f"Model formula: {formula}")

    md = smf.mixedlm(formula, df, groups=df['ppg_id'])
    mdf = md.fit()

    fixed_coefs = mdf.fe_params
    random_coefs = mdf.random_effects

    coef_list = []
    for key, value in random_coefs.items():
        coef_list.append({
            'ppg_id': key,
            **value
        })
    coef_df = pd.DataFrame(coef_list)
    coef_df = coef_df.merge(df[['ppg_id', 'retailer_id']].drop_duplicates(), on='ppg_id', how='left')

    return coef_df


def clean_coefficients_node(model_coefficients: pd.DataFrame, feature_engineered_data: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(
        model_coefficients,
        feature_engineered_data[['ppg_id', 'retailer_id']].drop_duplicates(),
        on='ppg_id',
        how='left'
    )
    return merged_df