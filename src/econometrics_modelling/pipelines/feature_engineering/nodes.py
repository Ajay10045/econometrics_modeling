import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def feature_engineering_node(rolled_up_beverage_data: pd.DataFrame, holiday_calendar: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Applies feature engineering to rolled-up beverage data.
    """
    df = rolled_up_beverage_data.copy()

    # 1️⃣ Calculate avg_price = total_sales / total_volume
    df['avg_price'] = df['total_sales'] / df['total_volume']

    # 2️⃣ Calculate EDLP price = rolling max of avg_price over 14 weeks by PPG and retailer
    df = df.sort_values(['ppg_id', 'retailer_id', 'week_id'])
    df['edlp_price'] = df.groupby(['ppg_id', 'retailer_id'])['avg_price'].transform(lambda x: x.rolling(window=14, min_periods=1).max())

    # 3️⃣ Merge with holiday calendar on week_id
    df = pd.merge(df, holiday_calendar, on='week_id', how='left')

    # 4️⃣ Calculate CPI, XPI, and OPI (dummy values, placeholder for now)
    df['cpi'] = np.random.uniform(1.0, 1.5, len(df))
    df['xpi'] = np.random.uniform(0.8, 1.2, len(df))
    df['opi'] = np.random.uniform(0.9, 1.1, len(df))

    # 5️⃣ Log transformations
    df['log_total_volume'] = np.log1p(df['total_volume'])
    df['log_avg_price'] = np.log1p(df['avg_price'])
    for col in ['promo_acv_tpr', 'promo_acv_feature', 'promo_acv_display', 'promo_acv_feature_display', 'cpi', 'xpi', 'opi']:
        df[f'log_{col}'] = np.log1p(df[col])

    # 6️⃣ Trend using LOESS smoothing
    loess_frac = params.get('feature_engineering.loess_frac', 0.3)
    df['trend'] = df.groupby(['ppg_id', 'retailer_id'])['log_total_volume'].transform(lambda x: lowess(x, np.arange(len(x)), frac=loess_frac, return_sorted=False))

    # 7️⃣ Seasonality placeholder (to be implemented based on param option)
    seasonality_method = params.get('feature_engineering.seasonality_method', 'dummy')
    if seasonality_method == 'dummy':
        df = pd.get_dummies(df, columns=['week_id'], prefix='week')

    return df