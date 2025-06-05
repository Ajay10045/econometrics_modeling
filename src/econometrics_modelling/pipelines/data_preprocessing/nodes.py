import pandas as pd

def data_rollup_node(raw_beverage_data: pd.DataFrame, product_master_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Merges raw POS data with product master, rolls up to PPG level, and aggregates sales and promo ACVs.
    """
    merged_data = pd.merge(
        raw_beverage_data,
        product_master_data,
        on='sku_id',
        how='left'
    )

    min_sales_threshold = params.get('preprocessing.min_sales_threshold', 0)
    merged_data = merged_data[merged_data['total_volume'] >= min_sales_threshold]

    # Weighted average function for ACV columns
    def weighted_avg(x, weight_col='total_volume'):
        return (x * merged_data.loc[x.index, weight_col]).sum() / merged_data.loc[x.index, weight_col].sum()

    grouped = merged_data.groupby(['ppg_id', 'retailer_id', 'week_id']).agg({
        'total_volume': 'sum',
        'promo_volume': 'sum',
        'total_sales': 'sum',
        'promo_sales': 'sum',
        'promo_acv_tpr': lambda x: weighted_avg(x),
        'promo_acv_feature': lambda x: weighted_avg(x),
        'promo_acv_display': lambda x: weighted_avg(x),
        'promo_acv_feature_display': lambda x: weighted_avg(x),
        'acv_weighted_distribution': lambda x: weighted_avg(x),
        'brand': 'first',
        'sub_brand': 'first',
        'size': 'first',
        'pack_count': 'first'
    }).reset_index()

    return grouped