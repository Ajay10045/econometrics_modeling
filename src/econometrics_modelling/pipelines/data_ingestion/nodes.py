import pandas as pd
import numpy as np

def generate_raw_beverage_data() -> pd.DataFrame:
    np.random.seed(42)
    sku_ids = ['Cola_250ml', 'Cola_500ml', 'Juice_1L', 'Soda_330ml']
    retailer_ids = ['Retailer_A', 'Retailer_B', 'Retailer_C']
    weeks = pd.date_range(start='2024-01-01', periods=52, freq='W-MON').isocalendar().week

    data = []
    for week in weeks:
        for sku in sku_ids:
            for retailer in retailer_ids:
                total_volume = np.random.randint(50, 500)
                promo_volume = np.random.randint(0, total_volume // 2)
                total_sales = total_volume * np.round(np.random.uniform(1.0, 3.0), 2)
                promo_sales = promo_volume * np.round(np.random.uniform(1.0, 3.0), 2)
                promo_acv_tpr = np.round(np.random.uniform(0, 100), 2)
                promo_acv_feature = np.round(np.random.uniform(0, 100), 2)
                promo_acv_display = np.round(np.random.uniform(0, 100), 2)
                promo_acv_feature_display = np.round(np.random.uniform(0, 100), 2)
                acv_weighted_distribution = np.round(np.random.uniform(60, 100), 2)

                data.append([
                    sku, retailer, week, total_volume, promo_volume,
                    total_sales, promo_sales, promo_acv_tpr, promo_acv_feature,
                    promo_acv_display, promo_acv_feature_display, acv_weighted_distribution
                ])

    return pd.DataFrame(data, columns=[
        'sku_id', 'retailer_id', 'week_id', 'total_volume', 'promo_volume',
        'total_sales', 'promo_sales', 'promo_acv_tpr', 'promo_acv_feature',
        'promo_acv_display', 'promo_acv_feature_display', 'acv_weighted_distribution'
    ])


def generate_product_master_data() -> pd.DataFrame:
    sku_ids = ['Cola_250ml', 'Cola_500ml', 'Juice_1L', 'Soda_330ml']
    ppg_ids = ['PPG1', 'PPG2', 'PPG3', 'PPG4']
    brand = ['Cola', 'Cola', 'Juice', 'Soda']
    sub_brand = ['Classic', 'Zero', 'Fresh', 'Fizz']
    size = ['250ml', '500ml', '1L', '330ml']
    pack_count = [1, 1, 1, 1]

    data = list(zip(sku_ids, ppg_ids, brand, sub_brand, size, pack_count))
    return pd.DataFrame(data, columns=['sku_id', 'ppg_id', 'brand', 'sub_brand', 'size', 'pack_count'])


def generate_holiday_calendar() -> pd.DataFrame:
    weeks = range(1, 53)
    holidays = np.random.choice([0, 1], size=52, p=[0.8, 0.2])
    return pd.DataFrame({
        'week_id': weeks,
        'holiday_flag': holidays
    })