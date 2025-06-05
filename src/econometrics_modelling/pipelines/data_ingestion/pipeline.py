from kedro.pipeline import Pipeline, node
from .nodes import generate_raw_beverage_data, generate_product_master_data, generate_holiday_calendar

def create_pipeline(**kwargs):
    return Pipeline([
        node(generate_raw_beverage_data, inputs=None, outputs="raw_beverage_data", name="generate_raw_beverage_data_node"),
        node(generate_product_master_data, inputs=None, outputs="product_master_data", name="generate_product_master_data_node"),
        node(generate_holiday_calendar, inputs=None, outputs="holiday_calendar", name="generate_holiday_calendar_node"),
    ])