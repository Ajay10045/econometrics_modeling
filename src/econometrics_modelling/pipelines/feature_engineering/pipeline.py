from kedro.pipeline import Pipeline, node
from .nodes import feature_engineering_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            feature_engineering_node,
            inputs=["rolled_up_beverage_data", "holiday_calendar", "params:feature_engineering"],
            outputs="feature_engineered_data",
            name="feature_engineering_node"
        )
    ])