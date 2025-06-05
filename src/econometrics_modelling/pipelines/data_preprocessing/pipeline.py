from kedro.pipeline import Pipeline, node
from .nodes import data_rollup_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            data_rollup_node,
            inputs=["raw_beverage_data", "product_master_data", "params:preprocessing"],
            outputs="rolled_up_beverage_data",
            name="data_rollup_node"
        )
    ])