from kedro.pipeline import Pipeline, node
from .nodes import mixed_modeling_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            mixed_modeling_node,
            inputs=["feature_engineered_data", "params:mixed_modeling"],
            outputs="model_coefficients",
            name="mixed_modeling_node"
        )
    ])