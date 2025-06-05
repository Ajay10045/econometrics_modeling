"""Project pipelines."""

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines


from kedro.pipeline import Pipeline
from econometrics_modelling.pipelines.data_ingestion import pipeline as data_ingestion_pipeline
from econometrics_modelling.pipelines.data_preprocessing import pipeline as data_preprocessing_pipeline
from econometrics_modelling.pipelines.feature_engineering import pipeline as feature_engineering_pipeline
from econometrics_modelling.pipelines.mixed_modelling import pipeline as mixed_modeling_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_ingestion": data_ingestion_pipeline.create_pipeline(),
        "data_preprocessing": data_preprocessing_pipeline.create_pipeline(),
        "feature_engineering": feature_engineering_pipeline.create_pipeline(),
        "mixed_modeling": mixed_modeling_pipeline.create_pipeline(),
        "__default__": data_ingestion_pipeline.create_pipeline() + data_preprocessing_pipeline.create_pipeline() + feature_engineering_pipeline.create_pipeline() + mixed_modeling_pipeline.create_pipeline()
    }