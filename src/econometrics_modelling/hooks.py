from kedro.framework.hooks import hook_impl
try:  # pragma: no cover - optional dependency
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except Exception:  # pragma: no cover - optional dependency
    SparkConf = None
    SparkSession = None
    import logging

    logging.getLogger(__name__).warning("pyspark not available; SparkHooks disabled")


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """
        if SparkConf is None or SparkSession is None:  # pragma: no cover - optional dependency
            return

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
