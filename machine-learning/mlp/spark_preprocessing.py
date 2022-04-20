from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
import os


def preprocess_data(params):

    save_dir = params["save_dir"]
    data_dir = params["data_dir"]

    spark = SparkSession.builder.appName("TaxiFarePrediction").getOrCreate()

    # read green taxi data for prototyping
    df_taxi = spark.read.option("header", True).csv(
        os.path.join(data_dir, "green_tripdata_2021-01.csv")
    )

    # generate dummy data
    df_taxi_filtered = (
        df_taxi.withColumn("traffic", randn(seed=42))
        .withColumn("weather", randn(seed=42))
        .withColumn("demand", rand(seed=42))
        .select(
            df_taxi.trip_distance.cast("double"),
            df_taxi.passenger_count.cast("integer"),
            "traffic",
            "weather",
            "demand",
            df_taxi.total_amount.cast("double").alias("fare"),
        )
        .dropna()
    )

    # split into train, test data
    df_train, df_test = df_taxi_filtered.limit(1000).randomSplit([0.8, 0.2], seed=42)

    # create pipeline for scaling train and test data
    # inputCols: features to be used for training and prediction
    flatten_assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "traffic", "weather", "demand"],
        outputCol="unscaled_features",
    )  # flatten into 1 col
    std_scaler = StandardScaler(
        inputCol="unscaled_features",
        outputCol="standardised_features",
        withMean=True,
        withStd=True,
    )
    min_max_scaler = MinMaxScaler(
        inputCol="standardised_features", outputCol="scaled_features"
    )  # default: min=0.0, max=1.0
    preproc_pipeline = Pipeline(stages=[flatten_assembler, std_scaler, min_max_scaler])

    # scale train and test data using train stats
    preproc_pipeline_model = preproc_pipeline.fit(df_train)
    df_train_scaled = preproc_pipeline_model.transform(df_train)
    df_test_scaled = preproc_pipeline_model.transform(df_test)

    # save processed data
    df_train_scaled.repartition(1).write.parquet(
        os.path.join(save_dir, "train_data"), mode="overwrite"
    )
    df_test_scaled.repartition(1).write.parquet(
        os.path.join(save_dir, "test_data"), mode="overwrite"
    )

    spark.stop()
