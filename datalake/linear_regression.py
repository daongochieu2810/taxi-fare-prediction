from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

inputCols = [
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "trip_type",
    "weather",
]

dateCols = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]


def getTripDuration(datetime_start, datetime_end):
    try:
        result = (
            datetime.strptime(datetime_end, "%Y-%m-%d %H:%M:%S")
            - datetime.strptime(datetime_start, "%Y-%m-%d %H:%M:%S")
        ).total_seconds()
        return result
    except Exception:
        return 0


def run(spark, data):
    data_modified = data.withColumnRenamed("tempAvg", "weather")
    data_modified = data_modified.select(
        [col(c).cast(DoubleType()) for c in inputCols] + [col(c) for c in dateCols]
    )

    tripDurationFunction = F.udf(getTripDuration, DoubleType())
    data_modified = data_modified.withColumn(
        "trip_duration",
        tripDurationFunction("lpep_pickup_datetime", "lpep_dropoff_datetime"),
    )
    assembler = VectorAssembler(
        inputCols=inputCols,
        outputCol="features",
    )
    output = assembler.transform(data_modified)
    final_data = output.select("features", "total_amount")
    final_data = spark.createDataFrame(final_data.limit(20).collect())
    train_data, test_data = final_data.randomSplit([0.7, 0.3])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print("RMSE: " + str(test_results.rootMeanSquaredError))
    print("R2: " + str(test_results.r2))
    # final_data.describe().show()
