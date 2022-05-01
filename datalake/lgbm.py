from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from datetime import datetime
from pyspark.sql.types import DoubleType

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
        [col(c).cast(DoubleType()) for c in data_modified.columns]
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
    output.head(1)

    final_data = output.select("features", "total_amount")
    final_data = spark.createDataFrame(final_data.limit(20).collect())

    train_data, test_data = final_data.randomSplit([0.8, 0.2])

    # train_data.describe().show()
    # test_data.describe().show()

    rf = RandomForestRegressor(featuresCol="features", labelCol="total_amount")
    rf_model = rf.fit(train_data)

    predictions = rf_model.transform(test_data)
    predictions.show(25)

    rfevaluator = RegressionEvaluator(
        predictionCol="prediction", labelCol="total_amount", metricName="rmse"
    )

    print("RMSE:", rfevaluator.evaluate(predictions))
