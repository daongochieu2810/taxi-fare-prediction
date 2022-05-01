from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.stat import Correlation

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

correlationCols = ["trip_distance", "passenger_count", "weather"]


finalPredictors = [
    "trip_distance",
    "passenger_count",
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


def generateCorrelationMatrix(df):
    correlationAssembler = VectorAssembler(
        inputCols=correlationCols, outputCol="features"
    )
    corr_output = correlationAssembler.transform(df)
    correlation_rdd = Correlation.corr(corr_output, "features")
    correlation_result = correlation_rdd.collect()[0][
        "pearson({})".format("features")
    ].values
    print(f"This is the correlation result: {correlation_result}")


# Manual forward selection functions

"""
Linear regression model with: Trip Distance as predictor
"""


def fwdSelectionDistance(df):
    assembler = VectorAssembler(
        inputCols=["trip_distance"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with only trip distance")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


"""
Linear regression model with: Passenger Count as predictor
"""


def fwdSelectionPassengerCount(df):
    assembler = VectorAssembler(
        inputCols=["passenger_count"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with only passenger_count")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


"""
Linear regression model with: Weather as predictor
"""


def fwdSelectionWeather(df):
    assembler = VectorAssembler(
        inputCols=["weather"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with only weather")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


"""
Linear regression model with: Trip Distance and Passenger Count as predictor
"""


def fwdSelectionDistancePassengerCount(df):
    assembler = VectorAssembler(
        inputCols=["passenger_count", "weather"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with trip distance and passenger count")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


"""
Linear regression model with: Trip Distance and Weather as predictor
"""


def fwdSelectionDistanceWeather(df):
    assembler = VectorAssembler(
        inputCols=["trip_distance", "weather"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with trip distance and weather")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


"""
Linear regression model with: Distance, Passenger Count, Weather as predictor
"""


def fwdSelectionDistancePassengerCountWeather(df):
    assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "weather"],
        outputCol="features",
    )
    output = assembler.transform(df)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    print(f"Results for linear regression with distance, passenger count, weather")
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print(f"This is the RMSE: {test_results.rootMeanSquaredError}")


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
        inputCols=finalPredictors,
        outputCol="features",
    )
    output = assembler.transform(data_modified)
    final_data = output.select("features", "total_amount")
    train_data, test_data = final_data.randomSplit([0.8, 0.2])
    lr = LinearRegression(
        featuresCol="features", labelCol="total_amount", regParam=0.0, solver="normal"
    )
    lr_model = lr.fit(train_data)
    test_results = lr_model.evaluate(test_data)
    test_results.residuals.show()
    print("RMSE: " + str(test_results.rootMeanSquaredError))
    print("R2: " + str(test_results.r2))
    # final_data.describe().show()
