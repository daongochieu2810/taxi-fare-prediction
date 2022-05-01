from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

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

'''
Evaluation code
'''
def evaluateLGBM(spark, train_data, test_data):
    rf = RandomForestRegressor(featuresCol="features", labelCol="total_amount")
    rf_model = rf.fit(train_data)
    rfparamGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
            #  .addGrid(rf.maxDepth, [2, 5, 10])
            .addGrid(rf.maxBins, [10, 20, 40, 80, 100])
            #  .addGrid(rf.maxBins, [5, 10, 20])
            .addGrid(rf.numTrees, [5, 20, 50, 100, 500])
            #  .addGrid(rf.numTrees, [5, 20, 50])
            .build())

    rfevaluator = RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="rmse")

    # Create 5-fold CrossValidator
    rfcv = CrossValidator(estimator = rf,
                        estimatorParamMaps = rfparamGrid,
                        evaluator = rfevaluator,
                        numFolds = 5,
                        collectSubModels=True)

    rfcvModel = rfcv.fit(train_data)
    print(rfcvModel)
    rfpredictions = rfcvModel.transform(test_data)
    RMSE = rfevaluator.evaluate(rfpredictions)
    print('RMSE:', RMSE)

    rfpredictions.head()
    print(rfcvModel.subModels)
    true_value = [val.total_amount for val in rfpredictions.select('total_amount').collect()]
    print(true_value)

    predicted_value = [val.prediction for val in rfpredictions.select('prediction').collect()]
    true_value = [val.total_amount for val in rfpredictions.select('total_amount').collect()]
    plt.figure(figsize=(10,10))
    plt.scatter(true_value, predicted_value, c='crimson')
    plt.title("Random Forest Results")
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()

'''
Driver code for LGBM model
'''
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
