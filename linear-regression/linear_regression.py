import findspark
findspark.init("D:\spark-3.2.1-bin-hadoop3.2")
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
import random
from datetime import datetime
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("lr_example").getOrCreate()
data = spark.read.csv("data/dummy_data.csv", inferSchema=True, header=True)

# create a dummy weather column
data_modified = data.withColumn("weather", F.round(F.rand() * 10))

def getTripDuration(datetime_start, datetime_end):
    try:
        result = (datetime.strptime(datetime_end, "%Y-%m-%d %H:%M:%S") - datetime.strptime(datetime_start, "%Y-%m-%d %H:%M:%S")).total_seconds()
        return result
    except Exception:
        return 0

tripDurationFunction = F.udf(getTripDuration, DoubleType())
data_modified = data_modified.withColumn("trip_duration", tripDurationFunction("lpep_pickup_datetime", "lpep_dropoff_datetime"))
assembler = VectorAssembler(inputCols=['PULocationID','DOLocationID','passenger_count','trip_distance','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge', 'total_amount','payment_type','trip_type','congestion_surcharge','weather','trip_duration'], outputCol="features")
output = assembler.transform(data_modified)
final_data = output.select("features", "total_amount")
train_data, test_data = final_data.randomSplit([0.7,0.3])
lr = LinearRegression(labelCol="total_amount")
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)
test_results.residuals.show()
test_results.rootMeanSquaredError
test_results.r2
final_data.describe().show()
