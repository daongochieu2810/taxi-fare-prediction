from datalake import DataLake
from merger import Merger
from pyspark.sql.functions import monotonically_increasing_id
import os


def get_month_str(month):
    if month < 10:
        return "0" + str(month)
    else:
        return str(month)


def preprocess(datalake):
    mergerInstance = Merger(datalake.spark)
    year = 2017
    month = 1
    while True:
        for root, _, files in os.walk(
            "./data/" + str(year) + "/" + get_month_str(month)
        ):
            for file in files:
                mergerInstance.load_taxi_data(os.path.join(root, file))
                mergerInstance.load_weather_data(
                    "./weather-data/" + str(year) + "_" + get_month_str(month) + ".txt"
                )
                mergerInstance.join()
                mergerInstance.merged_df.withColumn("id", monotonically_increasing_id())
                dfToSave = mergerInstance.merged_df.na.drop()
                datalake.upsert(dfToSave, "taxi_and_weather")
                break
        break
        if year == 2021 and month == 7:
            break
        if month < 12:
            month += 1
        else:
            month = 1
            year += 1


def run(datalake):
    datalake.execRawSql("select * from taxi_and_weather limit 10").show()


if __name__ == "__main__":
    datalake = DataLake()
    preprocess(datalake)
    # run(datalake)
