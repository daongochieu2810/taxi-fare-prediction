import os
from pyspark.sql.types import Row
import requests
from urllib.parse import quote_plus
from typing import Optional
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from dotenv import load_dotenv  # type: ignore

load_dotenv()
api_key = os.getenv("MAPS_API_KEY")


class Merger:

    spark: SparkSession
    weather_df: Optional[DataFrame]
    taxi_df: Optional[DataFrame]

    @staticmethod
    def get_geocode(area: str) -> tuple[int, int]:
        area = quote_plus(area)
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={area}&key={api_key}"
        res = requests.get(url).json()["results"][0]["geometry"]["location"]

        return (res["lat"], res["lng"])

    def __init__(self):
        self.spark = SparkSession.builder.appName("join").getOrCreate()

        # add spark dependencies
        self.spark.sparkContext.addPyFile(
            "https://pypi.python.org/packages/source/r/requests/requests-2.9.1.tar.gz"
        )

        self.weather_df = None
        self.taxi_df = None

    def load_weather_data(self, path: str):
        def map_to_lat_lon(tup):
            loc, data = tup
            lat, lon = self.get_geocode(loc)
            return ((lat, lon), data)

        data = list(
            map(
                map_to_lat_lon, self.spark.read.json(path).collect()[0].asDict().items()
            )
        )
        self.weather_df = self.spark.createDataFrame(data, schema=["location", "data"])

        def loc_to_loc_rows(row):
            lat, lon = row[0]
            datas = row[1]

            ret = []

            for data in datas:
                temp = data.asDict()
                temp["lat"] = lat
                temp["lon"] = lon

                ret.append(Row(**temp))

            return ret

        self.weather_df = self.weather_df.rdd.flatMap(loc_to_loc_rows).toDF()

    def load_taxi_data(self, path: str):
        self.taxi_df = self.spark.read.csv(path)

    def join(self):
        assert self.weather_df is not None
        assert self.taxi_df is not None
