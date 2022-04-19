import os
from pyspark.sql.types import Row
import requests
from urllib.parse import quote_plus
from datetime import datetime
from geopy.distance import geodesic 
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
    merged_df: Optional[DataFrame]

    @staticmethod
    def get_geocode(area: str) -> tuple[int, int]:
        area = quote_plus(area)
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={area}&key={api_key}"

        try:
            res = requests.get(url).json()["results"][0]["geometry"]["location"]
        except:
            res = {"lat": 0.0, "lng": 0.0}

        return (res["lat"], res["lng"])

    def __init__(self):
        self.spark = SparkSession.builder.appName("join").getOrCreate()

        # add spark dependencies
        self.spark.sparkContext.addPyFile(
            "https://pypi.python.org/packages/source/r/requests/requests-2.9.1.tar.gz"
        )
        self.spark.sparkContext.addPyFile(
            "https://pypi.python.org/packages/source/p/python-dotenv/python-dotenv-0.20.0.tar.gz"
        )

        self.weather_df = None
        self.taxi_df = None
        self.merged_df = None

    def load_weather_data(self, path: str):
        shared_get_geocode = self.spark.sparkContext.broadcast(self.get_geocode)

        def map_to_lat_lon(row: Row):
            loc = list(row.asDict().keys())[0]
            temp = {"data": row[loc]}
            lat, lon = shared_get_geocode.value(loc)
            temp["location"] = (lat, lon)
            return Row(**temp)

        self.weather_df = self.spark.read.json(path).rdd.map(map_to_lat_lon).toDF()

        def loc_to_loc_rows(row):
            lat, lon = row["location"]
            datas = row["data"]

            ret = []

            for data in datas:
                temp = data.asDict()
                temp["lat"] = lat
                temp["lon"] = lon

                ret.append(Row(**temp))

            return ret

        self.weather_df = self.weather_df.rdd.flatMap(loc_to_loc_rows).toDF()

    def load_taxi_data(self, path: str):
        self.taxi_df = self.spark.read.csv(path, header=True)

        def add_date(row):
            temp = row.asDict()
            try:
                temp["date"] = str(
                    datetime.fromisoformat(row["lpep_pickup_datetime"]).date()
                )
            except:
                temp["date"] = "1970-01-01"
            return Row(**temp)

        self.taxi_df = self.taxi_df.rdd.map(add_date).toDF(sampleRatio=0.01)

        shared_get_geocode = self.spark.sparkContext.broadcast(self.get_geocode)

        def map_to_lat_lon(row):
            lat, lon = shared_get_geocode.value(row["Zone"])
            return Row(**{"lat": lat, "lon": lon, "LocationID": row["LocationID"]})

        lookup_df = self.spark.read.csv(
            "sample-data/taxi_zone_lookup.csv", header=True
        ).select(["LocationID", "Zone"]).rdd.map(map_to_lat_lon).toDF()

        lookup_pu_df = (
            lookup_df.alias("lookup_pu_df")
            .withColumnRenamed("lat", "PU_lat")
            .withColumnRenamed("lon", "PU_lon")
        )

        self.taxi_df = self.taxi_df.join(
            lookup_pu_df, self.taxi_df.PULocationID == lookup_pu_df.LocationID
        )

        lookup_do_df = (
            lookup_df.alias("lookup_pu_df")
            .withColumnRenamed("lat", "DO_lat")
            .withColumnRenamed("lon", "DO_lon")
        )

        self.taxi_df = self.taxi_df.join(
            lookup_do_df, self.taxi_df.DOLocationID == lookup_do_df.LocationID
        )

    def join(self):
        assert self.weather_df is not None
        assert self.taxi_df is not None

        weather_list = self.weather_df.collect()

        shared_weather_list = self.spark.sparkContext.broadcast(weather_list)

        def augment_with_weather_data(row: Row):
            matched_row: list[Row] = list(filter(lambda w: w.date == row.date, shared_weather_list.value))

            if len(matched_row) == 0:
                return row

            min_row: Row = Row() 
            min_dist = float("inf")

            for other in matched_row:
                dist = geodesic((other.lat, other.lon), (row.PU_lat, row.PU_lon))
                if dist < min_dist:
                    min_dist = dist
                    min_row = other 

            return Row(**{**row.asDict(), **min_row.asDict()})

        self.merged_df = self.taxi_df.rdd.map(augment_with_weather_data).toDF(sampleRatio=0.01)
