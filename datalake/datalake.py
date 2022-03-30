from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from datalake.gateway import Gateway

# input_df = spark.createDataFrame(
#     [
#         ("100", "2015-01-01", "2015-01-01T13:51:39.340396Z"),
#         ("101", "2015-01-01", "2015-01-01T12:14:58.597216Z"),
#         ("102", "2015-01-01", "2015-01-01T13:51:40.417052Z"),
#         ("103", "2015-01-01", "2015-01-01T13:51:40.519832Z"),
#         ("104", "2015-01-02", "2015-01-01T12:15:00.512679Z"),
#         ("105", "2015-01-02", "2015-01-01T13:51:42.248818Z"),
#     ],
#     ("id", "creation_date", "last_update_time"),
# )


class DataLake:
    def __init__(self):
        self.spark = (
            SparkSession.builder.appName("Hudi_Data_Processing_Framework")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.hive.convertMetastoreParquet", "false")
            .config(
                "spark.jars.packages",
                "org.apache.hudi:hudi-spark3.0.3-bundle_2.12:0.10.1,org.apache.spark:spark-avro_2.12:3.0.3,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1",
            )
            .getOrCreate()
        )

        self.hudi_options = {
            # ---------------DATA SOURCE WRITE CONFIGS---------------#
            "hoodie.table.name": "hudi_test",
            "hoodie.datasource.write.recordkey.field": "id",
            "hoodie.datasource.write.precombine.field": "last_update_time",
            "hoodie.datasource.write.partitionpath.field": "creation_date",
            "hoodie.datasource.write.hive_style_partitioning": "true",
            "hoodie.upsert.shuffle.parallelism": 1,
            "hoodie.insert.shuffle.parallelism": 1,
            "hoodie.consistency.check.enabled": True,
            "hoodie.index.type": "BLOOM",
            "hoodie.index.bloom.num_entries": 60000,
            "hoodie.index.bloom.fpp": 0.000000001,
            "hoodie.cleaner.commits.retained": 2,
        }

        self.gateway = Gateway(self.spark)

    def insert(self, df):
        (
            df.write.format("org.apache.hudi")
            .options(**self.hudi_options)
            .mode("append")
            .save("/tmp/hudi_test")
        )

    def update(self, df):
        update_df = df.limit(1).withColumn(
            "last_update_time", lit("2016-01-01T13:51:39.340396Z")
        )
        (
            update_df.write.format("org.apache.hudi")
            .options(**self.hudi_options)
            .mode("append")
            .save("/tmp/hudi_test")
        )

    def read(self):
        output_df = self.spark.read.format("org.apache.hudi").load("/tmp/hudi_test/*/*")
        output_df.show()
