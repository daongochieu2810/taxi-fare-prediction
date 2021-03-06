from pyspark.sql import SparkSession

# General Constants
HUDI_FORMAT = "org.apache.hudi"
TABLE_NAME = "hoodie.table.name"
RECORDKEY_FIELD_OPT_KEY = "hoodie.datasource.write.recordkey.field"
PRECOMBINE_FIELD_OPT_KEY = "hoodie.datasource.write.precombine.field"
OPERATION_OPT_KEY = "hoodie.datasource.write.operation"
BULK_INSERT_OPERATION_OPT_VAL = "bulk_insert"
UPSERT_OPERATION_OPT_VAL = "upsert"
BULK_INSERT_PARALLELISM = "hoodie.bulkinsert.shuffle.parallelism"
UPSERT_PARALLELISM = "hoodie.upsert.shuffle.parallelism"
HUDI_CLEANER_POLICY = "hoodie.cleaner.policy"
KEEP_LATEST_COMMITS = "KEEP_LATEST_COMMITS"
HUDI_COMMITS_RETAINED = "hoodie.cleaner.commits.retained"
PAYLOAD_CLASS_OPT_KEY = "hoodie.datasource.write.payload.class"
EMPTY_PAYLOAD_CLASS_OPT_VAL = "org.apache.hudi.common.model.EmptyHoodieRecordPayload"
# Partition Constants
NONPARTITION_EXTRACTOR_CLASS_OPT_VAL = "org.apache.hudi.hive.NonPartitionedExtractor"
MULIPART_KEYS_EXTRACTOR_CLASS_OPT_VAL = (
    "org.apache.hudi.hive.MultiPartKeysValueExtractor"
)
KEYGENERATOR_CLASS_OPT_KEY = "hoodie.datasource.write.keygenerator.class"
NONPARTITIONED_KEYGENERATOR_CLASS_OPT_VAL = (
    "org.apache.hudi.keygen.NonpartitionedKeyGenerator"
)
COMPLEX_KEYGENERATOR_CLASS_OPT_VAL = "org.apache.hudi.ComplexKeyGenerator"
PARTITIONPATH_FIELD_OPT_KEY = "hoodie.datasource.write.partitionpath.field"


class DataLake:
    config = {
        "primary_key": "id",
        "sort_key": "id",
    }

    def __init__(self):
        self.spark = (
            SparkSession.builder.appName("Hudi_Data_Processing_Framework")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config(
                "spark.jars.packages",
                "org.apache.hudi:hudi-spark3.0.3-bundle_2.12:0.10.1,org.apache.spark:spark-avro_2.12:3.0.3,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1",
            )
            .config("spark.sql.hive.convertMetastoreParquet", "false")
            .config("spark.sql.hive.metastore.sharedPrefixes", "org.apache.derby")
            .config("hive.metastore.uris", "thrift://localhost:9083")
            .config("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
            .enableHiveSupport()
            .getOrCreate()
        )
        self.spark.sparkContext.addPyFile("./datalake/merger.py")
        self.spark.sparkContext.addPyFile("./datalake/lgbm.py")
        self.spark.sparkContext.addPyFile("./datalake/linear_regression.py")
        self.spark.sparkContext.addPyFile("./datalake/mlp.py")

        self.spark.sql("create database if not exists maindb")
        self.spark.sql("use maindb")

    def readAllFromTable(self, table_name):
        return self.spark.sql("select * from " + table_name)

    def execRawSql(self, sqlScript):
        return self.spark.sql(sqlScript)

    def upsert(self, df, table_name):
        config = self.config
        (
            df.write.format(HUDI_FORMAT)
            .option(PRECOMBINE_FIELD_OPT_KEY, config["sort_key"])
            .option(RECORDKEY_FIELD_OPT_KEY, config["primary_key"])
            .option(TABLE_NAME, table_name)
            .option(OPERATION_OPT_KEY, BULK_INSERT_OPERATION_OPT_VAL)
            .option(BULK_INSERT_PARALLELISM, 3)
            .option(
                KEYGENERATOR_CLASS_OPT_KEY, NONPARTITIONED_KEYGENERATOR_CLASS_OPT_VAL
            )
            .mode("append")
            .saveAsTable("maindb." + table_name)
        )
