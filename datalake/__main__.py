from datalake import DataLake
from datetime import datetime


def get_json_data(start, count, increment=0):
    now = str(datetime.today().replace(microsecond=0))
    data = [{"id": i} for i in range(start, start + count)]
    return data


def run():
    datalake = DataLake()
    df1 = datalake.create_json_df(datalake.get_json_data(0, 40))
    df1.printSchema()
    df1.show(3)
    # df1.write.saveAsTable("mydb.test")
    # datalake.spark.sql("show tables in mydb").show()

    df2 = datalake.spark.sql("select * from example_hudi_table")  # df1.withColumn(
    #  "modified_timestamp",
    #   F.to_timestamp(F.col("modified_time"), "yyyy-MM-dd HH:mm:ss"),
    # ).drop("modified_time")
    df2.show(10, False)
    df2.printSchema()

    datalake.upsert(df2, "taxi")
    datalake.spark.sql("show tables").show()
    df2 = datalake.spark.sql("select * from taxi")  # df1.withColumn(
    #  "modified_timestamp",
    #   F.to_timestamp(F.col("modified_time"), "yyyy-MM-dd HH:mm:ss"),
    # ).drop("modified_time")
    df2.show(20, False)
    df2.printSchema()
    return
    # datalake.spark.sql("create table " + datalake.config["table_name"]).show(100, False)
    datalake.spark.sql("show tables").show(100, False)

    df2 = datalake.spark.read.format("org.apache.hudi").load(
        datalake.config["target"] + "/*"
    )

    df2.show(10, False)
    print(df2.count())

    df2 = datalake.create_json_df(get_json_data(2000000, 10, 1))
    df2.createOrReplaceTempView("data_to_delete_v")

    # join the incoming delete ids with the original table.
    # NOT USABLE SINCE HIVE NOT SET UP
    # df3 = datalake.spark.sql(
    #     "select a.* from "
    #     + datalake.config["table_name"]
    #     + " a, data_to_delete_v b where a.id = b.id"
    # )
    # print(df3.count())


if __name__ == "__main__":
    run()
