from pyspark.streaming import StreamingContext
import json


class Gateway:
    def __init__(self, spark) -> None:
        self.spark = spark
        self.sc = spark.sparkContext
        self.sc.setLogLevel("FATAL")
        self.ssc = StreamingContext(self.sc, 60)

    def connectConsumer(self, topic):
        """Connect to Kafka stream"""
        consumer = None
        print("\n" + "-" * 4 + " Waiting for records " + "-" * 4 + "\n")
        try:
            consumer = (
                self.spark.readStream.format("kafka")
                # TODO: Fill in host and port info
                .option("kafka.bootstrap.servers", "host1:port1,host2:port2")
                .option("subscribe", topic)
                .load()
            )
        except Exception as ex:
            print("Encountered error consuming from Kafka:\n   {}".format(str(ex)))
        finally:
            return consumer.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

    def consumeStream(self, topic):
        """Get stream from Kafka"""
        print("\nConsuming records from Kafka ({})...".format(topic))
        kafka_consumer = self.connectConsumer(topic)
        parsed = kafka_consumer.map(lambda record: json.loads(record[1]))
        parsed.count().map(lambda n: "Number of records: {}".format(n)).pprint()
        parsed.pprint()
        return parsed

    def processStream(self, parsed, topic, kafka_producer):
        """Process parsed stream and output to other topics"""
        cleaned_records = parsed.map(
            # TODO: Fill in mapping function
            lambda row: row
        ).filter(lambda row: row)

        cleaned_records.pprint()

        cleaned_records.foreachRDD(
            lambda rdd: self.publishStream(rdd, kafka_producer, topic)
        )

    # Write to lake storage
    def publishStream(self, rdd, kafka_producer, topic):
        """Iterate over RDD to push records to Kafka"""
        print("Pushed to topic {}\n".format(topic))
        records = rdd.collect()
        # for record in records:
        #     pyProducer.sendRecords(kafka_producer, topic, record)

    def startStreaming(self):
        """Start Spart processing"""
        self.ssc.start()
        self.ssc.awaitTermination()
