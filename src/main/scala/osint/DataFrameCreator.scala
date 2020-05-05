package osint

import org.apache.spark.ml.linalg.SQLDataTypes
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}

class DataFrameCreator (spark: SparkSession) {

	def tweetSeqToDF(tweets: Seq[Tweet]) : DataFrame = {
		val fields = new Array[StructField](4)
		fields(0) = StructField(DFColNames.DATE, StringType)
		fields(1) = StructField(DFColNames.ID, StringType)
		fields(2) = StructField(DFColNames.TWEET, StringType)
		fields(3) = StructField(DFColNames.FEATURES, SQLDataTypes.VectorType)
		val schema = StructType(fields)
		val sqlContext : SQLContext = spark.sqlContext
		sqlContext.createDataFrame(spark.sparkContext.parallelize(tweets.map(v => Row.fromTuple(v.date, v.id, v.text, v.features))), schema)
	}
}
