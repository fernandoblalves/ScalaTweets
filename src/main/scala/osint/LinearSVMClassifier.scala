package osint

import java.util.logging.{Level, Logger}

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
/**
	* Created by fernando on 10-03-2017.
	*/
class LinearSVMClassifier extends Classifier with Serializable{

	@transient lazy val log: Logger = Logger.getLogger(getClass.getName)

	private var model : SVMModel = _
	val step : Double = Osint.properties(Osint.SVM_STEPSIZE).toDouble
	val c : Double = Osint.properties(Osint.SVM_C).toDouble

	override def train(data : DataFrame): Unit = {
		val rdd = dataFrameToRDD(data)
		// Run training algorithm to build the model
		val numIter : Int = 100
		log.log(Level.INFO, "Initiating SVM training with parameters: C="+c+", step="+step)
		model = SVMWithSGD.train(rdd, numIterations = numIter, stepSize = step, regParam = c)
		log.log(Level.INFO, "Model training finished")

		// Clear the default threshold.
		model.clearThreshold()
	}

	override def classify(data : DataFrame): DataFrame = {
		log.log(Level.INFO, "Converting DataFrame to RDD")
		val rdd = dataFrameToRDD(data)
		log.log(Level.INFO, "Conversion finished; beginning classification")
		// Compute raw scores on the test set.
		val predictions = rdd.map(point => model.predict(point.features))
		log.log(Level.INFO, "Classification finished; Transforming RDD to DataFrame")

		val sqlContext : SQLContext = Osint.spark.sqlContext
		val tupleRDD = data.rdd.zip(predictions).map(t => Row.fromSeq(t._1.toSeq ++ Seq(t._2)))
		sqlContext.createDataFrame(tupleRDD, data.schema.add("prediction", "Double"))

		//TODO this should work it doesn't since this "withColumn" method seems to be applicable only to add
		// new columns using information from the same dataframe; therefore I am using the horrible rdd conversion
		//TODO perhaps this can be remade with an udf that classifies the points
		//val sqlContext : SQLContext = Osint.spark.sqlContext
		//import sqlContext.implicits._
		//val plDF = predictions.toDF("predictions")
		//data.withColumn("prediction", plDF.col("predictions"))
	}

	private def classify: UserDefinedFunction = udf{vector: Vector => model.predict(org.apache.spark.mllib.linalg.Vectors.dense(vector.toArray))}
	override def saveModel(): Unit = {
		val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
		//TODO this does not overwrite existing models
		model.save(Osint.sparkContext, "file://" + savePath + "SVMModel_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
	}

	override def loadModel(): Unit = {
		val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
		model = SVMModel.load(Osint.sparkContext, "file://" + savePath+"SVMModel_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
	}

	override def toString: String = {
		"SVM_"+ c + "_" + step
	}

	private def dataFrameToRDD(dataFrame : DataFrame) : RDD[LabeledPoint] = {
		val rddMl = dataFrame.select("label", "features").rdd.map(r => (r.getInt(0).toDouble, r.getAs[org.apache.spark.ml.linalg.SparseVector](1)))
		rddMl.map(r => new LabeledPoint(r._1, org.apache.spark.mllib.linalg.Vectors.dense(r._2.toArray)))
	}
}
