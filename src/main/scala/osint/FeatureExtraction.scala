package osint

import org.apache.spark.sql.DataFrame

/**
  * Created by fernando on 10-03-2017.
  */
abstract class FeatureExtraction {

  def transform(dataset : DataFrame) : DataFrame
	def train(dataset : DataFrame) : Unit
	def loadModel() : Unit
	def saveModel() : Unit

}
