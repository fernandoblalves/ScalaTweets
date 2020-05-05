package osint

import org.apache.spark.sql.DataFrame

/**
  * Created by fernando on 10-03-2017.
  */
abstract class Classifier extends Serializable{

  def train(dataFrame: DataFrame) : Unit
  def classify(dataset : DataFrame) : DataFrame
  def saveModel() : Unit
  def loadModel() : Unit
  def toString: String
}
