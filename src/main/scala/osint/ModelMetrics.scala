package osint

import java.io.{BufferedWriter, FileWriter}
import java.util.logging.{Level, Logger}

import org.apache.spark.sql.DataFrame

/**
  * Created by fernando on 14-03-2017.
  */
class ModelMetrics(data: DataFrame) {
  val positive = 1.0
  val negative = 0.0

  var truePositive : Double = 0
  var falsePositive : Double = 0
  var trueNegative : Double = 0
  var falseNegative : Double = 0

  var positiveLabel : Double = 0
  var negativeLabel : Double = 0

  var truePositiveRate : Double = 0
  var falsePositiveRate : Double = 0
  var trueNegativeRate : Double = 0
  var falseNegativeRate : Double = 0

  var accuracy : Double = 0
  var precision : Double = 0
  var recall : Double = 0
  var tpTNAverage : Double = 0
  var fMeasure : Double = 0
  var gMean : Double = 0

  val labelsAndScore: DataFrame = data.select("label", "prediction")

  var label : Double = 0
  var prediction: Double = 0

  for (s <- labelsAndScore.collect) {
    //label is int
    label = if (s.get(0).asInstanceOf[Int] > 0) 1.0 else 0.0
    //prediction is double
    prediction = if (s.get(1).asInstanceOf[Double] > 0.0) 1.0 else 0.0

    if (label == positive && prediction == positive)
      truePositive += 1
    else if (prediction == positive && label == negative)
      falseNegative += 1
    else if (prediction == negative && label == negative)
      trueNegative += 1
    else //prediction == negative && label == positive
      falsePositive += 1

    if (label == 1.0)
      positiveLabel += 1
    else
      negativeLabel += 1
  }

  val numberOfDecimals : Int = 5

  //these are confirmed
  if(positiveLabel > 0) {
    truePositiveRate = BigDecimal(truePositive / positiveLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
    falseNegativeRate = BigDecimal(falseNegative / negativeLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
  }
  if(negativeLabel > 0) {
    trueNegativeRate = BigDecimal(trueNegative / negativeLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
    falsePositiveRate = BigDecimal(falsePositive / positiveLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  if(positiveLabel + negativeLabel > 0)
    accuracy = BigDecimal((truePositive + trueNegative) / (positiveLabel + negativeLabel)).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
  if(truePositive + falsePositive > 0)
    precision = BigDecimal(truePositive / (truePositive + falsePositive)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
  if(truePositive + falseNegative > 0)
    recall = BigDecimal(truePositive / (truePositive + falseNegative)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble

  //these not really
  tpTNAverage = BigDecimal((truePositiveRate + trueNegativeRate) / 2).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
  if(precision + truePositiveRate > 0)
    fMeasure = BigDecimal(2 * (precision * truePositiveRate) / (precision + truePositiveRate)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
  gMean = BigDecimal(Math.sqrt(truePositiveRate * trueNegativeRate)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble

  def writeResults(path : String, numFeatures: Int) : ModelMetrics = {
    val writer : BufferedWriter = new BufferedWriter(new FileWriter(path, true))
    writer.write(numFeatures.toString + "," + this.toString + "\n")
    writer.close()

    this
  }

  def printResults(log : Logger): ModelMetrics = {
    log.log(Level.INFO, "True positive rate:\t" + truePositiveRate)
    log.log(Level.INFO, "True negative rate:\t" + trueNegativeRate)
    log.log(Level.INFO, "False positive rate:\t" + falsePositiveRate)
    log.log(Level.INFO, "False negative rate:\t" + falseNegativeRate)
    log.log(Level.INFO, "Average TP/TN:\t" + tpTNAverage + "\n")
    log.log(Level.INFO, "Accuracy:\t\t" + accuracy)
    log.log(Level.INFO, "Precision:\t\t" + precision)
    log.log(Level.INFO, "Recall:\t\t" + recall+"\n")
    log.log(Level.INFO, "F-measure:\t\t" + fMeasure)
    log.log(Level.INFO, "G-mean:\t\t" + gMean)

    this
  }

  override def toString: String = truePositiveRate + "," +
    trueNegativeRate + "," +
    falsePositiveRate + "," +
    falseNegativeRate + "," +
    tpTNAverage + "," +
    accuracy + "," +
    precision + "," +
    recall + "," +
    fMeasure + "," +
    gMean
}