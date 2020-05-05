package osint

import java.io.{BufferedWriter, FileWriter}
import java.util.logging.{Level, Logger}

/**
  * Created by fernando on 25-03-2017.
  */
class ModelMetricsAverage(data : ModelMetrics*) {
  val size : Int = data.length

  var truePositiveRateArray : Array[Double] = new Array[Double](size)
  var trueNegativeRateArray : Array[Double] = new Array[Double](size)
  var falsePositiveRateArray : Array[Double] = new Array[Double](size)
  var falseNegativeRateArray : Array[Double] = new Array[Double](size)
  var accuracyArray : Array[Double] = new Array[Double](size)
  var precisionArray : Array[Double] = new Array[Double](size)
  var recallArray : Array[Double] = new Array[Double](size)
  var tpTNAverageArray : Array[Double] = new Array[Double](size)
  var fMeasureArray : Array[Double] = new Array[Double](size)
  var gMeanArray : Array[Double] = new Array[Double](size)

  for(i <- 0 until size){
    val result = data(i)
    truePositiveRateArray(i) = result.truePositiveRate
    trueNegativeRateArray(i) = result.trueNegativeRate
    falsePositiveRateArray(i) = result.falsePositiveRate
    falseNegativeRateArray(i) = result.falseNegativeRate
    accuracyArray(i) = result.accuracy
    precisionArray(i) = result.precision
    recallArray(i) = result.recall
    tpTNAverageArray(i) = result.tpTNAverage
    fMeasureArray(i) = result.fMeasure
    gMeanArray(i) = result.gMean
  }

  val (minTruePositiveRate, maxTruePositiveRate) : (Double, Double) = minMax(truePositiveRateArray)
  val (minTrueNegativeRate, maxTrueNegativeRate) : (Double, Double) = minMax(trueNegativeRateArray)
  val (minFalsePositiveRate, maxFalsePositiveRate) : (Double, Double) = minMax(falsePositiveRateArray)
  val (minFalseNegativeRate, maxFalseNegativeRate) : (Double, Double) = minMax(falseNegativeRateArray)
  val (minAccuracy, maxAccuracy) : (Double, Double) = minMax(accuracyArray)
  val (minPrecision, maxPrecision) : (Double, Double) = minMax(precisionArray)
  val (minRecall, maxRecall) : (Double, Double) = minMax(recallArray)
  val (minTPTNAverage, maxTPTNAverage) : (Double, Double) = minMax(tpTNAverageArray)
  val (minFMeasure, maxFMeasure): (Double, Double) = minMax(fMeasureArray)
  val (minGMean, maxGMean) : (Double, Double) = minMax(gMeanArray)

  val (averageTruePositiveRate, stdDevTruePositiveRate) : (Double, Double) = getMeanAndStdDev(truePositiveRateArray)
  val (averageTrueNegativeRate, stdDevTrueNegativeRate) : (Double, Double) = getMeanAndStdDev(trueNegativeRateArray)
  val (averageFalsePositiveRate, stdDevFalsePositiveRate) : (Double, Double) = getMeanAndStdDev(falsePositiveRateArray)
  val (averageFalseNegativeRate, stdDevFalseNegativeRate) : (Double, Double) = getMeanAndStdDev(falseNegativeRateArray)
  val (averageAccuracy, stdDevAccuracy) : (Double, Double) = getMeanAndStdDev(accuracyArray)
  val (averagePrecision, stdDevPrecision) : (Double, Double) = getMeanAndStdDev(precisionArray)
  val (averageRecall, stdDevRecall) : (Double, Double) = getMeanAndStdDev(recallArray)
  val (averageTPTNAverage, stdDevTPTNAverage) : (Double, Double) = getMeanAndStdDev(tpTNAverageArray)
  val (averageFMeasure, stdDevFMeasure) : (Double, Double) = getMeanAndStdDev(fMeasureArray)
  val (averageGMean, stdDevGMean) : (Double, Double) = getMeanAndStdDev(gMeanArray)

  private def getMean(data : Seq[Double]) : Double = {
    var sum = 0.0
    for (a <- data) {
      sum += a
    }
    sum / size
  }

  private def getVariance(data : Seq[Double]) : (Double, Double) = {
    val mean = getMean(data)
    var temp : Double = 0
    for (a <- data) {
      temp += (a - mean) * (a - mean)
    }
    (mean, temp / size)
  }

  private def getMeanAndStdDev(data : Seq[Double]) : (Double, Double) = {
    val (mean, variance) = getVariance(data)
    (setDecimals(mean), setDecimals(Math.sqrt(variance)))
  }

  private def minMax(a: Array[Double]) : (Double, Double) = {
    if (a.isEmpty) throw new java.lang.UnsupportedOperationException("array is empty")
    a.foldLeft((a(0), a(0)))
    { case ((min, max), e) => (setDecimals(math.min(min, e)), setDecimals(math.max(max, e)))}
  }

  def printResults(folds: Int, log : Logger): ModelMetricsAverage = {
    log.log(Level.INFO, "Classification results using " +folds + " folds")
    log.log(Level.INFO, "\t\t\t    Avg\t StdDev\t    Min\t    Max")
    log.log(Level.INFO, "True positive rate:\t" + averageTruePositiveRate + "\t"+stdDevTruePositiveRate +
      "\t" + minTruePositiveRate + "\t" + maxTruePositiveRate)
    log.log(Level.INFO, "True negative rate:\t" + averageTrueNegativeRate + "\t"+stdDevTrueNegativeRate +
      "\t" + minTrueNegativeRate + "\t" + maxTrueNegativeRate)
    log.log(Level.INFO, "False positive rate:\t" + averageFalsePositiveRate + "\t"+stdDevFalsePositiveRate +
      "\t" + minFalsePositiveRate + "\t" + maxFalsePositiveRate)
    log.log(Level.INFO, "False negative rate:\t" + averageFalseNegativeRate + "\t"+stdDevFalseNegativeRate +
      "\t" + minFalseNegativeRate + "\t" + maxFalseNegativeRate)
    log.log(Level.INFO, "Average TP/TN:\t" + averageTPTNAverage + "\t"+stdDevTPTNAverage +
      "\t" + minTPTNAverage + "\t" + maxTPTNAverage + "\n")
    log.log(Level.INFO, "Accuracy:\t\t" + averageAccuracy + "\t"+stdDevAccuracy +
      "\t" + minAccuracy + "\t" + maxAccuracy)
    log.log(Level.INFO, "Precision:\t\t" + averagePrecision + "\t"+stdDevPrecision +
      "\t" + minPrecision + "\t" + maxPrecision)
    log.log(Level.INFO, "Recall:\t\t"  + averageRecall + "\t"+stdDevRecall  +
      "\t" + minRecall + "\t" + maxRecall + "\n")
    log.log(Level.INFO, "F-measure:\t\t" + averageFMeasure + "\t"+stdDevFMeasure +
      "\t" + minFMeasure + "\t" + maxFMeasure)
    log.log(Level.INFO, "G-mean:\t\t" + averageGMean + "\t"+stdDevGMean +
      "\t" + minTruePositiveRate + "\t" + maxTruePositiveRate)

    this
  }

  def writeResults(folds: Int, path : String, log : Logger): ModelMetricsAverage = {
    val writer : BufferedWriter = new BufferedWriter(new FileWriter(path, true))

    writer.write(averageTruePositiveRate + "," + stdDevTruePositiveRate + "," + minTruePositiveRate + "," + maxTruePositiveRate + "\n")
    writer.write(averageTrueNegativeRate + "," + stdDevTrueNegativeRate + "," + minTrueNegativeRate + "," + maxTrueNegativeRate + "\n")
    writer.write(averageFalsePositiveRate + "," + stdDevFalsePositiveRate + "," + minFalsePositiveRate + "," + maxFalsePositiveRate + "\n")
    writer.write(averageFalseNegativeRate + "," + stdDevFalseNegativeRate + "," + minFalseNegativeRate + "," + maxFalseNegativeRate + "\n")
    writer.write(averageTPTNAverage + "," + stdDevTPTNAverage + "," + minTPTNAverage + "," + maxTPTNAverage + "\n")
    writer.write(averageAccuracy + "," + stdDevAccuracy + "," + minAccuracy + "," + maxAccuracy + "\n")
    writer.write(averagePrecision + "," + stdDevPrecision + "," + minPrecision + "," + maxPrecision + "\n")
    writer.write(averageRecall + "," + stdDevRecall  + "," + minRecall + "," + maxRecall + "\n")
    writer.write(averageFMeasure + "," + stdDevFMeasure + "," + minFMeasure + "," + maxFMeasure + "\n")
    writer.write(averageGMean + "," + stdDevGMean + "," + minTruePositiveRate + "," + maxTruePositiveRate + "\n")

    writer.flush()
    writer.close()
    log.log(Level.INFO, "Results written to " + path)
    this
  }

  private def setDecimals(num : Double) : Double = BigDecimal(num).setScale(5, BigDecimal.RoundingMode.HALF_UP).toDouble

  /*
  def median: Double = {
    val sorted = data.sortWith(_ < _)
    if (sorted.length % 2 == 0) return (sorted((sorted.length / 2) - 1) + sorted(sorted.length / 2)) / 2.0
    sorted(sorted.length / 2)
  }
  */
}


