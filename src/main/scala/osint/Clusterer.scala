package osint

//import org.apache.commons.math3.ml.distance.{CanberraDistance, ChebyshevDistance, DistanceMeasure, EarthMoversDistance, EuclideanDistance, ManhattanDistance}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

//import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by fernando on 06-04-2017.
  */
abstract class Clusterer {

  val numFeatures : Int = Osint.properties(Osint.NUM_FEATURES).toInt
  var finalClusters : Seq[(Vector, Seq[Vector])] = new ArrayBuffer[(Vector, Seq[Vector])]()
  val rwoThreshold : Int = Osint.properties(Osint.ROW_THRESHOLD).toInt
  val tableThreshold : Int = Osint.properties(Osint.TABLE_THRESHOLD).toInt
  val eps : Int = Osint.properties(Osint.EPS).toInt
  //val distanceMeasure : DistanceMeasure = getDistanceMeasure
  val interesectMinThreshold : Double = 2.0/3.0
  var reclusteringTimes : Int = 0

  //def cluster(dataFrame: DataFrame) : Unit
  def recursiveClustering(dataFrame: DataFrame, outFile : String) : Unit
  def simpleClustering(dataFrame: DataFrame, outFile : String) : Unit

  def isFinal(cluster : Seq[Vector]) : Boolean = {
    getIntraClusterSimilarity(cluster) >= interesectMinThreshold
  }

  def getIntraClusterSimilarity(cluster : Seq[Vector]) : Double = {
    try {
      val numCommonWords: Double = cluster.map(_.toSparse.indices).reduce((a, b) => a.intersect(b)).length
      val smallestTweetSize: Double = cluster.map(_.toSparse.indices.length).min
      val result: Double = numCommonWords / smallestTweetSize
      println(numCommonWords + " ; " + smallestTweetSize + " ; " + result + ";" + interesectMinThreshold + ": " + (result > interesectMinThreshold))
      result
    }catch {
      case _ : Exception => cluster foreach println
        0.0
    }
  }

/*  //TODO this methodology only works for TF-IDF; for w2v we will need another
  def isFinal(cluster : Seq[Vector]): Boolean = {
    val max = cluster.map(_.toSparse.indices.length).max
    cluster.map(_.toArray).reduceLeft((v1, v2) => (v1, v2).zipped.map(_ + _)).count(_ > 0) < tableThreshold
  }

  //TODO works with w2v?
  def isFinal2(cluster : Seq[Vector]): Boolean = {
    var sum = 0
    val sparses = cluster.map(_.toSparse)
    val globalIndices = sparses.map(_.indices).reduce((a,b) => a.intersect(b))
    for (i <- globalIndices){
      val bool = cluster.map(_.toArray).map(_(i)).forall(_ == cluster.head.toArray(i))
      if (bool)
        sum += 1
    }

    sum / cluster.map(_.toSparse.indices.length).min > 0.8
  }

  def isFinal3(cluster : Seq[Vector]) : Boolean = {
    val union = cluster.map(_.toSparse.indices).reduce((a,b) => a.union(b))
    val sumArray : Array[Int] = new Array[Int](union.length)
    val denseArray = cluster.map(_.toDense.toArray)
    var i = 0
    for(index <- union){
      for(array <- denseArray){
        if(array(index) != 0.0){
          sumArray(i) += 1
        }
      }
      i += 1
    }

    val longestTweet = cluster.map(_.toSparse.indices.length).max
    sumArray.map(a => square(a/cluster.size)).sum / longestTweet > 0.9
    //TODO change to cube
  }

  def isFinal4(cluster : Seq[Vector]) : Boolean = {
    val union = cluster.map(_.toSparse.indices).reduce((a,b) => a.union(b)).length
    val intersection = cluster.map(_.toSparse.indices).reduce((a,b) => a.intersect(b)).length
    intersection/union > tableThreshold
  }
/*
  def isFinal5(cluster : Seq[Vector]) : Boolean = {
    val intersection = cluster.map(_.toSparse.indices).reduce((a,b) => a.intersect(b)).length
    val shortestTweet = cluster.map(_.toSparse.indices.length).min
    intersection == shortestTweet - adjustment
  }
*/
  def square(a : Double) : Double = a * a
  def cube(a : Double) : Double = a * a * a

  private def calculateBags(cluster : Seq[Vector]): Boolean = {
    //cluster foreach {a => a.toSparse.values.con}
    val rowMatrix = new RowMatrix(Osint.sparkContext.parallelize(cluster))
    val similarities = rowMatrix.columnSimilarities()
    var sum : Double = 0.0
    val arr = similarities.entries.map{case MatrixEntry(row: Long, col:Long, sim:Double) => Array(row,col,sim)}.collect()
    for(a <- arr)
      sum += a(2)
    val avg : Double = sum / cluster.size
    println("cluster size: " + cluster.size)
    println("sum: " + sum)
    println("avg cluster similarity: " + avg.toString)
    println()
    //val transformedRDD = similarities.entries.map{case MatrixEntry(row: Long, col:Long, sim:Double) => Array(row,col,sim).mkString(",")}
    //transformedRDD.collect() foreach println
    val longestTweet = cluster.map(_.toSparse.values.length).max
    println("longest: " + longestTweet)
    avg > longestTweet
  }

  private def getDistanceMeasure : DistanceMeasure = {
    val measureName = Osint.properties(Osint.DISTANCE_MEASURE)
    var measure : DistanceMeasure = null
    measureName match {
      case "canberra" => measure = new CanberraDistance()
      case "earth_movers" => measure = new EarthMoversDistance()
      case "euclidean" => measure = new EuclideanDistance()
      case "chebyshev" => measure = new ChebyshevDistance()
      case "manhattan" => measure = new ManhattanDistance()
    }
    measure
  }
*/
  def printClusters(data : DataFrame, clustersFile : String, statsFile : String) : Unit = {
    val tweetMap = data.select("tweet", "features", "id", "date").rdd.collect().map(r => (Vectors.dense(r.getAs[org.apache.spark.ml.linalg.SparseVector](1).toArray), r.getAs[String](3) + "\t"  + r.getAs[String](2) + "\t" + r.getAs[String](0))).toMap
    var i = 1

    val clusterWriter = new java.io.BufferedWriter(new java.io.FileWriter(Osint.properties(Osint.SAVE_MODEL_PATH) + clustersFile, true))
    for((exemplar, cluster) <- finalClusters){
      println("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
      clusterWriter.write("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

      println("Cluster " + i + "\n")
      clusterWriter.write("Cluster " + i + "\n")

      println(tweetMap(exemplar))
      clusterWriter.write(tweetMap(exemplar) + "\n")

      println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
      clusterWriter.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

      for(features <- cluster){
        println(tweetMap(features))
        clusterWriter.write(tweetMap(features) + "\n")
      }

      i += 1
    }

    clusterWriter.close()

    val numTweets = finalClusters.map(_._2.length).sum
    val numClusters = finalClusters.length
    val allIntraCs = finalClusters.map(c => getIntraClusterSimilarity(c._2))
    val intraCSAvg = allIntraCs.sum / allIntraCs.length
    val (_, intraCSSTDDev) = Statistics.getMeanAndStdDev(allIntraCs)
    var jaccard = 0.0
    var largestJaccard = 0.0
    var smallestJaccard = Double.MaxValue

    for(i <- finalClusters.indices){
      for(j <- i + 1 until finalClusters.size){
        jaccard = jaccardDistance(finalClusters(i)._2, finalClusters(j)._2)
        if(jaccard > largestJaccard)
          largestJaccard = jaccard
        if(jaccard < smallestJaccard)
          smallestJaccard = jaccard
      }
    }

    val writer = new java.io.BufferedWriter(new java.io.FileWriter(Osint.properties(Osint.SAVE_MODEL_PATH) + statsFile, true))
    writer.write(
      numTweets + "\t" +
      numClusters + "\t" +
      Statistics.setDecimals((numClusters.toDouble * 100.0) / numTweets.toDouble, 2) + "\t" +
      reclusteringTimes + "\t" +
      Statistics.setDecimals(intraCSAvg, 2) + "\t" +
      Statistics.setDecimals(intraCSSTDDev, 2) + "\t" +
      Statistics.setDecimals(smallestJaccard, 2) + "\t" +
      Statistics.setDecimals(largestJaccard, 2) + "\n")
    writer.close()
  }

  private def jaccardDistance(v1 : Seq[Vector], v2 : Seq[Vector]) : Double = {
    val i1 = v1.map(_.toSparse.indices).reduce((a, b) => a.intersect(b))
    val i2 = v2.map(_.toSparse.indices).reduce((a, b) => a.intersect(b))
    i1.intersect(i2).length.toDouble / i1.union(i2).length.toDouble
  }

  def dataFrameToRDD(dataFrame : DataFrame) : RDD[Vector] = {
    dataFrame.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.SparseVector](0)).map(r => Vectors.dense(r.toArray))
  }
}