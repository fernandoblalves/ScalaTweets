package osint

import org.apache.spark.sql.DataFrame
import org.apache.commons.math3.ml.clustering.Clusterable
import org.apache.commons.math3.ml.clustering.DBSCANClusterer
import org.apache.commons.math3.ml.distance._
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by fernando on 06-04-2017.
  */
class DBSCANCluster extends Clusterer {
  val pcaFeatures : Int = Osint.properties(Osint.PCA_FEATURES).toInt

  override def simpleClustering(dataFrame: DataFrame, outFile : String): Unit = {}

  override def recursiveClustering(dataFrame: DataFrame, outFile : String): Unit = {
//    var rdd : RDD[Vector] = dataFrameToRDD(dataFrame.select("features"))
//    var toReCluster : Seq[Vector] = null
//    do{
//      toReCluster = DBSCANCluster(rdd, eps, distanceMeasure)
//      rdd = Osint.spark.sparkContext.parallelize(toReCluster)
//    }while(toReCluster.nonEmpty)
  }

  def DBSCANCluster(rdd : RDD[Vector], eps : Double, measure: DistanceMeasure) : Seq[Vector] = {
/*    //val features = pca(rdd)
    val features = rdd
    val featureMap = features.zip(rdd).collect().toMap

    val points = features.collect().map(f => new ClusterableVector(f)).toList.asJava
    val min = 1

    val dbModel = new DBSCANClusterer[ClusterableVector](eps, min, measure)
    val clusters = dbModel.cluster(points)
    var toReCluster : Seq[Vector] = new ArrayBuffer[Vector]()

    println("num clusters: " + clusters.size)
    for(i <- 0 until clusters.size()){
      val c = clusters.get(i).getPoints.asScala.map(_.getVector)
      if(isFinal(c)){
        finalClusters :+= retrieveOriginalVectors(featureMap, c)
      }else
        toReCluster = toReCluster.union(c)
    }
    println("num final clusters " + finalClusters.size)
    println("to re-cluster " + toReCluster.size + "\n%%%%%%%%%%%%%%%%%%%%%%\n")
    toReCluster
*/null  }

  private def pca(input : RDD[Vector]) : RDD[Vector] = {
    val mat : RowMatrix = new RowMatrix(input)
    val pc : Matrix = mat.computePrincipalComponents(pcaFeatures)
    val projected : RowMatrix = mat.multiply(pc)

    projected.rows
  }

  def retrieveOriginalVectors(featureMap: Map[Vector, Vector], c: mutable.Buffer[Vector]): Seq[Vector] = {
    var cluster : Seq[Vector] = new ArrayBuffer[Vector]()
    for(v <- c){
      cluster :+= featureMap(v)
    }
    cluster
  }

  private class ClusterableVector(v : Vector) extends Clusterable with Serializable {
    def vector : Vector = v.copy
    def points : Array[Double] = vector.toArray

    override def getPoint : Array[Double] = points

    def getVector : Vector = vector
  }
}
