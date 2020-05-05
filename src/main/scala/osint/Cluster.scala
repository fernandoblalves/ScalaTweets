package osint

import org.apache.spark.ml.linalg.Vector

import scala.collection.mutable

//(size : Int, id : Int, exemplar : Tweet, exemplarFeatures : Vector, centroid : Vector, elements : mutable.MutableList[Tweet])
class Cluster(var size : Int,
							var id : String,
							var exemplar : Tweet,
							var exemplarFeatures : Vector,
							var centroid : Vector,
							var elements : mutable.MutableList[Tweet],
							var timestamp : String,
							var lastUpdateDate: String,
							var wts : Double = 1,
							var version: Int = 1
						 ) extends Serializable{

	def incrementVersion():Unit = {
		version += 1
	}

	def setVersion(version: Int): Unit = {
		this.version = version
	}

	override def toString: String = {
		this.getClass.toString + ", " + id + " " + exemplar.toString + ": " + elements.map(_.toString).mkString("[", "; ", "]")
	}

	override def equals(o: Any): Boolean = {
		if(!o.isInstanceOf[Cluster]){
			return false
		}

		val cluster = o.asInstanceOf[Cluster]
		if(cluster.id.equals(this.id) &&
			cluster.size == this.size &&
			equalTweets(cluster, this)
		){
			return true
		}

		false
	}

	private def equalTweets(cluster: Cluster, cluster1: Cluster): Boolean = {
		val tweets = cluster.elements.sortWith((t1,t2) => t1.id.compareTo(t2.id) > 0)
		val tweets1 = cluster1.elements.sortWith((t1,t2) => t1.id.compareTo(t2.id) > 0)

		tweets.zip(tweets1) foreach { case (t, t1) =>
			if(!t.id.equals(t1.id)){
				return false
			}
		}

		true
	}
}
