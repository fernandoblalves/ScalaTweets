package osint

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object ClusterUtils {

	private final val interesectMinThreshold : Double = 2.0/3.0
	private final val dtf: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")

	def isFinal(cluster : Seq[Vector]) : (Boolean, Double) = {
		val wts = getIntraClusterSimilarity(cluster)
		(wts >= interesectMinThreshold, wts)
	}

	def isFinal(cluster: DataFrame): (Boolean, Double) = {
		val features = cluster.collect().map(_.getAs[Vector](DFColNames.FEATURES))
		isFinal(features)
	}

	def getIntraClusterSimilarity(cluster : Seq[Vector]) : Double = {
		try {
			val numCommonWords: Double = cluster.map(_.toSparse.values).reduce((a, b) => a.intersect(b)).length
			val smallestTweetSize: Double = cluster.map(_.toSparse.indices.length).min
			if(numCommonWords == 0 || smallestTweetSize == 0)
				return 0.0
			//TODO smallest tweet 1?
			val result: Double = numCommonWords / smallestTweetSize
			//println(numCommonWords + " ; " + smallestTweetSize + " ; " + result + ";" + interesectMinThreshold + ": " + (result > interesectMinThreshold))
			result
		}catch {
			case _ : Exception => cluster foreach println
				0.0
		}
	}

	def jaccardDistance(v1 : Seq[Vector], v2 : Seq[Vector]) : Double = {
		val i1 = v1.map(_.toSparse.indices).reduce((a, b) => a.intersect(b))
		val i2 = v2.map(_.toSparse.indices).reduce((a, b) => a.intersect(b))
		i1.intersect(i2).length.toDouble / i1.union(i2).distinct.length.toDouble
	}

	def dataFrameToRDD(dataFrame : DataFrame) : RDD[Vector] = {
		dataFrame.select(DFColNames.FEATURES).rdd.map(_.getAs[org.apache.spark.ml.linalg.SparseVector](0)).map(r => Vectors.dense(r.toArray))
	}

	def getEarliestDate(tweets : Seq[Tweet]) : String = {
		var earliest: LocalDateTime = LocalDateTime.parse("2050/01/01 07:40:23", dtf)

		for (tweet <- tweets) {
			val df : LocalDateTime = LocalDateTime.parse(tweet.date, dtf)

			if (df.isBefore(earliest))
				earliest = df
		}

		earliest.format(dtf)
	}

	def getOldestTweet(tweets: Seq[Tweet]): String = {
		var oldest: LocalDateTime = LocalDateTime.parse("1999/01/01 07:40:23", dtf)

		for (tweet <- tweets) {
			val df : LocalDateTime = LocalDateTime.parse(tweet.date, dtf)

			if (df.isAfter(oldest))
				oldest = df
		}

		oldest.format(dtf)
	}

	def getExemplarFeatures(centroid : Vector, cluster : Seq[Vector]): Vector = {
		var min = Double.MaxValue
		var dist = 0.0
		var exemplar : Vector = null
		for(v <- cluster){
			dist = Vectors.sqdist(centroid, v)
			if(dist < min){
				min = dist
				exemplar = v
			}
		}
		exemplar
	}

	def getExemplarTweet(centroid: Vector, tweets: Seq[Tweet]): Tweet = {
		var min = Double.MaxValue
		var dist = 0.0
		var exemplar: Tweet = null
		tweets foreach{t =>
			dist = Vectors.sqdist(centroid, t.features)
			if(dist < min){
				min = dist
				exemplar = t
			}
		}
		exemplar
	}

	def solveNewExemplar(cluster: Cluster): Cluster = {
		val exemplarTweet = ClusterUtils.getExemplarTweet(cluster.centroid, cluster.elements)

		if (exemplarTweet.id.equals(cluster.exemplar.id)) {
			return cluster
		}
		cluster.exemplar = exemplarTweet
		cluster.exemplarFeatures = exemplarTweet.features
		//this solves the issue of the unremovable clusters
		//cluster.id = exemplarTweet.id

		cluster
	}

	def printClusters(clusters: Seq[Cluster]) : Unit = {
		var i = 1
		var tweetCounter = 0

		for (cluster <- clusters) {
			println("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

			println("Cluster " + i + "\n")

			val exemplarText = cluster.exemplar.text
			println(exemplarText)

			println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

			for (tweet <- cluster.elements) {
				tweetCounter += 1
				println(tweet.text)
			}

			i += 1
		}
	}
}
