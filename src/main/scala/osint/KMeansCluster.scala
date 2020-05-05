package osint
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

/**
	* Created by fernando on 06-04-2017.
	*/
class KMeansCluster extends Clusterer{

	private var prevK : Int = 0
	private var prevTweets : Long = 0
	private var anyFinal : Boolean = true

	override def recursiveClustering(dataFrame: DataFrame, outFile : String): Unit = {
		val dataFrameCopy = dataFrame
		dataFrame.persist(StorageLevel.MEMORY_ONLY)
		cluster(dataFrame)
		printClusters(dataFrame, outFile + "_clusters", outFile + "_stats")
		finalClusters = new ArrayBuffer[(Vector, Seq[Vector])]()

//		dataFrameCopy.persist(StorageLevel.MEMORY_ONLY)
//		newCluster(dataFrameCopy, 2)
//		printClusters(dataFrameCopy, outFile + "_clusters_new", outFile + "_stats_new")
	}

	@tailrec
	private def cluster(dataFrame : DataFrame): Unit = {
		if(dataFrame == null || dataFrame.count() == 0)
			return

		var numClusters = 2
		var kModel : KMeansModel = null
		var kMeans : KMeans = null
		var SSE, currentSSE : Double = Double.MaxValue
		var clusteringResult : DataFrame = null

		if(prevTweets  == dataFrame.count()) {
			anyFinal = false
			numClusters = prevK
		}
		prevTweets = dataFrame.count()

		println("Num clusters: " + numClusters)
		val numIterations = 50
		do{
			kMeans = new KMeans().setK(numClusters).setMaxIter(numIterations).setFeaturesCol("features")
			kModel = kMeans.fit(dataFrame)
			kModel.setPredictionCol("prediction")
			clusteringResult = kModel.transform(dataFrame)

			SSE = BigDecimal(kModel.computeCost(clusteringResult)).setScale(5, BigDecimal.RoundingMode.HALF_UP).toDouble

			if((SSE > 0.0 && SSE < currentSSE && SSE != currentSSE) || (!anyFinal && numClusters <= prevK)) {
				currentSSE = SSE
			}
			numClusters += 1
		} while(SSE == currentSSE && numClusters < dataFrame.count())

		prevK = numClusters

		val centroids = kModel.clusterCenters
		val tempSplits = clusteringResult.collect().map(r => (r.getAs[Int]("prediction"), new Tweet(r)))
		val splits = centroids.indices.map(centroid => tempSplits.filter(p => p._1 == centroid))
		clusteringResult.unpersist()

		var toReCluster : Seq[Tweet] = Seq[Tweet]()
		splits.zipWithIndex foreach {case (cluster, index) =>
			val clusterElements: Seq[Tweet] = cluster.map(_._2)
			val clusterFeatures: Seq[Vector] = clusterElements.map(_.features)
			if(isFinal(clusterFeatures)){
				val exemplar = getExemplar(centroids(index), clusterFeatures)

				finalClusters :+= (exemplar, clusterFeatures)
				anyFinal = true
			} else {
				toReCluster = toReCluster.union(clusterElements)
			}
		}

		println("num clusters: " + numClusters)
		println("num final clusters " + finalClusters.size)
		if(clusteringResult != null)
			println("to re-cluster: " + toReCluster.size + "\n%%%%%%%%%%%%%%%%%%%%%%\n")
		else
			println("to re-cluster: 0 \n%%%%%%%%%%%%%%%%%%%%%%\n")

		dataFrame.unpersist(false)
		if(toReCluster.nonEmpty) {
			val newDF = Osint.dataFrameCreator.tweetSeqToDF(toReCluster)
			newDF.persist()
			cluster(newDF)
		}else{
			cluster(null)
		}
	}

	@tailrec
	private def newCluster(dataFrame : DataFrame, initialClusters : Int): Unit = {
		if(dataFrame == null || dataFrame.count() == 0)
			return

		var numClusters = initialClusters
		var kModel : KMeansModel = null
		var kMeans : KMeans = null
		var SSE, currentSSE : Double = Double.MaxValue
		var clusteringResult : DataFrame = null
		anyFinal = true

		if(prevTweets  == dataFrame.count()) {
			anyFinal = false
			numClusters = prevK
		}
		prevTweets = dataFrame.count()
		if(numClusters < 2)
			numClusters = 2

		println("Num clusters: " + numClusters)
		val numIterations = 50
		do{
			kMeans = new KMeans().setK(numClusters).setMaxIter(numIterations).setFeaturesCol("features")
			kModel = kMeans.fit(dataFrame)
			kModel.setPredictionCol("prediction")
			clusteringResult = kModel.transform(dataFrame)

			SSE = BigDecimal(kModel.computeCost(clusteringResult)).setScale(5, BigDecimal.RoundingMode.HALF_UP).toDouble
			println("SSE: " + SSE)
			println("old SSE: " + currentSSE)

			if((SSE > 0.0 && SSE < currentSSE && SSE != currentSSE) || (!anyFinal && numClusters <= prevK)) {
				currentSSE = SSE
				numClusters += 1
			}
		} while(SSE == currentSSE && numClusters < dataFrame.count())

		prevK = numClusters

		val centroids = kModel.clusterCenters
		val tempSplits = clusteringResult.collect().map(r => (r.getAs[Int]("prediction"), new Tweet(r)))
		val splits = centroids.indices.map(centroid => tempSplits.filter(p => p._1 == centroid))
		clusteringResult.unpersist()

		var toReCluster : Seq[Tweet] = Seq[Tweet]()
		splits.zipWithIndex foreach {case (cluster, index) =>
			val clusterElements: Seq[Tweet] = cluster.map(_._2)
			val clusterFeatures: Seq[Vector] = clusterElements.map(_.features)
			if(isFinal(clusterFeatures)){
				val exemplar = getExemplar(centroids(index), clusterFeatures)

				finalClusters :+= (exemplar, clusterFeatures)
				anyFinal = true
				numClusters -= 1
			} else {
				toReCluster ++= clusterElements
			}
		}

		println("num clusters: " + numClusters)
		println("num final clusters " + finalClusters.size)
		if(clusteringResult != null)
			println("to re-cluster: " + clusteringResult.count() + "\n%%%%%%%%%%%%%%%%%%%%%%\n")
		else
			println("to re-cluster: 0 \n%%%%%%%%%%%%%%%%%%%%%%\n")

		dataFrame.unpersist(false)
		if(toReCluster.nonEmpty) {
			val newDF = Osint.dataFrameCreator.tweetSeqToDF(toReCluster)
			newDF.persist()
			newCluster(newDF, numClusters)
		}else{
			newCluster(null, numClusters)
		}
	}

//
//	override def simpleClustering(dataFrame: DataFrame, outFile : String) : Unit = {
//		var rdd : RDD[Vector] = dataFrame.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(r => Vectors.dense(r.toArray))
//		rdd.cache()
//
//		var numClusters = 2
//		var kModel : KMeansModel = null
//		//var SSE, currentSSE : Double = Double.MaxValue
//		var done: Boolean = false
//
//		val numIterations = Osint.properties(Osint.NUM_ITERATIONS).toInt
//		do{
//			//numClusters += 1
//			kModel = KMeans.train(rdd, numClusters, numIterations)
//			var anySaved = false
//
//			val clusters : ArrayBuffer[ArrayBuffer[Vector]] = ArrayBuffer.fill[Vector](numClusters, 0)(null)
//			val featureList :Seq[Vector] = rdd.collect()
//
//			val centroids = kModel.clusterCenters
//
//			for(v <- featureList)
//				clusters(kModel.predict(v)) :+= v.asInstanceOf[Vector]
//
//			println("num clusters: " + numClusters)
//			println("num tweets: " + featureList.size)
//			var i = 0
//
//			for (cluster <- clusters){
//				if(cluster.size == 1 || isFinal(cluster)){
//					anySaved = true
//					numClusters = 2
//					val exemplar = getExemplar(centroids(i), cluster)
//					finalClusters :+= (exemplar, cluster)
//				}
//				i += 1
//			}
//			var toCluster: Seq[Vector] = null
//			if(finalClusters.nonEmpty) {
//				val a = clusters.reduce(_.union(_))
//				val b = finalClusters.map(x => x._2).reduce(_.union(_))
//				toCluster = a diff b
//			}else{
//				toCluster = clusters.reduce(_.union(_))
//			}
//			if(toCluster.isEmpty)
//				done = true
//			if(!anySaved)
//				numClusters += 1
//			println("toCluster: " +toCluster.size)
//			println()
//
//			rdd = Osint.spark.sparkContext.parallelize(toCluster)
//
//		}while(!done)
//
//		printClusters(dataFrame, "new_clusters_1_clusters", "new_clusters_1_stats")
//		finalClusters = new ArrayBuffer[(Vector, Seq[Vector])]()
//		//simpleClustering2(dataFrame)
//	}
//
//	def simpleClustering2(dataFrame: DataFrame) : Unit = {
//		val rdd : RDD[Vector] = dataFrame.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(r => Vectors.dense(r.toArray))
//		rdd.cache()
//
//		var numClusters = 1
//		var kModel : KMeansModel = null
//		//var SSE, currentSSE : Double = Double.MaxValue
//		var done: Boolean = false
//
//		val numIterations = Osint.properties(Osint.NUM_ITERATIONS).toInt
//		do{
//			numClusters += 1
//			kModel = KMeans.train(rdd, numClusters, numIterations)
//
//			val clusters : ArrayBuffer[ArrayBuffer[Vector]] = ArrayBuffer.fill[Vector](numClusters, 0)(null)
//			val clusterList :Seq[Vector] = rdd.collect()
//
//			val centroids = kModel.clusterCenters
//
//			for(v <- clusterList)
//				clusters(kModel.predict(v)) :+= v
//
//			println("num clusters: " + numClusters)
//			var i = 0
//
//			if(areClustersValid(clusters)) {
//				done = true
//				for (cluster <- clusters) {
//					val exemplar = getExemplar(centroids(i), cluster)
//					finalClusters :+= (exemplar, cluster)
//					i += 1
//				}
//			}
//		}while(!done)
//
//		printClusters(dataFrame, "new_clusters_2_clusters", "new_clusters_2_stats")
//		finalClusters = new ArrayBuffer[(Vector, Seq[Vector])]()
//		//simpleClustering3(dataFrame)
//	}
//
//	def simpleClustering3(dataFrame: DataFrame) : Unit = {
//		var rdd : RDD[Vector] = dataFrame.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(r => Vectors.dense(r.toArray))
//		rdd.cache()
//
//		var numClusters = 2
//		var kModel : KMeansModel = null
//		//var SSE, currentSSE : Double = Double.MaxValue
//		var done: Boolean = false
//
//		val numIterations = Osint.properties(Osint.NUM_ITERATIONS).toInt
//		do{
//			//numClusters += 1
//			kModel = KMeans.train(rdd, numClusters, numIterations)
//			var anySaved = false
//
//			val clusters : ArrayBuffer[ArrayBuffer[Vector]] = ArrayBuffer.fill[Vector](numClusters, 0)(null)
//			val featureList :Seq[Vector] = rdd.collect()
//
//			val centroids = kModel.clusterCenters
//
//			for(v <- featureList)
//				clusters(kModel.predict(v)) :+= v.asInstanceOf[Vector]
//
//			println("num clusters: " + numClusters)
//			println("num tweets: " + featureList.size)
//			var i = 0
//
//			for (cluster <- clusters){
//				if(cluster.size == 1 || isFinal(cluster)){
//					anySaved = true
//					val exemplar = getExemplar(centroids(i), cluster)
//					finalClusters :+= (exemplar, cluster)
//				}
//				i += 1
//			}
//			var toCluster: Seq[Vector] = null
//			if(finalClusters.nonEmpty) {
//				val a = clusters.reduce(_.union(_))
//				val b = finalClusters.map(x => x._2).reduce(_.union(_))
//				toCluster = a diff b
//			}else{
//				toCluster = clusters.reduce(_.union(_))
//			}
//			if(toCluster.isEmpty)
//				done = true
//			if(!anySaved)
//				numClusters += 1
//			println("toCluster: " +toCluster.size)
//			println()
//
//			rdd = Osint.spark.sparkContext.parallelize(toCluster)
//
//		}while(!done)
//
//		printClusters(dataFrame, "new_clusters_3_clusters", "new_clusters_3_stats")
//		finalClusters = new ArrayBuffer[(Vector, Seq[Vector])]()
//		//recursiveClustering(dataFrame)
//	}
//
//	private def areClustersValid(clusters: Seq[Seq[Vector]]) : Boolean = {
//		for(cluster <- clusters){
//			if(!isFinal(cluster))
//				return false
//		}
//		true
//	}

	//  override def simpleClustering(dataFrame: DataFrame) : Unit = {
	//    val rdd : RDD[Vector] = dataFrame.select("features").rdd.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(r => Vectors.dense(r.toArray))
	//
	//    var numClusters = 1
	//    var kModel : KMeansModel = null
	//    var SSE, currentSSE : Double = Double.MaxValue
	//
	//    rdd.cache()
	//    val numIterations = Osint.properties(Osint.NUM_ITERATIONS).toInt
	//    do{
	//      kModel = KMeans.train(rdd, numClusters, numIterations)
	//      SSE = kModel.computeCost(rdd)
	//      println("SSE: " + SSE)
	//
	//      if(SSE > 0.0 && SSE <= currentSSE) {
	//        currentSSE = SSE
	//        numClusters += 1
	//        if(numClusters > rdd.collect().length)
	//          SSE = Double.MaxValue
	//      }
	//    }while(SSE == currentSSE)
	//
	//    val clusters : ArrayBuffer[ArrayBuffer[Vector]] = ArrayBuffer.fill[Vector](numClusters, 0)(null)
	//    val clusterList :Seq[Vector] = rdd.collect()
	//
	//    val centroids = kModel.clusterCenters
	//
	//    for(v <- clusterList)
	//      clusters(kModel.predict(v)) :+= v.asInstanceOf[Vector]
	//
	//    println("num clusters: " + numClusters)
	//    var i = 0
	//    for(cluster <- clusters){
	//      val exemplar = getExemplar(centroids(i), cluster)
	//      finalClusters :+= (exemplar, cluster)
	//      i += 1
	//    }
	//
	//    printClusters(dataFrame, "clustering_stats-no_rec")
	//  }

	private def getExemplar(centroid : Vector, cluster : Seq[Vector]) : Vector = {
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

	override def simpleClustering(dataFrame: DataFrame, outFile: String): Unit = {}
}
