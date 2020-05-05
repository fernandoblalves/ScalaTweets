package osint

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

/**
	* Created by fernando on 10-03-2017.
	*/
class Word2VecBreadModel extends FeatureExtraction with Serializable {
	private var vectorMap : Map[String, Vector] = Map[String, Vector]()
	//This is all pretty dirty, but these methods should be called only once
	private var maxSize = 0

	override def transform(dataset : DataFrame) : DataFrame = {
		if(vectorMap.isEmpty){
			throw new Exception("The w2v model needs to be loaded first")
		}

		//		maxSize = 0
		//		dataset.select("processedTweet").foreach{s =>
		//			val size = s.getAs[String](0).split(" ").length
		//			if(size > maxSize)
		//				maxSize = size
		//		}
		//		maxSize *= Osint.properties(Osint.W2V_VECTOR_SIZE).toInt
		//		println(maxSize)
		maxSize = dataset.select("processedTweet").collect().map(_.getAs[String](0).split(" ").length).max * Osint.properties(Osint.W2V_VECTOR_SIZE).toInt

		dataset.withColumn("features", transform(dataset("processedTweet")))
	}

	private def transform: UserDefinedFunction = udf { data : String =>
		var result: Vector = null
		for(token <- data.split(" ")) {
			var search: Vector = null
			if (vectorMap.contains(token)) {
				search = vectorMap(token)
			} else {
				search = vectorMap("<UNK>")
			}
			if(result == null)
				result = search
			else
				result = Vectors.dense(result.toArray ++ search.toArray)
		}

		val resArray = result.toArray
		result = Vectors.sparse(maxSize, resArray.indices.zip(resArray))
		//result = Vectors.dense(resArray ++ Array.fill(maxSize - resArray.length){0.0})

		result
	}

	override def train(dataset : DataFrame) : Unit = {
		//This is a dirty hack to simplify w2v usage since this class is in itself a hack
		loadModel()
	}

	private def stringToVector(value: String): Vector = {
		Vectors.dense(value.split(", ").map(_.toDouble))
	}

	override def loadModel() : Unit = {
		for(line <- scala.io.Source.fromFile(Osint.properties(Osint.W2V_VECTORS_PATH)).getLines()){
			val splits = line.split(": ")
			//TODO remove []
			val (key, value) = (splits(0), splits(1))
			vectorMap += (key -> stringToVector(value.tail.dropRight(1)))
		}
	}

	override def saveModel() : Unit = {}
}
