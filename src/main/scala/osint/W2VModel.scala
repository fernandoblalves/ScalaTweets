package osint

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

/**
	* Created by fernando on 10-03-2017.
	*/
class W2VModel extends FeatureExtraction with Serializable {
	private var vectorMap : Map[String, Vector] = Map[String, Vector]()
	//This is all pretty dirty, but these methods should be called only once
	private var maxSize = 0
	private var model: Word2VecModel = _

	override def transform(dataset : DataFrame) : DataFrame = {
		if(vectorMap.isEmpty){
			loadModel()
		}

		dataset.withColumn("features", transform(dataset("processedTweet")))
	}

	private def transform: UserDefinedFunction = udf { data : String =>
		model.transform(data)
	}

	override def train(dataset : DataFrame) : Unit = {
		//This is a dirty hack to simplify w2v usage since this class is in itself a hack
		loadModel()
	}

	private def stringToVector(value: String): Vector = {
		Vectors.dense(value.split(", ").map(_.toDouble))
	}

	override def loadModel() : Unit = {
		val file = Osint.properties(Osint.W2V_VECTORS_PATH)
		def readUntil(inputStream: DataInputStream, term: Char, maxLength: Int = 1024 * 8): String = {
			var char: Char = inputStream.readByte().toChar
			val str = new StringBuilder
			while (!char.equals(term)) {
				str.append(char)
				assert(str.size < maxLength)
				char = inputStream.readByte().toChar
			}
			str.toString
		}

		val inputStream: DataInputStream = new DataInputStream(new FileInputStream(file))
		try {
			val header = readUntil(inputStream, '\n')
			val (records, dimensions) = header.split(" ") match {
				case Array(r, d) => (r.toInt, d.toInt)
			}
			val vectors = (0 until records).toArray.map(_ => {
				readUntil(inputStream, ' ') -> (0 until dimensions).map(_ => {
					java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
				}).toArray
			}).toMap
			model = new Word2VecModel(vectors)
		} finally {
			inputStream.close()
		}
	}

	override def saveModel() : Unit = {}
}
