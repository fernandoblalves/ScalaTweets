package osint

import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
	* Created by fernando on 14-02-2017.
	*/
object SetDistances {
	//TODO these implementations calculate the distance between an element and itself

	def euclidian(set: Seq[Vector]) : Double = {
		var sum : Double = 0
		val numElems = set.size
		for (v1 <- set) {
			for (v2 <- set) {
				sum += Vectors.sqdist(v1, v2)
			}
		}
		sum / (numElems * numElems)
	}

	def manhattan(set: Seq[Vector]) : Double = {
		val sum : Double = 0
		val numElems = set.size
		for (v1 <- set) {
			for (v2 <- set) {
				manhattan(v1, v2)
			}
		}
		sum / (numElems * numElems)
	}

	/*
		REVIEW
	*/
	def manhattan(v1: Vector, v2: Vector) : Double = {
		var sum : Double = 0
		val d1 = v1.toArray
		val d2 = v2.toArray
		for (dA <- d1) {
			for (dB <- d2) {
				sum += Math.abs(dA - dB)
			}
		}
		sum
	}

	def cosine(set: Seq[Vector]) : Double = {
		val sum : Double = 0
		val numElems = set.size
		for (v1 <- set) {
			for (v2 <- set) {
				cosine(v1, v2)
			}
		}
		sum / (numElems * numElems)
	}

	def cosine(v1: Vector, v2: Vector) : Double = {
		var dotProduct = 0.0
		var normA = 0.0
		var normB = 0.0
		val d1 = v1.toArray
		val d2 = v2.toArray
		for(i <- d1.indices) {
			dotProduct += d1(i) * d2(i)
			normA += Math.pow(d1(i), 2)
			normB += Math.pow(d2(i), 2)
		}
		dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
	}
}
