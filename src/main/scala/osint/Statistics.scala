package osint

/**
	* Created by fernando on 06-06-2017.
	*/
object Statistics {

	def getMean(data : Seq[Double]) : Double = {
		var sum = 0.0
		for (a <- data) {
			sum += a
		}
		sum / data.length
	}

	def getVariance(data : Seq[Double]) : (Double, Double) = {
		val mean = getMean(data)
		var temp : Double = 0
		for (a <- data) {
			temp += (a - mean) * (a - mean)
		}
		(mean, temp / data.length)
	}

	def getMeanAndStdDev(data : Seq[Double]) : (Double, Double) = {
		val (mean, variance) = getVariance(data)
		(setDecimals(mean, 5), setDecimals(Math.sqrt(variance), 5))
	}

	def minMax(a: Array[Double]) : (Double, Double) = {
		if (a.isEmpty) throw new java.lang.UnsupportedOperationException("array is empty")
		a.foldLeft((a(0), a(0)))
		{ case ((min, max), e) => (setDecimals(math.min(min, e), 5), setDecimals(math.max(max, e), 5))}
	}

	def setDecimals(num : Double, decimalPlaces : Int) : Double = BigDecimal(num).setScale(decimalPlaces, BigDecimal.RoundingMode.HALF_UP).toDouble
}
