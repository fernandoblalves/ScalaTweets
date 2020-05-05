package osint

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

class Tweet(
	var date : String,
	var id : String,
	var text : String,
	var features : Vector) extends Serializable{

	def this(row: Row){
		this(
			row.getAs[String](DFColNames.DATE),
			row.getAs[String](DFColNames.ID),
			row.getAs[String](DFColNames.TWEET),
			row.getAs[Vector](DFColNames.FEATURES)
		)
	}

	override def toString: String = {
		id + " " + date + " " + text
	}

	override def equals(o: Any): Boolean = {
		if(!o.isInstanceOf[Tweet]){
			return false
		}
		val tweet = o.asInstanceOf[Tweet]

		tweet.id == this.id
	}
}
