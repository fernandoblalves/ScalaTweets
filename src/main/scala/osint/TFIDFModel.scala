package osint

import java.util.logging.Logger

import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.DataFrame

import scala.reflect.io.Path

/**
  * Created by fernando on 10-03-2017.
  */
class TFIDFModel extends FeatureExtraction{

  val log: Logger = Logger.getLogger(getClass.getName)
  var idfModel  : IDFModel = _

  override def train(dataset : DataFrame) : Unit = {
		dataset.show(20000)
		val featurizedData = process(dataset)

		featurizedData.show(20000)
		val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
		idfModel = idf.fit(featurizedData)
	}

  override def transform(dataset: DataFrame): DataFrame = {
		val processed = process(dataset)
		idfModel.transform(processed)
	}

	private def process(dataFrame: DataFrame) : DataFrame = {
		val numFeatures : Int = Osint.properties(Osint.NUM_FEATURES).toInt
		val tokenizer = new Tokenizer().setInputCol("processedTweet").setOutputCol("words")
		val wordsData = tokenizer.transform(dataFrame)

		val hashingTF = new HashingTF()
			.setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)

		hashingTF.transform(wordsData)
	}

  override def loadModel() : Unit = {
    val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
    idfModel = IDFModel.load("file://" + savePath + "TF-IDFModel_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
  }

  override def saveModel() : Unit = {
    val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
    val path = Path(savePath + "TF-IDFModel_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
    if(path.exists)
      path.deleteRecursively()
    idfModel.save("file://" + path)
  }
}
