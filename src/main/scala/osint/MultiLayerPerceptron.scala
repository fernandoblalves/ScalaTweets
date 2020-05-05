package osint

import java.util.logging.{Level, Logger}

import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.sql.DataFrame

/**
	* Created by fernando on 10-03-2017.
	*/
class MultiLayerPerceptron extends Classifier with Serializable{

	@transient lazy val log: Logger = Logger.getLogger(getClass.getName)

	private var model : MultilayerPerceptronClassificationModel = _
	private val prefix : String = "MLP"
	private var numLayers : Int = Osint.properties(Osint.MLP_HIDDEN_LAYERS).toInt
	private var numPerceptrons : Int = Osint.properties(Osint.MLP_HIDDEN_PERCEPTRONS).toInt

	override def train(data : DataFrame): Unit = {
		val layers = new Array[Int](numLayers + 2)
		//TODO this won't be right using w2v since the size of the vectors depends on the longest tweet
		layers(0) = Osint.properties(Osint.NUM_FEATURES).toInt

		log.log(Level.INFO, "Number of layers: "+numLayers)
		for(i <- 0 to numLayers){
			layers(i + 1) = numPerceptrons
		}
		layers(numLayers + 1) = 2

		log.log(Level.INFO, "MLP layers: "+layers.mkString("<", ",", ">"))

		val numIterations : Int = Osint.properties(Osint.MLP_ITERATIONS).toInt
		// create the trainer and set its parameters
		val trainer = new MultilayerPerceptronClassifier()
			.setLayers(layers)
			.setBlockSize(128)
			.setSeed(1234L)
			.setMaxIter(numIterations)
			.setFeaturesCol("features")
			.setLabelCol("label")
			.setPredictionCol("prediction")

		// train the model
		model = trainer.fit(data)
	}

	override def classify(data : DataFrame): DataFrame = {
		model.transform(data)
	}

	override def saveModel(): Unit = {
		val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
		model.write.overwrite().save("file://" + savePath + prefix + "Model_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
	}

	override def loadModel(): Unit = {
		val savePath : String = Osint.properties(Osint.SAVE_MODEL_PATH)
		model = MultilayerPerceptronClassificationModel.load("file://" + savePath + prefix + "Model_" + Osint.properties(Osint.MODEL_NAME_SUFFIX))
	}

	override def toString: String = {
		"MLP_" + numLayers + "_" + numPerceptrons
	}
}