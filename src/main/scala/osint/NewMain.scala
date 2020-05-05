package osint

import java.io.File

/**
	* Created by fernando on 03-04-2017.
	*/
object NewMain {

	def main(args : Array[String]) : Unit = {
		//    val featureSizes ="30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 1500 2000 2500 3000".split(" ")
		//
		//    val perceptronNumber = "2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20".split(" ")
		//    val numLayers = "2 3 4".split(" ")
		//
		//    val cs = "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 3 4 5 6 7 8 9 10".split(" ")
		//    val stepSizes = "0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2".split(" ")
		//    val dataSets = "A B C D ABCD".split(" ")
		//    val featureExtractors = "TF-IDF".split(" ")

		//    val featureSizes = Array(args(1))
		//
		//    val perceptronNumber = "5 10 15 20".split(" ")
		//    val numLayers = "3 4".split(" ")
		//
		//    val cs = "0.01 0.02 0.05 0.1 1 10".split(" ")
		//    val stepSizes = "0.2 1 2 5".split(" ")
		//    val dataSets = Array(args(0))
		//    val featureExtractors = "TF-IDF".split(" ")

		//    val featureSizes ="100 3000".split(" ")
		//
		//    val perceptronNumber = "10 20".split(" ")
		//    val numLayers = "2 3 4".split(" ")
		//
		//    val cs = "0.1 1".split(" ")
		//    val stepSizes = "0.2 1".split(" ")
		//    val dataSets = "A B".split(" ")
		//    val featureExtractors = "TF-IDF".split(" ")

		val featureSizes ="30 50 80 100 200 300 500 750 1000 1500 3000".split(" ")

		val perceptronNumber = "5 7 10 12 14 16 18 20".split(" ")
		val numLayers = "2 3 4 5 6 7 8".split(" ")

		val cs = "0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0".split(" ")
		val stepSizes = "0.1 0.5 1.0 1.5 2.0 5.0".split(" ")
		val dataSets = "A B C D ABCD".split(" ")
		val featureExtractors = "TF-IDF".split(" ")

		//val root = "/home/fernando/Documents/DiSIEM/tweets/classificados/d1/"
		val root = "/root/falves/dataset/classificados/d1/"
		val stopwords = "stopwords_file=/root/falves/dataset/stopwords"
		val save = "save_model_path=/root/falves/results/"

		val fileList : List[String] = getListOfFiles("/root/falves/results/")
		var resultFile = ""

		//TODO add w2v

		for(featureExtractor <- featureExtractors) {
			for (fs <- featureSizes) {
				for (dataSet <- dataSets) {
					//SVM
					for (c <- cs) {
						for (stepSize <- stepSizes) {
							resultFile = featureExtractor + "_" + fs + "_SVM_" + c + "_" + stepSize + "_d1" + dataSet
							if(!fileList.contains("/root/falves/results/" + resultFile)) {
								println("executing " + resultFile)
								Osint.main(Array("classifier=SVM",
									"feature_extractor=" + featureExtractor,
									"mode=cross_validate",
									"num_features=" + fs,
									"svm_c=" + c,
									"svm_stepsize=" + stepSize,
									"train_dataset_path=" + root + dataSet,
									stopwords, save))
							} else
								println(resultFile + " already exists; skipping")
						}
					}

					//MLP
					for (perceptrons <- perceptronNumber) {
						for (layers <- numLayers) {
							resultFile = featureExtractor + "_" + fs + "_MLP_" + layers + "_" + perceptrons + "_d1" + dataSet
							if(!fileList.contains("/root/falves/results/" + resultFile)) {
								println("executing " + resultFile)
								Osint.main(Array("classifier=MLP",
									"feature_extractor=" + featureExtractor,
									"mode=cross_validate",
									"num_features=" + fs,
									"mlp_hidden_perceptrons=" + perceptrons,
									"mlp_hidden_layers=" + layers,
									"train_dataset_path=" + root + dataSet,
									stopwords, save))
							} else
								println(resultFile + " already exists; skipping")
						}
					}
				}
			}
		}
	}

	def getListOfFiles(dir: String):List[String] = {
		val d = new File(dir)
		if (d.exists && d.isDirectory) {
			d.listFiles.filter(_.isFile).map(_.toString).toList
		} else {
			List[File]().map(_.toString)
		}
	}
}
