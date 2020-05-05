package osint

/**
	* Created by fernando on 24-05-2017.
	*/
object Properties {

	val OVERWRITING : String = "overwriting"
	val MODE: String = "mode"
	val FEATURE_EXTRACTOR : String = "feature_extractor"
	val NUM_FEATURES : String = "num_features"
	val SPLIT_RATIO : String = "split_ratio"

	val CLASSIFIER : String = "classifier"

	val TRAIN_DATASET_PATH : String = "train_dataset_path"
	val VALIDATE_DATASET_PATH : String = "validate_dataset_path"
	val STOP_WORDS_FILE : String = "stopwords_file"
	val SAVE_MODEL_PATH : String = "save_model_path"
	val CROSS_VALIDATION_FOLDS : String = "cross_validation_folds"

	val SVM_STEPSIZE : String = "svm_stepsize"
	val SVM_C : String = "svm_c"
	val MLP_HIDDEN_LAYERS : String = "mlp_hidden_layers"
	val MLP_HIDDEN_PERCEPTRONS : String = "mlp_hidden_perceptrons"

	val NUM_ITERATIONS : String = "num_iterations"
	val CLUSTERER : String = "clusterer"
	val ROW_THRESHOLD : String = "column_threshold"
	val TABLE_THRESHOLD : String = "table_threshold"
	val EPS : String = "eps"
	val DISTANCE_MEASURE : String = "distance_measure"
	val PCA_FEATURES : String = "pca"
}
