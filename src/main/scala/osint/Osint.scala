package osint

import java.io.{BufferedWriter, FileWriter, InputStream, PrintWriter}
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.logging.{Level, Logger}

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.reflect.io.Path

/**
	* Created by fernando on 10-03-2017.
	*/
object Osint {

	//TODO move all these file names into the properties object

	val OVERWRITING : String = "overwriting"
	val MODE: String = "mode"
	val FEATURE_EXTRACTOR : String = "feature_extractor"
	val NUM_FEATURES : String = "num_features"
	val SPLIT_RATIO : String = "split_ratio"
	val MODEL_NAME_SUFFIX: String = "model_name_suffix"

	val CLASSIFIER : String = "classifier"

	val TF_IDF_DATASET : String = "TF-IDF_dataset"
	val TRAIN_DATASET_PATH : String = "train_dataset_path"
	val VALIDATE_DATASET_PATH : String = "validate_dataset_path"
	val STOP_WORDS_FILE : String = "stopwords_file"
	val SAVE_MODEL_PATH : String = "save_model_path"
	val CROSS_VALIDATION_FOLDS : String = "cross_validation_folds"
	val W2V_VECTORS_PATH : String = "w2v_vectors_path"
	val W2V_VECTOR_SIZE: String = "w2v_vector_size"

	val SVM_STEPSIZE : String = "svm_stepsize"
	val SVM_C : String = "svm_c"
	val SVM_ITERATIONS : String = "svm_iterations"
	val MLP_HIDDEN_LAYERS : String = "mlp_hidden_layers"
	val MLP_HIDDEN_PERCEPTRONS : String = "mlp_hidden_perceptrons"
	val MLP_ITERATIONS : String = "mlp_iterations"

	val NUM_ITERATIONS : String = "num_iterations"
	val CLUSTERER : String = "clusterer"
	val ROW_THRESHOLD : String = "column_threshold"
	val TABLE_THRESHOLD : String = "table_threshold"
	val EPS : String = "eps"
	val DISTANCE_MEASURE : String = "distance_measure"
	val PCA_FEATURES : String = "pca"

	val propertiesFile: String = "/properties.conf"
	var dataSetInUse : String = ""
	var log: Logger = _
	var properties: Map[String, String] = _
	var sparkContext: SparkContext = _
	var spark : SparkSession = _
	var dataFrameCreator: DataFrameCreator = _

	//TODO place column names in global vars or enumerate
	//TODO re-comment
	//TODO change TF-IDF dataset

	def main(args: Array[String]): Unit = {
		log = Logger.getLogger(getClass.getName)

		spark = SparkSession.builder().appName("OsintScala").getOrCreate()
		sparkContext = spark.sparkContext
		log.log(Level.INFO, "Spark session created")

		dataFrameCreator = new DataFrameCreator(spark)

		log.log(Level.INFO, "Beginning program execution")
		properties = readPropertiesFile
		log.log(Level.INFO, "Properties loaded")

		val changes : String = readCliArguments(args)
		log.log(Level.INFO, "Command line arguments read: "+changes)

		val overwriting : Boolean = properties(OVERWRITING).toBoolean
		if(!overwriting){
			dataSetInUse = properties(TRAIN_DATASET_PATH)
			val path = Path(getSavePath(getClassifier(properties(CLASSIFIER)).toString))
			if(path.exists){
				log.log(Level.WARNING, "Not overwriting results and exit file exists; program exiting")
				spark.stop()
				sys.exit(0)
			}
		}

		val mode = properties(MODE)
		val classifierType = properties(CLASSIFIER)
		log.log(Level.INFO, "Execution mode \"" + mode + "\" on classifier \"" + classifierType + "\"")

		//TODO change matching to global vars or enumerate
		mode match {
			case "train" => trainModel(classifierType)
			case "classify" => classify(classifierType)
			case "cross_validate" => crossValidate(classifierType)
			case "train_and_test" => trainAndTest(classifierType)
			case "cluster_only" => clusterOnly()
			case "process" => processText()
			case "ensemble" =>
				ensemble("MLP")
				ensemble("SVM")
			case "cluster_data" => globalClusterData()
			case "w2v_test" => w2vTest()
			case "clean" => clean()
			case "naive" => naive()
			case "cluster_positives" => clusterPositives()
			case "cluster_cleaner" => clusterCleaner()
			case x => log.log(Level.SEVERE, "Option " + x + " is not a valid execution mode.")
		}

		log.log(Level.INFO, "Execution finished; exiting")
		spark.stop()
	}

	private def w2vTest(): Unit = {
		val datasetWithFeatures = getTrainingFeatures(properties(TRAIN_DATASET_PATH))
		datasetWithFeatures.show(1000)
	}

	private def clusterPositives() : Unit = {
		val df = getClassificationFeatures(properties(VALIDATE_DATASET_PATH))
		val positives = df.filter(r => r.getAs[Int](0) == 1)
		cluster(positives, "", properties(MODEL_NAME_SUFFIX))
	}

	private def naive() : Unit = {
		val df = new TextProcessor().getDataset(properties(VALIDATE_DATASET_PATH))
		val filtered = df.withColumn("prediction", processNaiveWords(df("processedTweet")))
		import java.time.format.DateTimeFormatter
		val dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd")
		val april1st = LocalDate.parse("2016-04-01", dtf)
		val dates = filtered.select("date", "prediction").filter(_.getAs[Double](1) > 0).collect().map(r => LocalDate.parse(r.getAs[String](0).substring(0, 10), dtf)).filter(_.compareTo(april1st) > 0).sortWith((s1, s2) => s1.compareTo(s2) < 0)

		var weeks = new ListBuffer[ListBuffer[(Int, LocalDate)]]
		var week = -1
		var woy = -1
		for (date <- dates) {
			if (woy != getWeekOfYear(date)) {
				woy = getWeekOfYear(date)
				week += 1
				weeks += new ListBuffer[(Int, LocalDate)]
			}
			weeks(week) += ((woy, date))
		}

		println("num weeks " + weeks.size)
		println("df count:" + df.count())
		weeks foreach (r => println(r.head._1 + " : " + r.size))
		println(weeks.map(p => p.size).sum)


		println("\n------------------> " + filtered.select("prediction").filter(r => r.getAs[Double](0) > 0.0).count())
		println
		new ModelMetrics(filtered).writeResults(properties(SAVE_MODEL_PATH) + "naive/out.txt", properties(NUM_FEATURES).toInt).printResults(log)
	}

	val attackWords : Array[String] = Array("access", "acl", "admin", "advisory", "allow", "arbitrary", "aslr", "assurance", "attack", "auth", "buffer", "bug", "bypass", "certificate", "code", "command", "corruption", "csrf", "cve", "cyber", "denial", "deployment", "dereference", "disclosure", "execute", "exploit", "hack", "heap", "identity", "injection", "interception", "leak", "overflow", "privilege", "remote", "root", "scripting", "security", "stack", "threat", "unauthenticated", "vuln", "xss")
	//val attackWords : Seq[String] = Seq("identity theft identity fraud account", "credentials stealing trojans", "receiving unsolicited email", "spam", "unsolicited infected emails", "denial of service",  "distributed denial of service","distributed denial of network", "service network layer attack", "service application layer attack", "service amplification reflection attack", "malicious code software activity", "search engine poisoning", "exploitation of fake trust of social media", "worms trojans", "rootkits", "mobile malware", "infected trusted mobile apps", "elevation of privileges", "web application attacks injection attacks code injection SQL XSS", "spyware or deceptive adware", "viruses", "rogue security software rogueware scareware", "ransomware", "exploits exploit kits", "social engineering", "phishing attacks", "spear phishing attacks", "abuse of information leakage", "leakage affecting mobile privacy and mobile applications", "leakage affecting web privacy and web applications", "leakage affecting network traffic", "leakage affecting cloud computing", "generation and use of rogue certificates", "loss of integrity of sensitive information", "man in the middle session hijacking", "social engineering via signed malware", "fake SSL certificates", "manipulation of hardware and software", "anonymous proxies", "abuse of computing power of cloud to launch attacks cybercrime as a service", "abuse of vulnerabilities 0 day vulnerabilities", "access of web sites through chains of HTTP Proxies Obfuscation", "access to device software", "alternation of software", "rogue hardware", "manipulation of information", "repudiation of actions", "address space hijacking IP prefixes", "routing table manipulation", "DNS poisoning or DNS spoofing or DNS Manipulations", "falsification of record", "autonomous system hijacking", "autonomous system manipulation", "falsification of configurations", "misuse of audit tools", "misuse of information or information systems including mobile apps", "unauthorized activities", "Unauthorised use or administration of devices and systems", "unauthorised use of software", "unauthorized access to the information systems or networks like IMPI Protocol DNS Registrar Hijacking", "network intrusion", "unauthorized changes of records", "unauthorized installation of software", "Web based attacks drive by download or malicious URLs or browser based attacks", "compromising confidential information like data breaches", "hoax", "false rumour and or fake warning", "remote activity execution", "remote command execution", "remote access tool", "botnets remote activity", "targeted attacks", "mobile malware exfiltration", "spear phishing attacks targeted", "installation of sophisticated and targeted malware", "watering hole attacks", "failed business process", "brute force", "abuse of authorizations", "war driving", "intercepting compromising emissions", "interception of information", "corporate espionage", "nation state espionage", "information leakage due to unsecured wi fi like rogue access points", "interfering radiation", "replay of messages", "network reconnaissance network traffic manipulation and information gathering", "man in the middle session hijacking", "Remotely injected by agent", "Included in automated software update", "Instant Messaging", "Email via user-executed attachment", "Directly installed or inserted by threat agent", "Downloaded and installed by local malware", "Removable storage media or devices", "Web via auto-executed or drive-by infection", "Email via embedded link", "Network propagation", "Unknown", "Email via automatic execution", "Web via user-executed or downloaded content", "Other", "Send spam", "Unknown", "Packet sniffer", "Backdoor", "Exploit vulnerability in code", "Other", "Password dumper", "Scan or footprint network", "Downloader", "System or network utilities", "Click fraud or Bitcoin mining", "Adware", "Command and control", "Worm", "Spyware, keylogger or form-grabber", "Brute force attack", "Capture data from application or system process", "Ram scraper or memory parser", "Disable or interfere with security controls", "Capture data stored on system disk", "Ransomware", "Export data to another site or system", "Client-side or browser attack", "redirection", "XSS", "Content Spoofing", "MitB", "SQL injection attack", "Rootkit", "Destroy or corrupt stored data", "DoS attack", "Physical access within corporate facility", "Remote access connection to corporate network", "Local network access within corporate facility", "Unknown", "Non-corporate facilities or networks", "Other", "Use of unapproved software or services", "Storage or distribution of illicit content", "Unapproved workaround or shortcut", "Use of unapproved hardware or devices", "Unknown", "Inappropriate use of email or IM", "Abuse of physical access to asset", " Other", "Inappropriate use of network or Web access", "Handling of data in an unapproved manner", "Abuse of system access privileges", "Abuse of private or entrusted knowledge", "Physical access or connection", "Remote shell", "Unknown", "Backdoor or command and control channel", "Web application", "Graphical desktop sharing", "3rd party online desktop sharing", "Partner connection or credential", "VPN", "Other", "Cross-site scripting", "HTTP Response Splitting", "Unknown", "Buffer overflow", "Format string attack", "LDAP injection", "SSI injection", "Man-in-the-middle attack", "Path traversal", "URL redirector abuse", "Use of Backdoor or C2 channel", "Mail command injection", "Virtual machine escape", "OS commanding", "Soap array abuse", "Footprinting and fingerprinting", "Cryptanalysis", "SQL injection", "XML external entities", "Abuse of functionality", "XML injection", "Routing detour", "HTTP response smuggling", "Forced browsing or predictable resource location", "Cache poisoning", "Null byte injection", "Reverse engineering", "Brute force or password guessing attacks", "Fuzz testing", "Offline password or key cracking", "Cross-site request forgery", "XML entity expansion", "Remote file inclusion", "Session fixation", "Integer overflows", "XQuery injection", "Pass-the-hash", "XML attribute blowup", "Credential or session prediction", "Use of stolen authentication credentials", "HTTP request smuggling", "XPath injection", "Other", "Denial of service", "Special element injection", "HTTP request splitting", "Session replay", "theft", "fraud", "steal", "stolen", "trojan", "spam", "dos", "ddos", "attack", "reflection", "malicious", "poisoning", "exploit", "exploitation", "worm", "infected", "elevation", "escalation", "injection", "spyware", "adware", "virus", "rogueware", "scareware", "phishing", "leak", "leakage", "rogue", "integrity", "hijack", "MITM", "man in the middle", "malware", "fake", "certificate", "certificates", "cybercrime", "vulnerability", "vulnerabilities", "zero day", "0day", "hijacking", "spoofing", "misuse", "intrusion", "breach", "remote", "execution", "access", "botnet", "compromise", "replay", "infection", "Adware", "keylogger", "BO", "LDAP", "footprinting", "fingerprinting", "routing", "smuggling", "replay")
	private def processNaiveWords= udf {tweet: String =>
		var value = -1.0
		for (word <- attackWords) {
			if (tweet.contains(word.toLowerCase()))
				value = 1.0
		}
		value
	}

	private def clean() : Unit = {
		val df = new TextProcessor().getDataset(properties(TRAIN_DATASET_PATH))
		val dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd")
		var datesProcessed = df.collect().map(r => (r.getAs[Integer](0), LocalDate.parse(r.getAs[String](1).substring(0, 10), dtf), r.getAs[String](2), r.getAs[String](3)))

		val april1st = LocalDate.parse("2016-04-01", dtf)

		val writer = new BufferedWriter(new java.io.FileWriter(properties(VALIDATE_DATASET_PATH)))
		datesProcessed = datesProcessed.sortWith((s1, s2) => s1._2.compareTo(s2._2) < 0)
		for (date <- datesProcessed) {
			if(date._2.compareTo(april1st) > 0)
				writer.write(date._1 + "\t" + date._2 + "\t" + date._3 + "\t" + date._4 + "\n")
		}
		writer.close()
	}

	private def processText() : Unit = {
		dataSetInUse = properties(TRAIN_DATASET_PATH)
		val splits = dataSetInUse.split("/")
		val dataset = splits(splits.length - 2)++splits(splits.length - 1)
		val df = new TextProcessor().getDataset(dataSetInUse)
		df.cache()
		log.log(Level.INFO, "dataset ready")
		import java.io.BufferedWriter
		val writer = new BufferedWriter(new FileWriter(properties(SAVE_MODEL_PATH) + dataset))
		log.log(Level.INFO, "writer ready")
		df.select("label", "processedTweet").rdd.map(r => (r.getAs[Integer](0), r.getAs[String](1))).collect().foreach{pair => writer.write(pair._1 + "\t" + pair._2 + "\n")}
		//df.select("processedTweet").rdd.map(r => r.getAs[String](0)).collect().foreach{entry => writer.write(entry + "\n")}
		writer.close()
	}

	private def trainModel(classifierType : String) : Unit = {
		dataSetInUse = properties(TRAIN_DATASET_PATH)

		val classifier = getClassifier(classifierType)
		val datasetWithFeatures = getTrainingFeatures(properties(TRAIN_DATASET_PATH))
		datasetWithFeatures.persist()
		classifier.train(datasetWithFeatures)
		classifier.saveModel()
	}

	private def classify(classifierType: String) : Unit = {
		dataSetInUse = properties(VALIDATE_DATASET_PATH)

		val datasetWithFeatures = getClassificationFeatures(properties(VALIDATE_DATASET_PATH))
		datasetWithFeatures.persist()

		val classifier = getClassifier(classifierType)
		classifier.loadModel()

		val results = classifier.classify(datasetWithFeatures)
		new ModelMetrics(results).writeResults(getSavePath(classifier.toString), properties(NUM_FEATURES).toInt).printResults(log)

		val tps = results.filter(r => r.getAs[Double]("label") > 0.0 && r.getAs[Double]("prediction") > 0.0)
		val fps = results.filter(r => r.getAs[Double]("label") <= 0.0 && r.getAs[Double]("prediction") > 0.0)
		val tns = results.filter(r => r.getAs[Double]("label") <= 0.0 && r.getAs[Double]("prediction") <= 0.0)
		val fns = results.filter(r => r.getAs[Double]("label") > 0.0 && r.getAs[Double]("prediction") <= 0.0)

		Seq[DataFrame](tps,fps,tns,fns).zip(Seq[String]("tps", "fps", "tns", "fns")) foreach {case (df, label) =>
			val writer = new BufferedWriter(new FileWriter(label))
			df.collect() foreach{row =>
				writer.write(row.getAs[Double]("prediction") + " " + row.getAs[Double]("tweet"))
			}
		}

		//cluster(results.select("tweet", "features", "prediction", "date"), classifier.toString)
	}

	private def trainAndTest(classifierType : String) : Unit = {
		dataSetInUse = properties(TRAIN_DATASET_PATH)

		log.log(Level.INFO, "Initiating training")
		val datasetWithFeatures = getTrainingFeatures(properties(TRAIN_DATASET_PATH))
		datasetWithFeatures.cache()
		val (train, test) = splitDataFrameByRatio(properties(SPLIT_RATIO).toDouble, datasetWithFeatures)
		train.persist(StorageLevel.MEMORY_ONLY)
		test.persist(StorageLevel.MEMORY_ONLY)
		val classifier = getClassifier(classifierType)

		log.log(Level.INFO, "Initiating classification")
		classifier.train(train)
		val results = classifier.classify(test)
		new ModelMetrics(results).printResults(log).writeResults(properties(SAVE_MODEL_PATH)+classifier, properties(NUM_FEATURES).toInt)

		log.log(Level.INFO, "Initiating clustering")
		cluster(results.select("tweet", "features", "prediction", "date"), classifier.toString, "")
	}

	private def ensemble(classifierType : String) : Unit = {
		//dataSetInUse = properties(VALIDATE_DATASET_PATH)
		val results_d1 = new Array[DataFrame](4)
		val results_d2 = new Array[DataFrame](4)
		val results_d3 = new Array[DataFrame](4)
		var i = 0
		var classifier : Classifier = null

		for(datasetLetter <- "A B C D".split(" ")) {
			if(classifierType == "SVM") {
				if (datasetLetter.equals("A")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.5")
					properties = properties + ("svm_stepsize" -> "0.5")
				}else if (datasetLetter.equals("B")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.05")
					properties = properties + ("svm_stepsize" -> "1.5")
				}else if (datasetLetter.equals("C")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.2")
					properties = properties + ("svm_stepsize" -> "5.0")
				}else if (datasetLetter.equals("D")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.01")
					properties = properties + ("svm_stepsize" -> "1.5")
				}
			} else if(classifierType == "MLP") {
				if (datasetLetter.equals("A")) {
					properties = properties + ("num_features" -> "1500")
					properties = properties + ("mlp_hidden_layers" -> "4")
					properties = properties + ("mlp_hidden_perceptrons" -> "5")
				}else if (datasetLetter.equals("B")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("mlp_hidden_layers" -> "7")
					properties = properties + ("mlp_hidden_perceptrons" -> "20")
				}else if (datasetLetter.equals("C")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("mlp_hidden_layers" -> "3")
					properties = properties + ("mlp_hidden_perceptrons" -> "10")
				}else if (datasetLetter.equals("D")) {
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("mlp_hidden_layers" -> "7")
					properties = properties + ("mlp_hidden_perceptrons" -> "20")
				}
			}

			properties = properties + ("mode" -> "train")
			properties = properties + ("TF-IDF_dataset" -> ("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter))
			val trainDataset = getTrainingFeatures("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter)
			trainDataset.persist()
			classifier = getClassifier(properties(CLASSIFIER))
			classifier.train(trainDataset)
			results_d1(i) = classifier.classify(trainDataset)

			properties = properties + ("mode" -> "classify")
			properties = properties + ("TF-IDF_dataset" -> ("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter))
			val testDataset_d2 = getTrainingFeatures("/home/fernando/Documents/DiSIEM/tweets/classificados/d2_/" + datasetLetter)
			testDataset_d2.persist()
			results_d2(i) = classifier.classify(testDataset_d2)

			val testDataset_d3 = getTrainingFeatures("/home/fernando/Documents/DiSIEM/tweets/classificados/d3/" + datasetLetter)
			testDataset_d3.persist()
			results_d3(i) = classifier.classify(testDataset_d3)

			i += 1
		}

		val ensembleResults_d1 = results_d1.reduce((a,b) => a.union(b))
		new ModelMetrics(ensembleResults_d1).writeResults("/home/fernando/Documents/DiSIEM/spark/data/results/" + classifierType + "_d1_ensemble", properties(NUM_FEATURES).toInt).printResults(log)

		val ensembleResults_d2 = results_d2.reduce((a,b) => a.union(b))
		new ModelMetrics(ensembleResults_d2).writeResults("/home/fernando/Documents/DiSIEM/spark/data/results/" + classifierType + "_d2_ensemble", properties(NUM_FEATURES).toInt).printResults(log)

		val ensembleResults_d3 = results_d3.reduce((a,b) => a.union(b))
		new ModelMetrics(ensembleResults_d3).writeResults("/home/fernando/Documents/DiSIEM/spark/data/results/" + classifierType + "_d3_ensemble", properties(NUM_FEATURES).toInt).printResults(log)

		//cluster(ensembleResults.select("tweet", "features", "prediction", "date"), classifier.toString)
	}

	private def globalClusterData() : Unit = {
		val datasets = "d2_ d3".split(" ")
		val results = new Array[DataFrame](8)
		var i = 0
		var classifier : Classifier = null

		//svm_c=2.0
		//svm_stepsize=0.1
		//mlp_hidden_layers=7
		//mlp_hidden_perceptrons=14
		//num_features=750
		for(dataset <- datasets) {
			for (datasetLetter <- "A B C D".split(" ")) {
				if (datasetLetter.equals("A")) {
					properties = properties + ("classifier" -> "SVM")
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.5")
					properties = properties + ("svm_stepsize" -> "0.5")
				} else if (datasetLetter.equals("B")) {
					properties = properties + ("classifier" -> "MLP")
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("mlp_hidden_layers" -> "7")
					properties = properties + ("mlp_hidden_perceptrons" -> "20")
				} else if (datasetLetter.equals("C")) {
					properties = properties + ("classifier" -> "SVM")
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.2")
					properties = properties + ("svm_stepsize" -> "5.0")
				} else if (datasetLetter.equals("D")) {
					properties = properties + ("classifier" -> "SVM")
					properties = properties + ("num_features" -> "3000")
					properties = properties + ("svm_c" -> "0.01")
					properties = properties + ("svm_stepsize" -> "1.5")
				}

				properties = properties + ("mode" -> "train")
				properties = properties + ("TF-IDF_dataset" -> ("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter))
				val trainDataset = getTrainingFeatures("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter)
				trainDataset.persist()
				classifier = getClassifier(properties(CLASSIFIER))
				classifier.train(trainDataset)

				properties = properties + ("mode" -> "classify")
				properties = properties + ("TF-IDF_dataset" -> ("/home/fernando/Documents/DiSIEM/tweets/classificados/d1/" + datasetLetter))
				val testDataset = getTrainingFeatures("/home/fernando/Documents/DiSIEM/tweets/classificados/" + dataset + "/" + datasetLetter)
				testDataset.persist()
				results(i) = classifier.classify(testDataset)
				i += 1
			}
		}

		val ensembleResults = results.map(_.select("label", "date", "tweet", "id", "processedTweet", "words", "rawFeatures", "features", "prediction")).reduce((a,b) => a.union(b))
		val positiveTweets = ensembleResults.select("tweet", "features", "prediction", "date", "id").filter(r => r.getDouble(2) > 0.0)
		positiveTweets.persist(StorageLevel.MEMORY_ONLY)
		////////////////////////7

		import java.time.format.DateTimeFormatter
		val dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd")

		val dates = positiveTweets.select("tweet", "features", "date", "id").collect().map(r => (r.getAs[String](0), r.getAs[Vector](1), r.getAs[String](2).substring(0, 10), r.getAs[String](3)))
		var datesProcessed : List[IntermediateRow] = dates.map(r => IntermediateRow(r._1, r._2, LocalDate.parse(r._3, dtf), r._4)).toList

		var week = -1
		var woy = -1
		val april1st = LocalDate.parse("2016-04-01", dtf)
		datesProcessed = datesProcessed.filter(_.date.compareTo(april1st) > 0).sortWith((s1, s2) => s1.date.compareTo(s2.date) < 0)
		val weeks = new ListBuffer[ListBuffer[IntermediateRow]]

		for (date <- datesProcessed) {
			if (woy != getWeekOfYear(date.date)) {
				woy = getWeekOfYear(date.date)
				week += 1
				weeks += new ListBuffer[IntermediateRow]
			}
			weeks(week) += date
		}

		println("num weeks " + weeks.size)
		weeks foreach (r => println(r.size))


		for(w <- weeks) {
			val rows = w.map(s => Row.fromSeq(Seq(s.tweet, s.vector, s.date.toString, s.id)))
			val toDF = spark.sparkContext.parallelize(rows)

			val fields = new Array[StructField](4)
			fields(0) = StructField("tweet", StringType)
			fields(1) = StructField("features", org.apache.spark.ml.linalg.SQLDataTypes.VectorType)
			fields(2) = StructField("date", StringType)
			fields(3) = StructField("id", StringType)
			val schema = StructType(fields)
			val df = spark.sqlContext.createDataFrame(toDF, schema)
			df.persist(StorageLevel.MEMORY_ONLY)
			cluster(df, classifier.toString, "ensemble")
		}


		//new ModelMetrics(ensembleResults).writeResults("/home/fernando/Documents/DiSIEM/spark/data/results/" + classifierType + "_" + dataset + "_ensemble").printResults(log)

		//cluster(ensembleResults.select("tweet", "features", "prediction", "date"), classifier.toString)
	}

	case class IntermediateRow(tweet : String, vector :Vector, date : LocalDate, id : String)

	import java.time.LocalDate
	import java.time.temporal.WeekFields
	import java.util.Locale

	def compare(o1: IntermediateRow, o2: IntermediateRow): Int = {
		var result = getWeekOfYear(o1.date) - getWeekOfYear(o2.date)
		if (result == 0)
			result = o1.date.compareTo(o2.date)
		result
	}

	protected def getWeekOfYear(date: LocalDate): Int = {
		val wf = WeekFields.of(Locale.getDefault)
		date.get(wf.weekOfYear)
	}

	private def cluster(toCluster : DataFrame, classifierData : String, outFile : String) : Unit = {
		val clusterer = getClusterer(properties(CLUSTERER))
		clusterer.recursiveClustering(toCluster, outFile)
	}

	private def crossValidate(classifierType : String) : Unit = {
		dataSetInUse = properties(TRAIN_DATASET_PATH)
		val crossValidationFolds : Int = properties(CROSS_VALIDATION_FOLDS).toInt
		var classifier : Classifier = null

		val trainResults : ListBuffer[ModelMetrics] = new ListBuffer[ModelMetrics]()
		val testResults : ListBuffer[ModelMetrics] = new ListBuffer[ModelMetrics]()

		val datasetWithFeatures = getTrainingFeatures(properties(TRAIN_DATASET_PATH))
		datasetWithFeatures.persist(StorageLevel.MEMORY_ONLY)
		val arraySplits : Array[Double] = Array.fill[Double](crossValidationFolds)(1)
		val splits = datasetWithFeatures.randomSplit(arraySplits, seed = 11L)
		splits foreach(_.persist(StorageLevel.MEMORY_ONLY))

		//cross validation with n splits
		for (i <- 0 until crossValidationFolds) {
			val (training, test) = splitDataFrameForCrossValidation(splits, i)
			training.persist(StorageLevel.MEMORY_ONLY)
			test.persist(StorageLevel.MEMORY_ONLY)
			classifier = getClassifier(classifierType)
			log.log(Level.INFO, "Classifier ready")
			classifier.train(training)
			log.log(Level.INFO, "Classifier training finished")
			trainResults += new ModelMetrics(classifier.classify(training))
			testResults += new ModelMetrics(classifier.classify(test))
			log.log(Level.INFO, "Classification on test data finished")
		}

		new ModelMetricsAverage(trainResults: _*).printResults(crossValidationFolds, log).writeResults(crossValidationFolds, getSavePath(classifier.toString) + "_train", log)
		new ModelMetricsAverage(testResults: _*).printResults(crossValidationFolds, log).writeResults(crossValidationFolds, getSavePath(classifier.toString) + "_test", log)
	}

	private def splitDataFrameByRatio(ratio : Double, data : DataFrame) : (DataFrame, DataFrame) = {
		val arraySplits : Array[Double] = Array(ratio, 1 - ratio)
		val splits = data.randomSplit(arraySplits, seed = 11L)
		(splits(0), splits(1))
	}

	private def splitDataFrameForCrossValidation(splits : Array[DataFrame], i : Int) : (DataFrame, DataFrame) = {
		var training : DataFrame = null
		var test : DataFrame = null
		for(j <- 0 to 9){
			if(j == i)
				test = splits(i)
			else{
				if(training == null) {
					training = splits(j)
				}else{
					training = training.union(splits(j))
				}
			}
		}
		(training, test)
	}

	private def clusterOnly(): Unit = {
		def process(dataFrame: DataFrame) : DataFrame = {
			val numFeatures : Int = Osint.properties(Osint.NUM_FEATURES).toInt
			val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")
			val wordsData = tokenizer.transform(dataFrame)

			val hashingTF = new HashingTF()
				.setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)

			hashingTF.transform(wordsData)
		}

		dataSetInUse = properties(TRAIN_DATASET_PATH)
		val filename = properties(MODEL_NAME_SUFFIX)

		val fields = new Array[StructField](4)
		fields(0) = StructField("label", StringType)
		fields(1) = StructField("date", StringType)
		fields(2) = StructField("tweet", StringType)
		fields(3) = StructField("id", StringType)
		val schema = StructType(fields)
		val readData : RDD[Row] = Osint.spark.read.option("sep", "\t").csv("file://"+dataSetInUse).rdd //.option("inferSchema", "true").option("header", "true").
		var rawData = Osint.spark.sqlContext.createDataFrame(readData, schema)
		val initialSize = rawData.count()
		rawData = rawData.filter{r =>
			val s = r.getAs[String]("tweet")
			s != null && !s.equals("")
		}
		val processedSize = rawData.count()
		println("Lines removed: " + (initialSize - processedSize))

		val featurizedData = process(rawData)
		val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
		val idfModel = idf.fit(featurizedData)
		val datasetWithFeatures = idfModel.transform(featurizedData)

		cluster(datasetWithFeatures, filename, filename)
	}

	private def clusterCleaner(): Unit = {
		def filterRegexAndNumbers(data: String): String = {
			//remove hyperlinks
			val urlsRegex = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)"
			//remove strange characters
			val strangeRegex = "(\\\\)+[\\w]*"
			//replace "." with "dot"
			val dotRegex = "\\."
			//replace "-" with "hyphen"
			val hyphenRegex = "\\-"
			//remove all unnecessary characters
			val generalRegex = "[^\\s^a-z]"
			//replace various spaces to one
			val spacesRegex = "\\s+"

			//spaces are removed after removing numbers since that method may leave two spaces
			var replaced = data.replaceAll(urlsRegex, "")
			replaced = replaced.
				replaceAll(strangeRegex, "").
				replaceAll(dotRegex, "").
				replaceAll(hyphenRegex, "").
				replaceAll(generalRegex, "").
				replaceAll(spacesRegex, " ").
				trim

			replaced
		}

		val stopWords : Seq[String] = Osint.sparkContext.textFile("file://"+Osint.properties(Osint.STOP_WORDS_FILE)).collect()
		def filterStopWords(data: String): String = {
			var splitted : Array[String] = data.split(" ")

			for(stopWord <- stopWords)
				splitted = splitted.filter(!_.equals(stopWord))

			splitted.mkString(" ")
		}

		//TODO apply cluster cleaner
		val baseName = "/home/fernando/Documents/DiSIEM/spark/data/results_fosint/"
		val clusterTypes = Seq[String]("hashtags", "ner", "tweetHashtags", "tweetNer", "tweets")
		val infras = Seq[String]("amadeus.csv_clusters", "atos.csv_clusters", "edp.csv_clusters")
		val commentLine = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
		clusterTypes foreach { clusterType =>
			infras foreach { infra =>
				var clusterLines = new mutable.ListBuffer[ListBuffer[String]]()
				var currentBuffer = new mutable.ListBuffer[String]()
				val file = Source.fromFile(baseName + clusterType + "DF_" + infra)

				file.getLines() foreach { line =>
					if (!line.equals(commentLine)) {
						if (line.isEmpty) {
							clusterLines.append(currentBuffer.tail)
							currentBuffer = new mutable.ListBuffer[String]()
						} else {
							currentBuffer.append(line)
						}
					}
				}
				file.close()

				//now, onto sorting to make my life easier
				clusterLines = clusterLines.map { list =>
					val tokens = filterStopWords(filterRegexAndNumbers(list.head)).split(" ")
					val sorted = tokens.sortWith((a, b) => a.compareTo(b) <= 0).mkString(" ")
					val res = new mutable.ListBuffer[String]()
					res.append(sorted + "\t" + list.head)
					res.append(list.tail: _*)
					res
				}
				clusterLines = clusterLines.sortWith((a, b) => a.head.compareTo(b.head) <= 0)
				clusterLines = clusterLines.map { lines =>
					val head = lines.head
					var remainder = lines.tail
					remainder = remainder.map { str =>
						val splitted = str.split("\t")
						val sorted = splitted(2).split(" ").sortWith((a, b) => a.compareTo(b) <= 0).mkString(" ")
						(splitted.take(2) :+ sorted).mkString("\t") + "\t" + str
					}
					val res = new mutable.ListBuffer[String]()
					res.append(head)
					res.append(remainder: _*)
					res
				}

				//finally, writing the ground truth
				new PrintWriter("/home/fernando/Documents/DiSIEM/spark/data/results_fosint/clean_" + clusterType + "DF_" + infra) {
					try {
						clusterLines.zipWithIndex foreach { case (cluster, index) =>
							cluster.tail foreach (line => write(line + "\n"))
							write("\n")
						}
					} finally {
						close()
					}
				}
			}
		}
	}

	//TODO I should not have repeated code
	private def getTrainingFeatures(dataPath : String) : DataFrame = {
		val dataset = new TextProcessor().getDataset(dataPath)
		val featureExtractor = getFeatureExtractor
		featureExtractor.train(dataset)
		featureExtractor.saveModel()
		featureExtractor.transform(dataset)
	}

	private def getClassificationFeatures(dataPath : String) : DataFrame = {
		val dataset = new TextProcessor().getDataset(dataPath)
		val featureExtractor = getFeatureExtractor
		featureExtractor.loadModel()
		featureExtractor.transform(dataset)
	}

	/**
		* Reads the properties file and creates a map
		* @return The properties map containing various settings for the program execution
		*/
	private def readPropertiesFile : Map[String, String] = {
		val stream : InputStream = getClass.getResourceAsStream(propertiesFile)
		val propertiesHandler = Source.fromInputStream(stream)
		propertiesHandler.getLines()
			.filter(s => !s.startsWith("#"))//comment lines
			.map { s =>
			val split = s.split("=")
			(split(0), split(1))
		}.toMap[String, String]
	}

	/**
		* Reads command line arguments and replaces the values present on the properties map
		* @param args The command line arguments
		* @return
		*/
	private def readCliArguments(args : Array[String]) : String = {
		val cliMap = args.map { s =>
			val split = s.split("=")
			(split(0), split(1))
		}

		for ((k, v) <- cliMap) {
			if(properties contains k)
				properties = properties + (k -> v)
			else
				log.log(Level.WARNING, "Configuration not found: "+k)
		}
		cliMap.map{ case (k,v) => k + " -> "+ v}.mkString("", "; ", "")
	}

	/**
		* Creates the appropriate classifier
		* @param classifierType The type of classifier to create
		* @return
		*/
	private def getClassifier(classifierType: String): Classifier = {
		var classifier: Classifier = null
		classifierType match {
			case "SVM" =>
				classifier = new LinearSVMClassifier()
			case "MLP" =>
				classifier = new MultiLayerPerceptron()
			case _ => log.log(Level.SEVERE, "classifier \"" + classifierType + "\" not recognized")
				classifier = null
		}
		classifier
	}

	private def getFeatureExtractor : FeatureExtraction ={
		val whichExtractor = properties(FEATURE_EXTRACTOR)
		log.log(Level.INFO, "Feature extraction method: "+whichExtractor)
		var featureExtractor: FeatureExtraction = null
		whichExtractor match {
			case "TF-IDF" => featureExtractor = new TFIDFModel()
			case "W2V" => featureExtractor = new W2VModel()
		}
		featureExtractor
	}

	private def getClusterer(clusterType: String): Clusterer = {
		var clusterer: Clusterer = null
		clusterType match {
			case "kmeans" =>
				clusterer = new KMeansCluster()
			case "DBSCAN" =>
				clusterer = new DBSCANCluster()
			case _ => log.log(Level.SEVERE, "classifier \"" + clusterType + "\" not recognized")
				clusterer = null
		}
		clusterer
	}

	private def getSavePath(classifierData : String) : String = {
		val splits = dataSetInUse.split("/")
		val dataset = splits(splits.length - 2)++splits(splits.length - 1)
		val featureExtractor = properties(FEATURE_EXTRACTOR)
		val numFeatures = properties(NUM_FEATURES)
		properties(SAVE_MODEL_PATH) + featureExtractor + "_" + numFeatures + "_" + classifierData + "_" + dataset
	}
}