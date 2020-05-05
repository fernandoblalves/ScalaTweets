package osint

import java.util.logging.{Level, Logger}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

/**
  * Created by fernando on 13-03-2017.
  */
class TextProcessor extends Serializable{

  @transient lazy   val log: Logger = Logger.getLogger(getClass.getName)
  val stopWordsFile : String = Osint.properties(Osint.STOP_WORDS_FILE)
  val stopWords : Seq[String] = Osint.sparkContext.textFile("file://"+stopWordsFile).collect()

  /**
    * Method that reads the tweets from file, converts them to lower case,
    * and calls the text processing functions
    * @return The processed tweets, ready for feature extraction
    */
  def getDataset(dataPath : String, tranformNumbers: Boolean = true) : DataFrame = {
    log.log(Level.INFO, "Loading tweet dataset from: "+dataPath)
    val processedTweet : String = "processedTweet"

    // Generate the schema based on the string of schema
    val fields = new Array[StructField](4)
    fields(0) = StructField("label", StringType)
    fields(1) = StructField("date", StringType)
    fields(2) = StructField("tweet", StringType)
    fields(3) = StructField("id", StringType)
    val schema = StructType(fields)
    val readData : RDD[Row] = Osint.spark.read.option("sep", "\t").csv("file://"+dataPath).rdd //.option("inferSchema", "true").option("header", "true").
    var rawData = Osint.spark.sqlContext.createDataFrame(readData, schema)
    rawData = rawData.withColumn("label", rawData("label").cast(IntegerType))

    log.log(Level.INFO, "CSV dataset loaded")

    //text processing
    val loweredCase = rawData.withColumn(processedTweet, lowerCase(rawData("tweet")))
    val filteredStopWords = loweredCase.withColumn(processedTweet, filterStopWords(loweredCase(processedTweet)))
		var processedTweets : DataFrame = null
		if(tranformNumbers) {
			processedTweets = filteredStopWords.withColumn(processedTweet, filterRegexAndNumbers(filteredStopWords(processedTweet)))
		} else {
			processedTweets = filteredStopWords.withColumn(processedTweet, filterRegexWithoutNumbers(filteredStopWords(processedTweet)))
		}

    log.log(Level.INFO, "Tweet text processing finished")
    processedTweets
  }

  private def lowerCase= udf {make: String => make.toLowerCase}

  /**
    * Method that reads the stopwords from file, and removes them from the tweets
    */
  private def filterStopWords = udf {data : String =>
    var splitted : Array[String] = data.split(" ")

    for(stopWord <- stopWords)
      splitted = splitted.filter(!_.equals(stopWord))

    splitted.mkString(" ")
  }

  /**
    * Method that applies a set of regexes that remove special characters, and transforms numbers to their textual form
    * @return The processed tweets
    */
  private def filterRegexAndNumbers: UserDefinedFunction = udf { data : String =>
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
    replaced = removeNumbers(replaced).
      replaceAll(strangeRegex, "").
      replaceAll(dotRegex, " point ").
      replaceAll(hyphenRegex, " hyphen ").
      replaceAll(generalRegex, "").
      replaceAll(spacesRegex, " ").
      trim

    replaced
  }

	/**
		* Method that applies a set of regexes that remove special characters, but leaves numbers untouched
		* @return The processed tweets
		*/
	private def filterRegexWithoutNumbers = udf {data : String =>
		//remove hyperlinks
		val urlsRegex = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)"
		//remove strange characters
		val strangeRegex = "(\\\\)+[\\w]*"
		//replace "." with "dot"
		val dotRegex = "\\."
		//replace "-" with "hyphen"
		val hyphenRegex = "\\-"
		//remove all unnecessary characters
		val generalRegex = "[^\\s^a-z0-9]"
		//replace various spaces to one
		val spacesRegex = "\\s+"

		//spaces are removed after removing numbers since that method may leave two spaces
		var replaced = data.replaceAll(urlsRegex, "")
		replaced = replaced.
			replaceAll(strangeRegex, "").
			replaceAll(dotRegex, " point ").
			replaceAll(hyphenRegex, " hyphen ").
			//replaceAll(generalRegex, "").
			replaceAll(spacesRegex, " ").
			trim

		replaced
	}

  /**
    * Method that replaces all numbers with their word equivalent
    * @param line A string to process
    * @return The string with numbers replaced by text
    */
  private def removeNumbers(line : String) : String = {
    var indexStartNumber = 0
    var indexEndNumber = 0
    var newLine : String = line
    var number : String = null
    var convertedNumber : String = null
    var i = 0

    while (i != newLine.length) {
      indexEndNumber = 0
      indexStartNumber = i

      while (i < newLine.length() && Character.isDigit(newLine.charAt(i))) {
        indexEndNumber += 1
        i += 1
      }
      if (indexEndNumber > 0) {
        number = newLine.substring(indexStartNumber, indexStartNumber + indexEndNumber)
        convertedNumber = EnglishNumberToWords.convert(number.toLong)
        newLine = newLine.replaceAll(number, convertedNumber+" ")
        i -= 1
      }
      i += 1
    }
    newLine
  }
}