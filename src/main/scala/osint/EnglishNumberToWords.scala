package osint

import java.text.DecimalFormat

object EnglishNumberToWords {
  private val tensNames = Array("", " ten", " twenty", " thirty", " forty", " fifty", " sixty", " seventy", " eighty", " ninety")
  private val numNames = Array("", " one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten", " eleven", " twelve", " thirteen", " fourteen", " fifteen", " sixteen", " seventeen", " eighteen", " nineteen")

  private def convertLessThanOneThousand(n: Int) : String = {
    var number = n
    var soFar : String = null
    if (number % 100 < 20) {
      soFar = numNames(number % 100)
      number /= 100
    }
    else {
      soFar = numNames(number % 10)
      number /= 10
      soFar = tensNames(number % 10) + soFar
      number /= 10
    }
    if (number == 0) return soFar
    numNames(number) + " hundred" + soFar
  }

  def convert(number: Long): String = {
    // 0 to 999 999 999 999
    if (number == 0) return "zero"
    var snumber = number.toString
    // pad with "0"
    val mask = "000000000000"
    val df = new DecimalFormat(mask)
    snumber = df.format(number)
    // XXXnnnnnnnnn
    val billions = snumber.substring(0, 3).toInt
    // nnnXXXnnnnnn
    val millions = snumber.substring(3, 6).toInt
    // nnnnnnXXXnnn
    val hundredThousands = snumber.substring(6, 9).toInt
    // nnnnnnnnnXXX
    val thousands = snumber.substring(9, 12).toInt
    var tradBillions : String = null
    billions match {
      case 0 =>
        tradBillions = ""
      case 1 =>
        tradBillions = convertLessThanOneThousand(billions) + " billion "
      case _ =>
        tradBillions = convertLessThanOneThousand(billions) + " billion "
    }
    var result = tradBillions
    var tradMillions : String = null
    millions match {
      case 0 =>
        tradMillions = ""
      case 1 =>
        tradMillions = convertLessThanOneThousand(millions) + " million "
      case _ =>
        tradMillions = convertLessThanOneThousand(millions) + " million "
    }
    result = result + tradMillions
    var tradHundredThousands : String = null
    hundredThousands match {
      case 0 =>
        tradHundredThousands = ""
      case 1 =>
        tradHundredThousands = "one thousand "
      case _ =>
        tradHundredThousands = convertLessThanOneThousand(hundredThousands) + " thousand "
    }
    result = result + tradHundredThousands
    var tradThousand : String = null
    tradThousand = convertLessThanOneThousand(thousands)
    result = result + tradThousand
    // remove extra spaces!
    result.replaceAll("^\\s+", "").replaceAll("\\b\\s{2,}\\b", " ")
  }

  def main(args: Array[String]) {
    System.out.println("*** " + EnglishNumberToWords.convert(2014))
    System.out.println("*** " + EnglishNumberToWords.convert(6278))
    System.out.println("*** " + EnglishNumberToWords.convert(16))
    System.out.println("*** " + EnglishNumberToWords.convert(100))
    System.out.println("*** " + EnglishNumberToWords.convert(118))
    System.out.println("*** " + EnglishNumberToWords.convert(200))
    System.out.println("*** " + EnglishNumberToWords.convert(219))
    System.out.println("*** " + EnglishNumberToWords.convert(800))
    System.out.println("*** " + EnglishNumberToWords.convert(801))
    System.out.println("*** " + EnglishNumberToWords.convert(1316))
    System.out.println("*** " + EnglishNumberToWords.convert(1000000))
    System.out.println("*** " + EnglishNumberToWords.convert(2000000))
    System.out.println("*** " + EnglishNumberToWords.convert(3000200))
    System.out.println("*** " + EnglishNumberToWords.convert(700000))
    System.out.println("*** " + EnglishNumberToWords.convert(9000000))
    System.out.println("*** " + EnglishNumberToWords.convert(9001000))
    System.out.println("*** " + EnglishNumberToWords.convert(123456789))
    System.out.println("*** " + EnglishNumberToWords.convert(2147483647))
    System.out.println("*** " + EnglishNumberToWords.convert(3000000010L))
    /*
         *** zero
         *** one
         *** sixteen
         *** one hundred
         *** one hundred eighteen
         *** two hundred
         *** two hundred nineteen
         *** eight hundred
         *** eight hundred one
         *** one thousand three hundred sixteen
         *** one million
         *** two millions
         *** three millions two hundred
         *** seven hundred thousand
         *** nine millions
         *** nine millions one thousand
         *** one hundred twenty three millions four hundred
         **      fifty six thousand seven hundred eighty nine
         *** two billion one hundred forty seven millions
         **      four hundred eighty three thousand six hundred forty seven
         *** three billion ten
         **/
  }

}