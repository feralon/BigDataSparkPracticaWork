package es.upm.bigdata.group5
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
/**
 * @author group5
 */
object App {

  def main(args : Array[String]) {
    println("Deploying App..")

    val sc = new SparkContext("local[*]" , "App")

    val spark = SparkSession
      .builder()
      .appName("Spark SQL")
      .enableHiveSupport()
      .getOrCreate()

    // Define columns
    val col_Year = col("Year")
    val col_Month = col("Month")
    val col_DayofMonth = col("DayofMonth")
    val col_DayOfWeek = col("DayOfWeek")
    val col_DepTime = col("DepTime")
    val col_CRSDepTime = col("CRSDepTime")
    val col_ArrTime = col("ArrTime")
    val col_CRSArrTime = col("CRSArrTime")
    val col_UniqueCarrier = col("UniqueCarrier")
    val col_FlightNum = col("FlightNum")
    val col_TailNum = col("TailNum")
    val col_ActualElapsedTime = col("ActualElapsedTime")
    val col_CRSElapsedTime = col("CRSElapsedTime")
    val col_AirTime = col("AirTime")
    val col_ArrDelay = col("ArrDelay")
    val col_DepDelay = col("DepDelay")
    val col_Origin = col("Origin")
    val col_Dest = col("Dest")
    val col_Distance = col("Distance")
    val col_TaxiIn = col("TaxiIn")
    val col_TaxiOut = col("TaxiOut")
    val col_Cancelled = col("Cancelled")
    val col_CancellationCode = col("CancellationCode")
    val col_Diverted = col("Diverted")
    val col_CarrierDelay = col("CarrierDelay")
    val col_WeatherDelay = col("WeatherDelay")
    val col_NASDelay = col("NASDelay")
    val col_SecurityDelay = col("SecurityDelay")
    val col_LateAircraftDelay = col("LateAircraftDelay")

    // Load files with correct schema (variable types)
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .schema(
        StructType(Array(
          StructField(col_Year.toString(), IntegerType, nullable=false),
          StructField(col_Month.toString(), IntegerType, nullable=false),
          StructField(col_DayofMonth.toString(), IntegerType, nullable=false),
          StructField(col_DayOfWeek.toString(), IntegerType, nullable=false),
          StructField(col_DepTime.toString(), StringType, nullable=false),
          StructField(col_CRSDepTime.toString(), StringType, nullable=false),
          StructField(col_ArrTime.toString(), StringType, nullable=false),
          StructField(col_CRSArrTime.toString(), StringType, nullable=false),
          StructField(col_UniqueCarrier.toString(), StringType, nullable=false),
          StructField(col_FlightNum.toString(), IntegerType, nullable=false),
          StructField(col_TailNum.toString(), StringType, nullable=false),
          StructField(col_ActualElapsedTime.toString(), IntegerType, nullable=false),
          StructField(col_CRSElapsedTime.toString(), IntegerType, nullable=false),
          StructField(col_AirTime.toString(), IntegerType, nullable=false),
          StructField(col_ArrDelay.toString(), IntegerType, nullable=false),
          StructField(col_DepDelay.toString(), IntegerType, nullable=false),
          StructField(col_Origin.toString(), StringType, nullable=false),
          StructField(col_Dest.toString(), StringType, nullable=false),
          StructField(col_Distance.toString(), IntegerType, nullable=false),
          StructField(col_TaxiIn.toString(), IntegerType, nullable=false),
          StructField(col_TaxiOut.toString(), IntegerType, nullable=false),
          StructField(col_Cancelled.toString(), BooleanType, nullable=false),
          StructField(col_CancellationCode.toString(), StringType, nullable=false),
          StructField(col_Diverted.toString(), BooleanType, nullable=false),
          StructField(col_CarrierDelay.toString(), IntegerType, nullable=false),
          StructField(col_WeatherDelay.toString(), IntegerType, nullable=false),
          StructField(col_NASDelay.toString(), IntegerType, nullable=false),
          StructField(col_SecurityDelay.toString(), IntegerType, nullable=false),
          StructField(col_LateAircraftDelay.toString(), IntegerType, nullable=false)
        ))
      )
      .load("B:\\Temp\\199*.csv") // Load only 90s files

      // Filter forbidden columns
      .drop(col_ArrTime)
      .drop(col_ActualElapsedTime)
      .drop(col_AirTime)
      .drop(col_TaxiIn)
      .drop(col_Diverted)
      .drop(col_CarrierDelay)
      .drop(col_WeatherDelay)
      .drop(col_NASDelay)
      .drop(col_SecurityDelay)
      .drop(col_LateAircraftDelay)

      // Transform data
        // Year + Month + Day --> Date
      .withColumn("Date", to_date(concat(col_Year.cast(StringType),lit("-"),
        when(col_Month < 10, concat(lit("0"),col_Month.cast(StringType))).otherwise(col_Month.cast(StringType)),
        lit("-"),
        when(col_DayofMonth < 10, concat(lit("0"),col_DayofMonth.cast(StringType))).otherwise(col_DayofMonth.cast(StringType))),
        "yyyy-MM-dd"))


      // SQL test examples
      df.createTempView("df_tmp")
      spark.sql("SELECT Distance * 1.6 AS DST_KM FROM df_tmp LIMIT 10").show()
      spark.sql("SELECT Date as OriginDate, weekday(Date) AS WeekDay FROM df_tmp LIMIT 20").show()

      // Retain (until a key is pressed) spark local server on spark_submit
      System.in.read
      spark.stop
}

}
