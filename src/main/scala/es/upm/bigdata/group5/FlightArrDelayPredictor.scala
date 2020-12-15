package es.upm.bigdata.group5

import es.upm.bigdata.group5.models.{DecisionTreeGenerator, GBTGenerator, LinearRegressionGenerator, RandomForestGenerator}
import es.upm.bigdata.group5.utilities.Constants._
import org.apache.spark.SparkContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import scala.util.control.Breaks.break

/**
 * FlightArrDelayPredictor
 * @author group5
 */
object FlightArrDelayPredictor {

  /**
   * Case class represents the App configuration
   * @param inputFilePaths the inputFilePaths
   * @param trainingModel the trainingModel
   * @param sampleDf the sampleDf
   * @param saveOutputMeasures the location of saveOutputMeasures
   * @param promptSqlCLI if prompt a sql cli at the spark console
   * @param predict The location of predicting models (if prediction is wanted)
   * @param predictedDfPath The location of the predicted model
   */
  private case class AppConfiguration(inputFilePaths: List[String], trainingModel: String, sampleDf: Double,
                                      saveOutputMeasures: String, promptSqlCLI: Boolean, predict: List[String],
                                      predictedDfPath:String)


  /**
   * Generates the config based on the arguments
   * @param args arguments
   * @return the config
   */
  private def generateConfig(args: Array[String]): AppConfiguration = {
    var config = AppConfiguration(
      inputFilePaths = List("*.csv"),
      trainingModel = "LinearRegression",
      sampleDf = 0.0d,
      saveOutputMeasures = null,
      promptSqlCLI = false,
      predict = null,
      predictedDfPath = "Predicted"
    )

    if(args.length > 1)
      args.reduce((r1, r2) => r1 + " " + r2).split("(?=--)").foreach(
        arg => {
          val splitArg = arg.split(" ")
          splitArg(0) match {
            case ArgInputFilePaths => config = config.copy(inputFilePaths = splitArg.drop(1).toList)
            case ArgTrainingModel if splitArg.length == 2 => config = config.copy(trainingModel = splitArg(1))
            case ArgSaveOutputMeasures  if splitArg.length == 2 => config = config
              .copy(saveOutputMeasures = splitArg(1))
            case ArgPromptSqlCLI if splitArg.length == 1 => config = config.copy(promptSqlCLI = true)
            case ArgSampleTrainingDf if splitArg.length == 2 =>
                try
                  config = config.copy(sampleDf = splitArg(1).toDouble)
                catch{
                  case _: Exception => throw new IllegalArgumentException("Error: Invalid Format on " + splitArg(0))
                }
            case ArgPredict => config = config.copy(predict = splitArg.drop(1).toList)
            case ArgPredictedDfPath if splitArg.length == 2 => config = config.copy(predictedDfPath = splitArg(1))
            case unknown => throw new IllegalArgumentException ("Error: Unknown/Misused Arg: " + unknown )
          }
        }
      )

    config
  }

  /**
   * Loads a dataframe, ensuring that there are at least minimum data for prediction or training
   * @param spark spark session
   * @param paths file paths
   * @param training if its for training
   * @return non empty Dataset
   */
  private def loadDataFrame(spark:SparkSession, paths:List[String], training: Boolean): Dataset[Row] ={
    // Load files with correct schema (variable types)
    val preDf = spark.read
      .option("header", "true")
      .option("delimiter", ",")
      .schema(
        StructType(Array(
          StructField(ColYear, IntegerType, nullable = true),
          StructField(ColMonth, IntegerType, nullable = true),
          StructField(ColDayOfMonth, IntegerType, nullable = true),
          StructField(ColDayOfWeek, IntegerType, nullable = true),
          StructField(ColDepTime, IntegerType, nullable = true),
          StructField(ColCRSDepTime, StringType, nullable = true),
          StructField(ColArrTime, StringType, nullable = true),
          StructField(ColCRSArrTime, StringType, nullable = true),
          StructField(ColUniqueCarrier, StringType, nullable = true),
          StructField(ColFlightNum, IntegerType, nullable = true),
          StructField(ColTailNum, StringType, nullable = true),
          StructField(ColActualElapsedTime, IntegerType, nullable = true),
          StructField(ColCRSElapsedTime, IntegerType, nullable = true),
          StructField(ColAirTime, IntegerType, nullable = true),
          StructField(ColArrDelay, DoubleType, nullable = true),
          StructField(ColDepDelay, IntegerType, nullable = true),
          StructField(ColOrigin, StringType, nullable = true),
          StructField(ColDest, StringType, nullable = true),
          StructField(ColDistance, IntegerType, nullable = true),
          StructField(ColTaxiIn, IntegerType, nullable = true),
          StructField(ColTaxiOut, IntegerType, nullable = true),
          StructField(ColCancelled, BooleanType, nullable = true),
          StructField(ColCancellationCode, StringType, nullable = true),
          StructField(ColDiverted, BooleanType, nullable = true),
          StructField(ColCarrierDelay, IntegerType, nullable = true),
          StructField(ColWeatherDelay, IntegerType, nullable = true),
          StructField(ColNASDelay, IntegerType, nullable = true),
          StructField(ColSecurityDelay, IntegerType, nullable = true),
          StructField(ColLateAircraftDelay, IntegerType, nullable = true)
        ))
      )
      // Load all CSVs
      .csv(paths: _*)

      // Drop rows where nulls on interest columns
      .filter(col(ColDepTime).isNotNull and col(ColDepDelay).isNotNull and col(ColOrigin).isNotNull
        and col(ColTaxiOut).isNotNull)
      .filter(col(ColArrDelay).isNotNull or lit(!training))

    if(preDf.count() == 0){
      throw new IllegalArgumentException(s"Empty Dataset or Wrong format (minimum columns required): ${paths.mkString(",")}. Exiting...")
    }

    // Add columns
    val df = preDf
        .withColumn(ColDepHour, substring(lpad(col(ColDepTime).cast(StringType), 4, "0"), 1, 2)
        .cast(IntegerType))
        .withColumn(ColDepMin, substring(lpad(col(ColDepTime).cast(StringType), 4, "0"), 3, 2)
          .cast(IntegerType))
    df
  }

  /**
   *  Main point of entry for the App
   * @param args arguments
   */
  def main(args: Array[String]) {
    println("Deploying App...")

    val configApp = generateConfig(args)

    val sc = new SparkContext("local[*]", "App")

    val spark = SparkSession
      .builder()
      .appName("Spark SQL")
      .enableHiveSupport()
      .getOrCreate()

    // Load training df
    val df = loadDataFrame(spark,configApp.inputFilePaths, training = true)

      // Drop forbidden columns
      .drop(ColArrTime)
      .drop(ColActualElapsedTime)
      .drop(ColAirTime)
      .drop(ColTaxiIn)
      .drop(ColDiverted)
      .drop(ColCarrierDelay)
      .drop(ColWeatherDelay)
      .drop(ColNASDelay)
      .drop(ColSecurityDelay)
      .drop(ColLateAircraftDelay)

      // Select only used columns in our models
      .select(ColDepTime, ColDepHour, ColDepMin, ColArrDelay, ColOrigin, ColDepDelay, ColTaxiOut)

    val sampledDf = if (configApp.sampleDf > 0) df.sample(withReplacement = false, configApp.sampleDf) else df

    // Num of distinct airports
    val distinctAirports = sampledDf.select(ColOrigin).dropDuplicates().count()

    // Choose the model
    val modelGen = configApp.trainingModel match {
      case LinearRegressionModelName => new LinearRegressionGenerator()
      case DecisionTreeModelName => new DecisionTreeGenerator(distinctAirports.toInt)
      case GBTModelName => new GBTGenerator(distinctAirports.toInt)
      case RandomForestModelName =>  new RandomForestGenerator(distinctAirports.toInt)
      case m => throw new IllegalArgumentException("Unknown model: " + m)
    }

    // Pipeline for data transformation
    val dataTransformer = modelGen.getDataTransformer.fit(sampledDf)

    // Apply transformations
    val featureDf = dataTransformer.transform(sampledDf)

    // Split into train/test(80%) and validation(20%)
    val seed = 3044
    val Array(trainingData, validationData) = featureDf.randomSplit(Array(0.8, 0.2), seed)

    // Hyperparameter tuning by Cross validator
    val cv = new CrossValidator()
      .setEstimator(modelGen.getModel)
      .setEvaluator(new RegressionEvaluator().setMetricName("r2").setLabelCol(ColArrDelay))
      .setEstimatorParamMaps(modelGen.getParamGridBuilt)
      .setNumFolds(3)
      .setParallelism(4)

    // Model
    val modelTrained = cv.fit(trainingData)

    println(s"\n\n\nTRAINED:")
    // Trained Model  Specs
    println(s"Best Model Parameters: ${modelTrained.bestModel.asInstanceOf[PipelineModel].stages(0).extractParamMap()}")

    // Evaluate model
    val evaluationDf = modelTrained.bestModel.transform(validationData)

    // Instantiate metrics object
    val metrics = new RegressionMetrics(evaluationDf.select("prediction", ColArrDelay).rdd.
      map(row => (row.getDouble(0), row.getDouble(1))))


    println(s"\n\n\nValidation Metrics: ")
    // Squared error
    println(s"MSE = ${metrics.meanSquaredError}")
    println(s"RMSE = ${metrics.rootMeanSquaredError}")

    // R-squared
    println(s"R-squared = ${metrics.r2}")

    // Mean absolute error
    println(s"MAE = ${metrics.meanAbsoluteError}")

    // Explained variance
    println(s"Explained variance = ${metrics.explainedVariance}")


    // Save the Metrics:
    if(configApp.saveOutputMeasures != null){
      sc.parallelize(Seq(s"MSE = ${metrics.meanSquaredError}", s"RMSE = ${metrics.rootMeanSquaredError}",
        s"R-squared = ${metrics.r2}", s"MAE = ${metrics.meanAbsoluteError}",
        s"Explained variance = ${metrics.explainedVariance}"))
        .coalesce(1)
        .saveAsTextFile(configApp.saveOutputMeasures)
    }

    // Predict (And save the df)
    if(configApp.predict != null){
      println(s"\n\nPredicting...:\n\n")

      val dfToPredict = loadDataFrame(spark, configApp.predict, training = false)
      val colsToDrop = modelGen.getIntermediateColumnNames
      val featureDfToPredict = dataTransformer.transform(dfToPredict)

      modelTrained.bestModel.transform(featureDfToPredict)
        .drop(colsToDrop: _*)
        .write
        .option("header", "true")
        .csv(configApp.predictedDfPath)
    }

    // SQL CLI
    if(configApp.promptSqlCLI){
        evaluationDf.show(20, truncate = false)
        println("Validation View Name is: data")
        evaluationDf.createTempView("data")
      while (true) {
        println("Enter query: ")
        val sql_query = scala.io.StdIn.readLine()
        if (sql_query.matches("^:q.*"))
          break
        try {
          spark.sql(sql_query: String).show(100, truncate = false)
        } catch {
          case e: Exception => e.printStackTrace()
        }
      }
    }

    spark.stop
  }

}
