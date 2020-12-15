package es.upm.bigdata.group5.utilities

/**
 * Constant values definition
 */
object Constants {

  // Args
  val ArgInputFilePaths = "--inputFilePaths"
  val ArgTrainingModel= "--trainingModel"
  val ArgSampleTrainingDf ="--sampleTrainingDf"
  val ArgSaveOutputMeasures = "--saveOutputMeasures"
  val ArgPromptSqlCLI = "--promptSqlCLI"
  val ArgPredict = "--predict"
  val ArgPredictedDfPath = "--predictedDfPath"

  //Models
  val RandomForestModelName =  "RandomForest"
  val GBTModelName =  "GBT"
  val DecisionTreeModelName =  "DecisionTree"
  val LinearRegressionModelName =  "LinearRegression"

  // Columns
  val ColYear = "Year"
  val ColMonth = "Month"
  val ColDayOfMonth = "DayofMonth"
  val ColDayOfWeek = "DayOfWeek"
  val ColDepTime = "DepTime"
  val ColCRSDepTime = "CRSDepTime"
  val ColArrTime = "ArrTime"
  val ColCRSArrTime = "CRSArrTime"
  val ColUniqueCarrier = "UniqueCarrier"
  val ColFlightNum = "FlightNum"
  val ColTailNum = "TailNum"
  val ColActualElapsedTime = "ActualElapsedTime"
  val ColCRSElapsedTime = "CRSElapsedTime"
  val ColAirTime = "AirTime"
  val ColArrDelay = "ArrDelay"
  val ColDepDelay = "DepDelay"
  val ColOrigin = "Origin"
  val ColDest = "Dest"
  val ColDistance = "Distance"
  val ColTaxiIn = "TaxiIn"
  val ColTaxiOut = "TaxiOut"
  val ColCancelled = "Cancelled"
  val ColCancellationCode = "CancellationCode"
  val ColDiverted = "Diverted"
  val ColCarrierDelay = "CarrierDelay"
  val ColWeatherDelay = "WeatherDelay"
  val ColNASDelay = "NASDelay"
  val ColSecurityDelay = "SecurityDelay"
  val ColLateAircraftDelay = "LateAircraftDelay"
  val ColDepHour = "DepHour"
  val ColDepMin = "DepMin"
}
