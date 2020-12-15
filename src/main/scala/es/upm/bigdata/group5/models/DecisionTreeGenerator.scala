package es.upm.bigdata.group5.models

import es.upm.bigdata.group5.utilities.Constants._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.ParamGridBuilder

class DecisionTreeGenerator(distinctAirports:Int, maxDepthParamValues: Array[Int], maxBinsParamValues: Array[Int]) extends ModelGenerator {


  private val this.maxDepthParamValues = maxDepthParamValues
  private val this.maxBinsParamValues = maxBinsParamValues


  private val dt = new DecisionTreeRegressor()
    .setLabelCol(ColArrDelay)
    .setFeaturesCol("features")

  private val cols = Array(ColDepHour, ColDepMin, ColOrigin + "_n",
    ColDepDelay, ColTaxiOut)

  private val vectorization = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("featuresNotIndexed")

  private val indexer = new VectorIndexer()
    .setInputCol("featuresNotIndexed")
    .setOutputCol("featuresNotSC")
    .setMaxCategories(distinctAirports)

  private val scaler = new StandardScaler()
    .setInputCol("featuresNotSC")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(false)

  private val stringIndexer = new StringIndexer()
    .setInputCols(Array(ColOrigin))
    .setOutputCols(Array(ColOrigin + "_n"))
    .setHandleInvalid("keep")

  def this(distinctAirports: Int) = this(distinctAirports, Array(3, 5), Array(32, distinctAirports / 2,
    distinctAirports))

  override def getDataTransformer: Pipeline = new Pipeline()
    .setStages(Array(stringIndexer, vectorization, indexer, scaler))

  override def getModel: Pipeline = new Pipeline().setStages(Array(dt))


  override def getParamGridBuilt: Array[ParamMap] =
    new ParamGridBuilder()
      .addGrid(dt.maxDepth, maxDepthParamValues)
      .addGrid(dt.maxBins, maxBinsParamValues)
      .build()

  override def getIntermediateColumnNames: List[String] =
    super.getIntermediateColumnNames ++ List("featuresNotIndexed", "featuresNotSC",
      ColOrigin+ "_n")
}
