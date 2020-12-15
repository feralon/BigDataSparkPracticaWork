package es.upm.bigdata.group5.models

import es.upm.bigdata.group5.utilities.Constants._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.ParamGridBuilder

class RandomForestGenerator(distinctAirports:Int, maxDepthParamValues: Array[Int], numTreesParamValues: Array[Int]) extends ModelGenerator {

  private val this.maxDepthParamValues = maxDepthParamValues
  private val this.numTreesParamValues = numTreesParamValues

  private val rf = new RandomForestRegressor()
    .setLabelCol(ColArrDelay)
    .setFeaturesCol("features")
    .setMaxBins(distinctAirports)
    .setBootstrap(true)

  private val cols = Array(ColDepTime, ColOrigin + "_n",
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

  def this(distinctAirports: Int) = this(distinctAirports,  Array(5, 7, 9), Array(20, 30, 35))

  override def getDataTransformer: Pipeline = new Pipeline()
      .setStages(Array(stringIndexer, vectorization, indexer, scaler))

  override def getModel: Pipeline = new Pipeline()
      .setStages(Array(rf))

  override def getParamGridBuilt: Array[ParamMap] =
    new ParamGridBuilder()
      .addGrid(rf.maxDepth, maxDepthParamValues)
      .addGrid(rf.numTrees, numTreesParamValues)
      .build()

  override def getIntermediateColumnNames: List[String] =
          super.getIntermediateColumnNames ++ List("featuresNotIndexed", "featuresNotSC",
            ColOrigin+ "_n")
}
