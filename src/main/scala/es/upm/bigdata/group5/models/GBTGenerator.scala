package es.upm.bigdata.group5.models

import es.upm.bigdata.group5.utilities.Constants._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.ParamGridBuilder

class GBTGenerator(distinctAirports:Int, stepSizeParamValues: Array[Double], maxIterParamValues: Array[Int]) extends ModelGenerator {

  private val this.stepSizeParamValues = stepSizeParamValues
  private val this.maxIterParamValues = maxIterParamValues

  private val gbTree = new GBTRegressor()
    .setLabelCol(ColArrDelay)
    .setFeaturesCol("features")
    .setMaxBins(distinctAirports)

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

  def this(distinctAirports: Int) = this(distinctAirports,  Array(0.3, 0.5, 0.8), Array(10,15))

  override def getDataTransformer: Pipeline = new Pipeline()
    .setStages(Array(stringIndexer, vectorization, indexer, scaler))

  override def getModel: Pipeline = new Pipeline().setStages(Array(gbTree))

  override def getParamGridBuilt: Array[ParamMap] =
    new ParamGridBuilder()
      .addGrid(gbTree.stepSize, stepSizeParamValues)
      .addGrid(gbTree.maxIter, maxIterParamValues)
      .build()

  override def getIntermediateColumnNames: List[String] =
    super.getIntermediateColumnNames ++ List("featuresNotIndexed", "featuresNotSC",
      ColOrigin+ "_n", "featuresNotIndexed")
}


