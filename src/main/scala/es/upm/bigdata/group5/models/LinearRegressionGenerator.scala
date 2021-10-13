package es.upm.bigdata.group5.models

import es.upm.bigdata.group5.utilities.Constants._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.ParamGridBuilder

class LinearRegressionGenerator(elasticNetParamValues: Array[Double], regParamValues: Array[Double]) extends ModelGenerator {

  private val this.elasticNetParamValues = elasticNetParamValues
  private val this.regParamValues = regParamValues

  private val regression = new LinearRegression()
    .setLabelCol(ColArrDelay)
    .setFeaturesCol("features")
    .setMaxIter(10)

  private val cols = Array(ColDepTime, ColOrigin + "_numb",
    ColDepDelay, ColTaxiOut)

  private val vectorizer = new VectorAssembler()
    .setInputCols(cols)
    .setOutputCol("featuresNotSC")

  private val scaler = new StandardScaler()
    .setInputCol("featuresNotSC")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(false)

  private val stringIndexer = new StringIndexer()
    .setInputCols(Array(ColOrigin))
    .setOutputCols(Array(ColOrigin + "_n"))
    .setHandleInvalid("keep")

  private val oneHotEncoder = new OneHotEncoder()
    .setInputCols(Array(ColOrigin + "_n"))
    .setOutputCols(Array(ColOrigin + "_numb"))

  def this() = this(Array(0.2, 0.5, 0.8), Array(0, 0.3, 0.5))

  override def getDataTransformer: Pipeline = new Pipeline()
      .setStages(Array(stringIndexer, oneHotEncoder, vectorizer, scaler))

  override def getModel: Pipeline = new Pipeline()
      .setStages(Array(regression))

  override def getParamGridBuilt: Array[ParamMap] =
    new ParamGridBuilder()
      .addGrid(regression.elasticNetParam, elasticNetParamValues)
      .addGrid(regression.regParam, regParamValues)
      .build()

  override def getIntermediateColumnNames: List[String] =
    super.getIntermediateColumnNames ++ List("featuresNotSC",
      ColOrigin+ "_n", ColOrigin + "_numb" )
}
