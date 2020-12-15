package es.upm.bigdata.group5.models

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Pipeline}

trait ModelGenerator {
  /**
   * Gets Pipeline with data transformations
   * @return Pipeline with data transformations
   */
  def getDataTransformer: Pipeline

  /**
   * Gets Pipeline with the model
   * @return Pipeline with the model
   */
  def getModel: Pipeline

  /**
   * Gets ParamMap variables for Hyperparameter tuning
   * @return ParamMap variables for Hyperparameter tuning
   */
  def getParamGridBuilt: Array[ParamMap]

  /**
   * Gets the names of intermediate columns generated for the model
   * @return  the names of intermediate columns generated for the model
   */
  def getIntermediateColumnNames: List[String] = Seq("features").toList
}
