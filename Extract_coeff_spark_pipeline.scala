$bash> spark-shell --driver-memory 16G --executor-memory 20G --executor-cores 8 --num-executors 20 --jars sr_scala_2.11-0.1.jar,ml-utils_2.11-1.0.10-SNAPSHOT.jar

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.{VectorAssembler, StandardScalerModel, NumImputerModel} 
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.attribute.AttributeGroup

val model = PipelineModel.load("/data/tdc/datalab/SR/1.Service/customer/modelFit/cust_service_v2_sparkModel_030718")
val df = spark.table("tdcprdrs_app_sr.daily_smart_routing_sr_customer_siebel_2018_08_07")
val schema = model.transform(df).schema

val lrModel = model.stages.last.asInstanceOf[LogisticRegressionModel]
val imputer = model.stages(0).asInstanceOf[NumImputerModel]
val va = model.stages(1).asInstanceOf[VectorAssembler]
val ss = model.stages(2).asInstanceOf[StandardScalerModel]


val num_size = va.getInputCols.size
val ssOutputCols = for (i <- List.range(0, num_size)) yield s"num_features_scaled_$i"
val vaOutputCols = for (i <- List.range(0, num_size)) yield s"num_features_$i"

val ssMap = ssOutputCols.zip(vaOutputCols).toMap
val vaMap = vaOutputCols.zip(va.getInputCols).toMap
val imputerMap = imputer.getOutputCols.zip(imputer.getInputCols).toMap

val featureAttrs = AttributeGroup.fromStructField(schema(lrModel.getFeaturesCol)).attributes.get
val lrfeatures = featureAttrs.map(_.name.get)

val inputs = for (f <- lrfeatures) yield imputerMap.getOrElse(vaMap.getOrElse(ssMap.getOrElse(f, f), f), f)

val featureNames: Array[String] = if (lrModel.getFitIntercept) {
  Array("(Intercept)") ++ inputs
} else {
  inputs
}

val lrModelCoeffs = lrModel.coefficients.toArray
val coeffs = if (lrModel.getFitIntercept) {
  Array(lrModel.intercept) ++ lrModelCoeffs
} else {
  lrModelCoeffs
}

featureNames.zip(coeffs).foreach {case (feature, coeff) => println(s"$feature,$coeff")}


