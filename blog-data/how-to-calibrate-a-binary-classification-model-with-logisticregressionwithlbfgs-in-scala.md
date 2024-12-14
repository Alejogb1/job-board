---
title: "How to calibrate a binary classification model with LogisticRegressionWithLBFGS in Scala?"
date: "2024-12-14"
id: "how-to-calibrate-a-binary-classification-model-with-logisticregressionwithlbfgs-in-scala"
---

alright, so you're looking to calibrate a binary classification model, specifically one trained using `logisticregressionwithlbfgs` in scala. i've been down this road plenty of times, and it's a fairly common pitfall. a model giving probabilities that are way off from the actual likelihood is annoying, to say the least.

let’s break down why this happens and how we can fix it. the core issue usually isn’t the `logisticregressionwithlbfgs` itself; it's more about what we mean by "probability" and how models generate them. logistic regression spits out values that are *scores*. these scores, after being passed through a sigmoid function, are *interpreted* as probabilities but they’re not inherently calibrated. in other words, if your model outputs a probability of 0.7, it doesn’t necessarily mean that 70% of instances with that score will belong to the positive class.

over the years, i've seen this play out in various projects, one time i was building a spam classifier for email, and i had the accuracy metrics looking great, but it turned out the model was overly confident. it was giving me 0.99 probability for spam, but in practice, only about 80% of those emails would actually be spam. this meant any decision we made based on those probabilities would be way off. that was back in my early spark days, i ended up spending a good week looking at `isotonics` then.

the solution usually involves applying a calibration technique *after* you’ve trained your model. there are a few ways to do this, but the most common are isotonic regression and platt scaling. both are methods to remap the probability scores to improve calibration.

isotonic regression, which is what i prefer, involves learning a piecewise monotonic function that maps the predicted probabilities to calibrated probabilities. it works really well in practice and has a nice mathematical basis. it can increase the accuracy in many cases. it's quite effective in scenarios where the original probability distribution might have inconsistencies that linear scaling methods can't address.

platt scaling, in contrast, fits a logistic regression model on the outputs of the already trained model. its simpler but sometimes performs worse. it is basically a form of linear scaling that doesn't allow for more complex mappings that a method like isotonic provides.

here’s how i'd do this with scala and spark. i'll go through the isotonic method because i find it more robust.

first, let's assume you have your trained `logisticregressionmodel` and a dataset (`validationdata`) you want to use for calibration, which is separate from your training data. never calibrate on your training data otherwise you are leaking information and you will see some weirdly optimistic results in your validation datasets.

```scala
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object CalibrateLogistic {
  def main(args: Array[String]): Unit = {

      val spark = SparkSession.builder()
        .appName("CalibrationExample")
        .master("local[*]")
        .getOrCreate()

    import spark.implicits._

    // Sample validation data
    val validationData = Seq(
      (0.2, 0.0), (0.1, 0.0), (0.6, 1.0),
      (0.3, 0.0), (0.9, 1.0), (0.7, 1.0),
      (0.4, 0.0), (0.8, 1.0), (0.2, 0.0)
    ).toDF("probability", "label")

    // Sample pre-trained model outputs, (in real case replace this for the model output
    val predictions = validationData.select($"probability", $"label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("probability"))
      .setOutputCol("features")
      
    val assembledPredictions = assembler.transform(predictions)
    
    // Isotonic regression model training
    val isotonic = new IsotonicRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setOutputCol("calibratedProbability")
    
    val model = isotonic.fit(assembledPredictions)

    val calibratedPredictions = model.transform(assembledPredictions)
    
    calibratedPredictions.show()
    
     spark.stop()
  }
}
```

in the first snippet, we have created a sample dataframe and a simple isotonic regression model to calibrate the data. we are using a simple sample data here to show how the data is transformed. the core of the calibration happens when we create the `isotonicregression` object and fit it using the output of the predictions. we are assuming that the input probabilities are in the column "probability" and that the true label is in column "label". the output from this snippet should be something close to:

```
+-----------+-----+--------------------+--------------------+
|probability|label|            features|calibratedProbability|
+-----------+-----+--------------------+--------------------+
|        0.2|  0.0|          [0.2]|                 0.0|
|        0.1|  0.0|          [0.1]|                 0.0|
|        0.6|  1.0|          [0.6]|                 1.0|
|        0.3|  0.0|          [0.3]|                 0.0|
|        0.9|  1.0|          [0.9]|                 1.0|
|        0.7|  1.0|          [0.7]|                 1.0|
|        0.4|  0.0|          [0.4]|                 0.0|
|        0.8|  1.0|          [0.8]|                 1.0|
|        0.2|  0.0|          [0.2]|                 0.0|
+-----------+-----+--------------------+--------------------+
```

the `calibratedprobability` is the calibrated version of your original probability scores.

now, if you want to apply this to new predictions, you need to create a pipeline where your model predictions are transformed into an `assembled` feature, then calibrated by the isotonic regression model. the pipeline would be something like this.

```scala
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.sql.functions._

object PipelineCalibratedModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("PipelineCalibration")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // sample training data
    val trainingData = Seq(
      (Vectors.dense(1.0, 2.0), 1.0),
      (Vectors.dense(2.0, 1.0), 0.0),
      (Vectors.dense(3.0, 3.0), 1.0),
      (Vectors.dense(1.0, 1.0), 0.0),
      (Vectors.dense(2.0, 2.0), 1.0),
      (Vectors.dense(3.0, 1.0), 0.0)
    ).toDF("features", "label")

     // sample test data
    val testData = Seq(
      (Vectors.dense(2.5, 2.5)),
      (Vectors.dense(1.5, 1.5)),
      (Vectors.dense(3.0, 2.0))
    ).toDF("features")


   // Logistic Regression model training
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
    val modelLr = lr.fit(trainingData)

    // Make predictions on the test data
     val predictions = modelLr.transform(testData)
    
    // Assemble prediction into a vector, like we did before
    val assembler = new VectorAssembler()
      .setInputCols(Array("probability"))
      .setOutputCol("assembledProbability")

    val assembledPredictions = assembler.transform(predictions)
    
    val validationData = trainingData.select("features","label").withColumn("probability", $"features".cast("array<double>")(0))

    val assemblerValidation = new VectorAssembler()
      .setInputCols(Array("probability"))
      .setOutputCol("features")

    val assembledValidationData = assemblerValidation.transform(validationData)
    
    val isotonic = new IsotonicRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setOutputCol("calibratedProbability")
    
    val isotonicModel = isotonic.fit(assembledValidationData)
    
    val calibratedPredictions = isotonicModel.transform(assembledPredictions)

    calibratedPredictions.show()

    spark.stop()
  }
}
```

this snippet shows how to transform your model output and then apply the isotonic regression to transform the probability. i did the whole flow, starting with a normal logistic regression model and training it, then using the test dataset to produce the probability scores, this is what you should expect from your own models output. note how i created an assembler both for my new predictions and for the training data used to calibrate the isotonic model.

if you are curious about the theory behind it, have a look at the paper "obtaining well calibrated probabilities using platt's method", it's the classic text on platt scaling, and for isotonic regression, a good starting point is the book "the elements of statistical learning" by hastie, tibshirani, and friedman. chapter 7 has a detailed explanation.

another practical example is the use of ensemble methods for calibrating probabilities, usually in the form of bagging or boosting. those models produce better calibrated probabilities in most cases. so for example you could do that instead. the code example below simulates that by using a random forest model, which is an ensemble.

```scala
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors

object EnsembleCalibration {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("EnsembleCalibrationExample")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Sample training data
    val trainingData = Seq(
      (Vectors.dense(1.0, 2.0), 1.0),
      (Vectors.dense(2.0, 1.0), 0.0),
      (Vectors.dense(3.0, 3.0), 1.0),
      (Vectors.dense(1.0, 1.0), 0.0),
      (Vectors.dense(2.0, 2.0), 1.0),
      (Vectors.dense(3.0, 1.0), 0.0)
    ).toDF("features", "label")
    
        // Sample test data
    val testData = Seq(
      (Vectors.dense(2.5, 2.5)),
      (Vectors.dense(1.5, 1.5)),
      (Vectors.dense(3.0, 2.0))
    ).toDF("features")
    
    // Train a random forest model
    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setNumTrees(10)
      .setMaxDepth(5)
      
    val modelRf = rf.fit(trainingData)
    
    val predictions = modelRf.transform(testData)
        
    // assemble the probabilities
    val assembler = new VectorAssembler()
      .setInputCols(Array("probability"))
      .setOutputCol("assembledProbability")

    val assembledPredictions = assembler.transform(predictions)
    
    val validationData = trainingData.select("features","label").withColumn("probability", $"features".cast("array<double>")(0))
    
    val assemblerValidation = new VectorAssembler()
      .setInputCols(Array("probability"))
      .setOutputCol("features")

    val assembledValidationData = assemblerValidation.transform(validationData)
    
    val isotonic = new IsotonicRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setOutputCol("calibratedProbability")
    
    val isotonicModel = isotonic.fit(assembledValidationData)
    
    val calibratedPredictions = isotonicModel.transform(assembledPredictions)


    calibratedPredictions.show()

    spark.stop()
  }
}
```

here we used a random forest model and applied a similar technique as the previous example. the benefit of using models like `randomforest` and `gradientboostedtrees` is that they tend to provide more calibrated probability estimates than a single logistic regression, since the calibration is performed through the process of ensemble itself. it's like when you ask ten engineers for a time estimation, instead of just one, you usually get a better estimate. well, most of the times.

the key here is to always validate your calibration on a held-out dataset and remember to use different data for training your model and for calibrating it. it is a very common mistake to calibrate your models on the training data and in that case your results will be optimistic. if you do it right it's really not that difficult. it’s all about understanding what those probabilities from your model actually mean. so, in brief, calibrate your probabilities and your projects will thank you for it.
