---
title: "For PySpark text classification, should training vocabulary be used for testing, or should a separate test vocabulary serve as features?"
date: "2024-12-23"
id: "for-pyspark-text-classification-should-training-vocabulary-be-used-for-testing-or-should-a-separate-test-vocabulary-serve-as-features"
---

Alright, let’s unpack this. I’ve certainly seen my share of text classification pipelines in PySpark, and the question of training vs. testing vocabulary always warrants careful consideration. From my experience, going down the path of using training vocabulary for your test set can seem like a quicker route, but it's often riddled with potential pitfalls, particularly when you're aiming for robust, real-world performance. I'll break down the reasons why and show you some code examples.

Fundamentally, the problem stems from data leakage. If we use the same vocabulary derived from our training data for feature extraction on our test data, we're effectively informing the test set about the distribution of our training set. This can lead to inflated performance metrics on the test data that do not accurately reflect how the model will perform on unseen data in real-world deployment scenarios. It’s like a student looking at the answer key while taking a test, which defeats the purpose of assessing actual comprehension.

The correct practice, in almost all cases, involves deriving a vocabulary solely from your training data, and then using *that* vocabulary to encode *both* your training and test sets. Think of it as creating a dictionary from your training set’s word frequency that then gets applied consistently. This ensures that your model's evaluation is a true reflection of its generalization capability.

Now, let’s get into some specific code examples to illustrate what I mean. We'll use PySpark's ML library. These examples focus on a simplified process, and in reality, you’ll have to build out the preprocessing and hyperparameter tuning steps, but this is to get to the core idea:

**Example 1: Incorrect Method – Using Test Data to Build Vocabulary**

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName("text_classification").getOrCreate()

# Sample Data (Replace with your actual dataset)
training_data = spark.createDataFrame([
    (0, "this is a great document"),
    (1, "this document is very poor"),
    (0, "another amazing piece of text"),
    (1, "a terrible document"),
], ["label", "text"])

test_data = spark.createDataFrame([
    (0, "a wonderful paper"),
    (1, "this is a bad read"),
], ["label", "text"])

# --- Incorrect Approach ---
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=1000) # Set numFeatures explicitly to ensure deterministic results.
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.01)

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

# Build a 'global' pipeline using both datasets.
combined_data = training_data.union(test_data).withColumn('isTraining', lit(True))
combined_data_pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
combined_model = combined_data_pipeline.fit(combined_data)

# Transforming combined data, then splitting. The issue here is that our model's pipeline is aware of the test vocabulary
transformed_combined_data = combined_model.transform(combined_data)
transformed_training = transformed_combined_data.filter('isTraining')
transformed_test = transformed_combined_data.filter('isTraining == False')
model = lr.fit(transformed_training)

# This will be inaccurate performance
predictions = model.transform(transformed_test)
predictions.show() # Display for verification purposes

```

In this problematic example, we implicitly build a model on both the training and test dataset, and this is the mistake. The hashingTF and IDF stages of the pipeline will get information about the test dataset, which then results in our model training and prediction using the same dictionary from training set with the addition of words in the test set. The `fit` step from combined_data_pipeline builds its vocabulary on combined data and then applies that vocabulary on training and test set separately.

**Example 2: Correct Method – Using Training Vocabulary for Testing**

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("text_classification").getOrCreate()

# Sample Data (Replace with your actual dataset)
training_data = spark.createDataFrame([
    (0, "this is a great document"),
    (1, "this document is very poor"),
    (0, "another amazing piece of text"),
    (1, "a terrible document"),
], ["label", "text"])

test_data = spark.createDataFrame([
    (0, "a wonderful paper"),
    (1, "this is a bad read"),
], ["label", "text"])


# --- Correct Approach ---
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=1000) # Set numFeatures explicitly to ensure deterministic results.
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.01)

# Construct Pipeline and fit to training only.
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

# Only fit to the training dataset.
model = pipeline.fit(training_data)


transformed_training = model.transform(training_data)
transformed_test = model.transform(test_data)

model_trained = lr.fit(transformed_training)

predictions = model_trained.transform(transformed_test)
predictions.show()

```

This second example shows the correct way. We train our transformation pipeline `model` strictly on the training dataset. This ensures that our feature set for the training and test datasets is based on vocabulary from the training data. By applying this transformation separately to both train and test data sets, we maintain the separation and proper evaluation. The Logistic Regression `model_trained` will then use the transformed training data to train the model, and then apply it to the transformed test dataset.

**Example 3: Dealing with Out-of-Vocabulary Words**

Now, it is possible that the test dataset will contain words that are not in the training set's vocabulary. In practice, hashingTF and IDF usually handle such words, simply by hashing it to an already existing bucket and assigning it a zero IDF score if it was never seen in the training set. However, you need to be aware of how it can affect your results. For example, let's see a modified test dataset and a slight change to the previous example:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("text_classification").getOrCreate()

# Sample Data (Replace with your actual dataset)
training_data = spark.createDataFrame([
    (0, "this is a great document"),
    (1, "this document is very poor"),
    (0, "another amazing piece of text"),
    (1, "a terrible document"),
], ["label", "text"])

test_data = spark.createDataFrame([
    (0, "a wonderful manuscript"),
    (1, "this is an awful read"),
    (1, "a very dreadful essay"), # Added new record for illustration
], ["label", "text"])

# --- Correct Approach ---
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.01)

# Construct Pipeline and fit to training only.
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

# Only fit to the training dataset.
model = pipeline.fit(training_data)


transformed_training = model.transform(training_data)
transformed_test = model.transform(test_data)

model_trained = lr.fit(transformed_training)

predictions = model_trained.transform(transformed_test)
predictions.show()

```

In this example, words like 'manuscript,' 'awful', and 'dreadful' are not present in the training data. The hashingTF and IDF stages will encode them based on the pre-built vocabulary, but effectively, they are treated as *unknown* terms. Depending on your task and vocabulary, this can impact accuracy. Dealing with out-of-vocabulary (OOV) words robustly is a complex topic, often involving pre-trained embeddings and techniques that are beyond the scope of this question. If, in practice, you find it’s a significant problem, there are ways to handle this, such as building a more comprehensive training set, incorporating subword tokenization or using pre-trained word embeddings, but you have to be mindful of using the training vocabulary to produce those embeddings or any other preprocessing needed.

**Recommended Resources**

For a deeper dive into the theoretical underpinnings of text classification and feature engineering, I'd recommend looking into:

1.  **"Speech and Language Processing" by Daniel Jurafsky and James H. Martin:** A comprehensive textbook covering the whole field of Natural Language Processing, including thorough sections on text representation and feature extraction for machine learning. It provides the theoretical basis for many techniques.
2.  **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** While not specifically about PySpark, this book offers invaluable insights into the practical implementation of machine learning pipelines and the importance of proper validation techniques (including understanding data leakage). It can give you practical guidance about general implementation practices that apply regardless of the tool used.
3.  **Original PySpark ML documentation:** Always start with the official documentation for `pyspark.ml`, particularly the sections concerning feature extraction and transformations. This will explain exactly how the library's functions are implemented.
4. **Research papers on word embeddings** such as Word2Vec or FastText, as these delve deeper into the strategies for handling vocabulary and unknown words when they become an issue. Look them up on Google Scholar or other research databases.

In summary, always strive to create your vocabulary exclusively from your training data and consistently apply that vocabulary for both training and testing, if your objective is to produce a trustworthy classifier. It is an essential practice for constructing robust and reliable text classification models in any framework, including PySpark.
