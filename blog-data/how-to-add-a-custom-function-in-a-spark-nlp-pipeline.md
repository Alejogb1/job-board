---
title: "How to add a custom function in a SPARK NLP Pipeline?"
date: "2024-12-15"
id: "how-to-add-a-custom-function-in-a-spark-nlp-pipeline"
---

alright, so, you're trying to shoehorn your own custom function into an spark nlp pipeline? i get it. been there, done that. it's not exactly a walk in the park but definitely doable. i remember the first time i tried this, i was working on a project classifying support tickets at a previous company, you can think of it like this: 'customer is angry because his account is blocked', things like that. we had this really specific need to handle some edge cases, stuff that the out-of-the-box spark nlp stuff just couldn't touch, it was for example extracting very specific combinations of entities which were a little out of what a standard NER model could handle, and those were key to predict a final classification.

the core problem is that spark nlp pipelines are designed around these transformers and annotators, and they expect data to flow through them in a particular way. your custom thing, that's probably not going to fit in that framework, at least not without a little bit of persuasion.

so, here's how i've generally tackled this. the trick is to basically wrap your function into something spark nlp can understand. it usually means creating a custom transformer. it's not the fanciest solution, but it is generally the most flexible one, for what i need.

first, let’s go over a quick explanation of what a transformer in spark nlp is, it takes a dataframe that contains at least a column with text and spits out a new dataframe with the same columns and also new columns with annotations, these annotations can have any structure that spark nlp's library provides such as word, document, sentence, and so on.

let's assume you've got this python function that does the magic, let's call it 'my_custom_function', and it takes a string and returns a list of something:

```python
def my_custom_function(text):
  """
    this is your custom function
    it takes a string and should return a list.
  """
  # do some awesome processing of your text here
  # let's say you want to return tokens that start with "a"
  tokens = text.split()
  return [token for token in tokens if token.startswith("a")]
```

now, this function, as it is, will not work out of the box, to work with spark nlp you need to wrap it inside a class that inherits from `Transformer`. let’s do that. this is how i've structured it previously when i had similar problems. the class should follow all the spark nlp transformer expectations:

```python
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

class CustomTransformer(Transformer, HasInputCol, HasOutputCol):

  def __init__(self, inputCol=None, outputCol=None, function=None):
      super(CustomTransformer, self).__init__()
      self.inputCol = inputCol
      self.outputCol = outputCol
      self.function = function
      self._setDefault(inputCol=None, outputCol=None)


  def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        udf_function = F.udf(self.function, ArrayType(StringType()))
        return dataset.withColumn(output_col, udf_function(F.col(input_col)))
```

now, let's break this down. we're creating a class called `CustomTransformer`, which inherits from `Transformer` and the parameter classes `HasInputCol, HasOutputCol`. This makes our transformer play nice with the rest of the spark nlp machinery. the `__init__` function is just the constructor, we are setting the input and output columns that we'll use later, and the custom function. the `_transform` is where the magic happens, this function is called when we `fit` and `transform` our data.

here, we grab the input and output column names using `getInputCol()` and `getOutputCol()`, and finally, we create an `udf` (user defined function), which basically tells spark to wrap the python function so it can be executed in spark's context. then, we call the udf and create a new column that will be named after what you set on the output column, which will be the result of the udf function.

finally, to add this to your pipeline, you just instantiate this transformer with the right configuration. check out this example:

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import DocumentAssembler, SentenceDetector
from sparknlp.base import LightPipeline

# Create a spark session
spark = SparkSession.builder.appName("CustomFunction").getOrCreate()

# Create a simple DataFrame
data = spark.createDataFrame([("apple a day keeps the doctor away",),("another amazing apple is in the basket",)], ["text"])

# Create an instance of our CustomTransformer
custom_transformer = CustomTransformer(
    inputCol="text",
    outputCol="custom_output",
    function=my_custom_function
)

# Spark NLP pipeline
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    custom_transformer
])

# Fit and transform data
model = pipeline.fit(data)
transformed_data = model.transform(data)

transformed_data.select("text","custom_output").show(truncate = False)

light_model = LightPipeline(model)
light_result = light_model.annotate("a new apple is in the fridge")
print(light_result)
```

in this code, we created a spark session, then created a dummy dataframe, then, we instantiated the `CustomTransformer` that we created earlier, passing the input and output column names, and the custom function. Then we create a standard spark nlp pipeline using the document assembler and sentence detector, and finally we add our custom transformer. we proceed to fit and transform the data and print a sample of the results. in the end, we create a light pipeline that we can use to process a single sentence.

now, there are a few things to consider here. first, for the python `udf`s to work, the function needs to be picklable, this can create problems if you use complex objects in the function. second, depending on what your function does, it can be a performance bottleneck, so, if possible, try to use spark operations in the udf function to speed up things.

if your custom function needs access to the annotations from previous steps in the pipeline, that's a different story. in that case, you'd need to define your `_transform` method to accept `Annotation` objects as input and process them. that goes a little deeper, but it's usually similar.

also, just so you know, the documentation on this, while available, sometimes feels a bit, let's say, "sparse," so, be sure to experiment a bit and read the source code if you get stuck. some of the documentation can feel like the authors were too busy building the stuff to actually, you know, document it all. if you want to dive deep, i recommend the book "natural language processing with python" from bird, klein and loper, which is really thorough in python nlp in general, but it also has information about building custom functions and understanding how pipelines work. also, check out the spark documentation, while it does not explain much about how to use spark nlp, it goes in-depth in how to use dataframes, `udfs`, transformers and pipelines. these are a good starting points, i believe. and of course, there is also the source code, sometimes is the only place where you can find answers, in particular, the `sparknlp` github repo.

and, if you find yourself thinking that the spark nlp library seems to have some parts built by different teams that never talked to each other, you are not alone. i mean, it's powerful, but sometimes is a little bit... surprising.

so, yeah, that’s basically it. you're taking your custom python thing, you wrap it in a transformer, and boom, it's now part of your spark nlp pipeline. it's not exactly rocket science, but a bit more involved than just plugging things in, but that is usually how i did it. good luck, and remember to back up your data, it is not a bad practice.
