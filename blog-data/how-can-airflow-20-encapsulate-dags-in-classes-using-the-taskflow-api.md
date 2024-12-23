---
title: "How can Airflow 2.0 encapsulate DAGs in classes using the Taskflow API?"
date: "2024-12-23"
id: "how-can-airflow-20-encapsulate-dags-in-classes-using-the-taskflow-api"
---

Let's tackle this one; it's a question I've seen pop up a fair bit, and it’s often an indicator of a developer moving towards more robust and maintainable Airflow deployments. I remember a particularly complex project a couple of years back involving multiple data integrations that really highlighted the need for better DAG organization. The sheer volume of code in those monolithic DAG files was making even basic updates a perilous venture. That's where class-based DAG encapsulation really shone, and the Taskflow API in Airflow 2.0 makes it much more elegant than previous iterations. So, let’s break down how you can approach this.

The key here is transitioning from procedural DAG definitions to a more object-oriented approach. Instead of a sprawling set of globally defined tasks and operators, we wrap related tasks and their configurations within a class. This encapsulates logical units of work, promoting reusability and reducing the chance of name clashes, while improving overall code readability. The Taskflow API, with decorators like `@task` and `@dag`, is central to this process.

Think of the core benefit as improved modularity. Rather than having your entire pipeline logic crammed into a single script, we’re creating self-contained units. This separation of concerns makes it easier to reason about individual parts of your workflow, facilitating testing and making debugging less of a headache. If you adhere to good design principles within those classes, you'll be setting yourself up for easier maintenance and scaling down the line.

Let’s delve into some concrete examples to solidify this. First, we’ll create a simple example where a processing class orchestrates a few basic tasks:

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.models import BaseOperator

class DataProcessor:

    def __init__(self, base_dir, source_type):
      self.base_dir = base_dir
      self.source_type = source_type

    @task()
    def extract(self):
        print(f"Extracting data from {self.base_dir}/{self.source_type}...")
        return {"status": "extracted", "location": f"{self.base_dir}/{self.source_type}"}

    @task()
    def transform(self, extraction_result):
       if extraction_result["status"] == "extracted":
           print(f"Transforming data at {extraction_result['location']}...")
           return {"status": "transformed"}
       else:
           raise ValueError("Invalid extraction status")


    @task()
    def load(self, transform_result):
      if transform_result["status"] == "transformed":
          print("Loading transformed data...")
          return {"status":"loaded"}
      else:
          raise ValueError("Invalid transform status")


@dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None, tags=["example", "class-dag"])
def class_based_dag():

    processor = DataProcessor(base_dir="/data", source_type="csv")
    extraction = processor.extract()
    transformation = processor.transform(extraction)
    processor.load(transformation)


class_based_dag()

```

In this first snippet, `DataProcessor` encapsulates extraction, transformation, and loading tasks. The `@task` decorator makes these methods Airflow tasks. The DAG is concise and simply orchestrates the processing flow, which remains encapsulated inside the class. Notice how dependencies are handled implicitly by the Taskflow API using the return values of the function. Also, we are passing an instance of our processor into the DAG function, and from there the `@task` methods are called.

Let’s take a step further and consider that you might have specific configurations for different types of tasks. You might need different connection settings for extracting data from different databases. Here’s how you could adapt the class to handle that, using the `BaseOperator` class:

```python
from airflow.decorators import dag
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from datetime import datetime

class DatabaseExtractor(BaseOperator):

    template_fields = ['query','table_name']

    @apply_defaults
    def __init__(self, db_conn_id, query, table_name, **kwargs):
        super().__init__(**kwargs)
        self.db_conn_id = db_conn_id
        self.query = query
        self.table_name = table_name

    def execute(self, context):
        print(f"Executing query: {self.query} on table {self.table_name} using {self.db_conn_id}")
        #In a real system this would use the hook to connect to the database and perform the operation
        return {"query":self.query,"table":self.table_name,"status":"extracted"}


class DatabaseLoader(BaseOperator):

  template_fields = ['table_name', 'columns']
  @apply_defaults
  def __init__(self, db_conn_id, table_name, columns, **kwargs):
    super().__init__(**kwargs)
    self.db_conn_id = db_conn_id
    self.table_name = table_name
    self.columns = columns

  def execute(self, context):
    #in a real environment this would push the data to the table.
    print(f"Loading data into {self.table_name} with columns {self.columns} using {self.db_conn_id}")
    return {"table": self.table_name, "status":"loaded"}

class SqlProcessingPipeline:
  def __init__(self, db_conn_id):
    self.db_conn_id = db_conn_id

  def create_extraction_task(self, query, table_name, task_id):
    return DatabaseExtractor(task_id=task_id, db_conn_id = self.db_conn_id, query=query, table_name=table_name)

  def create_load_task(self, table_name, columns, task_id):
    return DatabaseLoader(task_id=task_id, db_conn_id=self.db_conn_id, table_name=table_name, columns=columns)


@dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None, tags=["example", "class-dag-operators"])
def sql_pipeline_dag():

    pipeline = SqlProcessingPipeline(db_conn_id="my_db_conn")
    extraction_task = pipeline.create_extraction_task(
        query = "SELECT * FROM users",
        table_name="users",
        task_id="extract_users"
    )

    load_task = pipeline.create_load_task(
        table_name="users_processed",
        columns=['id', 'name', 'email'],
        task_id="load_users"
    )
    extraction_task >> load_task

sql_pipeline_dag()

```

Here, the tasks are now customized operators extending `BaseOperator`, incorporating database connections and queries. Note the usage of `template_fields` that enables Jinja templating of your queries. The `SqlProcessingPipeline` class acts as a factory, allowing the creation of multiple extraction or load tasks, further improving reusability. The DAG orchestrates using `>>`, the traditional operator dependency setup. This showcases a hybrid approach combining classes with specific operators and makes it very easy to modify this to any different kind of hook/operator you have in place.

Finally, let's consider a scenario where you need to re-use some sub-tasks across different pipelines:

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.models import BaseOperator

class SubTaskLibrary:
  @task()
  def clean_data(self, data):
    print("Cleaning data...")
    return {"status":"cleaned", "data": data}

  @task()
  def validate_data(self, data):
    if data["status"] == "cleaned":
      print("Validating cleaned data...")
      return {"status":"validated", "data":data["data"]}
    else:
        raise ValueError("Invalid data status")


class PipelineManager:

  def __init__(self):
      self.library = SubTaskLibrary()

  def build_pipeline(self, pipeline_name, raw_data):
      @dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None, tags=["example", "class-dag"])
      def dynamic_pipeline():
          cleaned_data = self.library.clean_data(raw_data)
          validated_data = self.library.validate_data(cleaned_data)
          print(f"Pipeline {pipeline_name} final data {validated_data['data']}")
      return dynamic_pipeline

pipeline_manager = PipelineManager()
data1 = {"type":"json","value":"{}"}
pipeline1 = pipeline_manager.build_pipeline("json_pipeline",data1)
pipeline1()

data2 = {"type":"csv","value":"1,2,3"}
pipeline2 = pipeline_manager.build_pipeline("csv_pipeline",data2)
pipeline2()

```
Here, the `SubTaskLibrary` defines common data manipulation tasks, and the `PipelineManager` constructs DAGs with these shared tasks. This approach is particularly useful when you have pipelines that perform similar operations but on slightly different data or with different parameters. The DAGs are built dynamically, demonstrating how to create multiple different workflows with shared logic. This approach is ideal if you have a large number of workflows with similar requirements.

For further reading, I'd recommend "Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter, for a comprehensive understanding of Airflow, including DAG structure and design patterns. The official Apache Airflow documentation is also essential, particularly the sections on the Taskflow API and custom operators. I’d also suggest looking into "Designing Data-Intensive Applications" by Martin Kleppmann, which while not strictly about airflow, will help you understand best practices when constructing complex data pipelines.

In summary, encapsulating DAGs in classes using the Taskflow API is a significant leap toward building scalable and maintainable Airflow deployments. It improves code organization, encourages reuse, and makes your workflow pipelines far more robust. The examples above illustrate a few approaches to doing this. It’s an approach that has saved me a great deal of time in the past and I believe it will do the same for you.
