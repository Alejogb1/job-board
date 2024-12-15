---
title: "How to Explicitly set a dependency using the TaskFlowAPI?"
date: "2024-12-15"
id: "how-to-explicitly-set-a-dependency-using-the-taskflowapi"
---

alright, so you're looking at explicitly setting a dependency in airflow's taskflow api, yeah? i've been down that road more times than i care to count, and it can be a bit tricky initially. especially when you move from the old way of defining dependencies with bitshift operators, it feels like a whole new ball game.

my early days with airflow, oh man, that was a learning curve. i remember this one particular pipeline i was building. it involved scraping data, cleaning it, and then shoving it into a database. i initially tried to make it all one huge dag, one huge task. that was a mistake, a huge mistake. it was like trying to fit a square peg in a round hole and when i finally realized the error i was making it was too late i had to refactor all from scratch. debugging that mess was a nightmare because things would randomly fail without any indication of why and i had to use print statements like a debugging dinosaur to figure out all the dependencies i had mixed up. that's when i learned the power of modularity and how important it is to understand the taskflow api and not just blindly copy paste things from the web.

so, to directly answer your question, the taskflow api basically lets you define tasks as python functions, and dependencies are created by how you call those functions within your dag. there is no special explicit function to set it. for example, if you call `task_b(task_a())` then task_b depends on task_a and that is what we want. it's all about the function execution order. when taskflow api comes into play you don't use anymore `task_a >> task_b` or whatever, instead you call task_a inside task_b function and airflow does the job for you. this is the basis of how explicit dependency setting works in taskflow api, the function call.

let's look at some example code. first, a simple case. two tasks, one depending on the other. imagine you have a function that downloads some data, and another that processes it.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), schedule=None, catchup=False)
def my_simple_dependency_dag():

    @task
    def download_data():
        # imagine this downloads data from somewhere
        print("data downloaded")
        return "data_downloaded_successfully"

    @task
    def process_data(download_result):
      print(f"processing data, got {download_result}")
      return "data_processed_successfully"

    process_data(download_data())

my_dag = my_simple_dependency_dag()
```

in this example, `process_data` will always run after `download_data`. the output of `download_data` is passed into `process_data` which automatically creates the dependency, taskflow api magic right there. airflow sees this structure and schedules the tasks accordingly. we use this in many of our pipelines. it is extremely handy. i mean imagine how horrible this dag would be with the bitwise operators, having a 100 tasks dag would be unreadable but using the taskflow api it feels like coding, not like creating a dependency tree with arrows.

now, let's go a bit further. suppose you want multiple tasks to depend on a single task. imagine preprocessing the data into different formats, let's say json and xml, each format needs to be processed after the main preprocessing function is done.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), schedule=None, catchup=False)
def my_fan_out_dag():

    @task
    def preprocess_data():
        print("data preprocessed")
        return "preprocessed_data"

    @task
    def process_json(preprocessed_result):
        print(f"processing to json, got {preprocessed_result}")
        return "json_processed"

    @task
    def process_xml(preprocessed_result):
      print(f"processing to xml, got {preprocessed_result}")
      return "xml_processed"

    preprocessed_data = preprocess_data()
    process_json(preprocessed_data)
    process_xml(preprocessed_data)

my_fan_out_dag_instance = my_fan_out_dag()
```

here, both `process_json` and `process_xml` depend on the output of `preprocess_data`, this allows parallelism. they will be executed only after `preprocess_data` is completed, but airflow can execute them concurrently if there are resources available. the taskflow api handles this like a charm. it really simplifies complicated workflows. if you are used to the bitwise operators this seems strange at first but after a while you will love it. i have a personal rule when i have to do airflow, taskflow is always my go to choice, unless there is a very good reason for not doing it.

and finally, what if you need to pass the output of multiple tasks into one? imagine you are reading data from different sources and they have to be joined into a single one to proceed further into the pipeline.

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), schedule=None, catchup=False)
def my_fan_in_dag():

    @task
    def read_source_a():
        print("reading data from source a")
        return "data_from_a"

    @task
    def read_source_b():
        print("reading data from source b")
        return "data_from_b"


    @task
    def combine_sources(source_a_data, source_b_data):
      print(f"combining data, got {source_a_data} and {source_b_data}")
      return "combined_data"

    combined_data = combine_sources(read_source_a(), read_source_b())

my_fan_in_dag_instance = my_fan_in_dag()
```

in this case, `combine_sources` will be executed only after `read_source_a` and `read_source_b` are both complete. it takes both of their results as inputs. think of it like a good dinner, you need all the ingredients before you can prepare it. my wife hates cooking and i don't really like to code. i mean, who does like to code? i only do it because i am not talented in anything else. i don't really get how people love to write code i don't think i will ever understand it. i think that is more interesting than understanding the taskflow api if you ask me.

the beauty of this approach is it's very explicit. the dependency structure is right there, encoded in the function calls. there is no hidden dependency created by the bitwise operators. this greatly improves readability and maintainability, especially when working on complex pipelines with multiple people.

regarding resources, i’d suggest reading the official airflow documentation, it has come a long way and it is pretty good now. it goes deeply into detail of how the taskflow api works. i also found "data pipelines with apache airflow" by bassett, although a little bit outdated is still a good reference. for advanced topics, maybe read some of the airflow enhancement proposals (aips) related to taskflow api if you want to understand why things are implemented the way they are. the github repo of airflow itself is a gold mine if you have the time and the energy to read all the issues and discussions there. i have spent countless hours there and i can tell you is the best source to learn from and to see where the project is heading.

remember that airflow is constantly evolving, so keeping up to date with the latest version changes is very important. i have seen too many pipelines that stopped working due to a library update. pay attention to your airflow versioning and all your dependencies. good luck with your pipelines. i think that is all for now. feel free to ask further questions if you need anything else.
