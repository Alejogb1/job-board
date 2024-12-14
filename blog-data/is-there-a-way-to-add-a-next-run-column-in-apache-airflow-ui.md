---
title: "Is there a way to add a "next run" column in apache airflow UI?"
date: "2024-12-14"
id: "is-there-a-way-to-add-a-next-run-column-in-apache-airflow-ui"
---

no problem, i've definitely been down this road before, and it's a common frustration with airflow's ui. the short answer is: airflow doesn't natively provide a "next run" column in its ui. that’s a bummer, i get it. it would be super handy. but, we can totally get a similar result with a little elbow grease and, more importantly, python.

let me explain why it isn’t a thing and how we usually work around it. airflow’s core model is about dag *runs*, not scheduled *future* runs. the ui focuses on the state of past and current executions. while the scheduler knows the upcoming run time, it doesn't actively push that data into the web ui view. the ui is built to display what has already happened or is happening. this isn't a flaw, but a design choice based on the model of airflow as a workflow orchestrator.

so, we have to roll our own. here's how i’ve tackled this in the past, and how i'd approach it now. the main idea is to fetch the next scheduled time from airflow and inject it into the dag list view somehow. we can't directly modify the ui's html through config, but we can use airflow’s api. that's where we do our heavy lifting.

here's a couple of ways you can achieve a “next run” like view:

**method 1: a custom dag attribute and plugin**

the cleanest way to do this is using a custom dag property and an airflow webserver plugin. with a custom attribute, we can add the "next_run_at" value as metadata to the dag. with a plugin we hook that to the ui.

first, we need to calculate the next run time in our dags. this involves grabbing the schedule interval and using that along with the dag's last run to extrapolate the next schedule. i've wrapped that logic inside a function to keep it tidy:

```python
from airflow.utils import timezone
from airflow.models import Dag, dag
from datetime import timedelta

def calculate_next_dagrun(dag: Dag):
    if dag.schedule_interval is None: #it's not a scheduled dag so no 'next_run'
      return None
    now = timezone.utcnow()
    if dag.get_last_dagrun(): #if no previous run next run will just be in the 'future' from now
      last_run_time = dag.get_last_dagrun().execution_date
      if not last_run_time:
        return None
      next_run_time = dag.following_schedule(last_run_time)
      if next_run_time < now: # the next schedule time is on the past
          next_run_time = dag.following_schedule(next_run_time) #calculate the next one again
    else:
      next_run_time = dag.following_schedule(now)

    return next_run_time
```

that function gives us the next logical run based on the last run and schedule defined for a given dag. after that, we need to add it as a custom property. i've done that in the next snippet. i make sure it calculates the value when the dag is parsed and not when the task is running.

```python
def create_next_run_dag(dag_id, schedule_interval, start_date,default_args, tasks):

    @dag(dag_id=dag_id, schedule=schedule_interval, start_date=start_date, catchup=False, default_args=default_args,  render_template_as_jinjja=True,  on_failure_callback = None)
    def next_run_dag_function():
          dag_obj = Dag(dag_id)
          next_run_value = calculate_next_dagrun(dag_obj)
          setattr(dag_obj,'next_run_at',next_run_value)
          tasks()
    return next_run_dag_function()


# example of usage
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_task():
    print("hello world")

default_args_example = {
    'owner': 'me',
    'retries': 1,
}

def create_tasks():
  PythonOperator(
    task_id='print_message',
    python_callable=my_task
    )

dag_example = create_next_run_dag(
    dag_id='example_dag',
    schedule_interval = timedelta(minutes=1),
    start_date=datetime(2023,1,1),
    default_args = default_args_example,
    tasks = create_tasks
)
```

now for the fun part – making the ui show this. we’ll use an airflow plugin to modify the dag list view. first, we install the necessary library with: `pip install flask-appbuilder`. next we create a simple class to extends the ui to inject the data. this class will modify the base view to include the new property that we set:

```python
from airflow.plugins_manager import AirflowPlugin
from airflow.www.app import appbuilder
from airflow.utils import timezone
from flask_appbuilder import expose, BaseView
from flask import Markup

class CustomDagView(BaseView):

    route_base = "/custom_dag"

    @expose("/dags/")
    def dags_view(self):
        dags = self.appbuilder.get_session.query(DagModel).all()
        dags_with_next_run = []
        for dag in dags:
            next_run_at = getattr(dag, 'next_run_at', None)
            next_run_at_str = next_run_at.isoformat() if next_run_at else "n/a"
            dags_with_next_run.append({
               "dag_id": dag.dag_id,
               "next_run_at": Markup(next_run_at_str)
            })
        return self.render_template("custom_dag/dags_list.html",
                                     dags=dags_with_next_run,
                                    )

class CustomUiPlugin(AirflowPlugin):
    name = "custom_dag_view"
    flask_blueprints = [
        {
            "name": "custom_dag_view",
            "blueprint":  CustomDagView().create_blueprint(appbuilder, __name__),
        }
    ]
```

after creating that class, you should also create the 'custom_dag/dags_list.html' file under your 'plugins' folder (the same where you created this python file) this html file should look like this:

```html
{% extends "appbuilder/base.html" %}

{% block title %}
    Custom Dag View
{% endblock %}

{% block content %}
   <div class="card">
            <h5 class="card-header">Dags List</h5>
          <div class="card-body">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Dag Id</th>
                            <th>Next run at</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for dag in dags %}
                            <tr>
                                <td>{{ dag.dag_id }}</td>
                                <td>{{ dag.next_run_at }}</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
    </div>
{% endblock %}
```
with this you've created a new view that lists all dags and their next run time.

**method 2: a custom endpoint**

another way is to expose an api endpoint and then you can display the data on some custom dashboards, this is very useful if you already have a tool to display data and query on a rest api.

```python
from airflow.api.auth.decorators import  action_logging
from airflow.api.common.experimental import check_and_get_dag
from airflow.api_connexion import security
from airflow.api_connexion.schemas.dag_schema import DAGCollection, DAGSchema, DAGEndpointsSchema
from airflow.utils import timezone
from airflow.utils.state import DagRunState
from airflow.models import Dag
from flask import request
from flask import Response
from typing import List, Optional
import json

@security.requires_access([("DAG", "READ")])
@action_logging
def get_dags_with_next_run():
    """Return a list of DAGs with their next run time."""
    dags = Dag.get_all_dags()
    now = timezone.utcnow()
    dag_list = []
    for dag in dags:
        if not dag.is_active:
          continue

        next_run = calculate_next_dagrun(dag)

        if next_run:
            next_run_str = next_run.isoformat()
        else:
            next_run_str = "n/a"
        dag_data = {"dag_id": dag.dag_id, "next_run_at": next_run_str}
        dag_list.append(dag_data)
    return Response(json.dumps(dag_list),mimetype='application/json')


#this adds the endpoint to airflow
def init_appbuilder_views(appbuilder):
  try:
      appbuilder.add_api(
            DAGEndpointsSchema,
            {"get_dags_next_run": get_dags_with_next_run},
        )
  except AttributeError as e:
        print(f"can't import api due to error: {e}")
        return False
  return True

```

then you can add this function to the `init_appbuilder` of the airflow configuration file. now you can call `/api/v1/dags_next_run` to get all dags with their next_run and do with that as you wish. you can integrate this to any custom dashboards.

**important notes and gotchas**

*   **timezone issues**: be mindful of timezones. airflow stores times in utc, and your system/ui might be in a different timezone. the above examples are timezone aware, but double check your local timezone config for ui elements. i once spent hours debugging a "next run" that was in the past only to realize the server timezone was different than mine. it was like watching a clock travel back in time - a painful experience that only technical users understand.
*   **performance**: if you have a huge number of dags, these queries can add some overhead. make sure to test performance with a large number of dags. caching can be a good solution to reduce the workload on your db.
*   **plugins:** the first approach uses a plugin and has some maintenance overhead as you need to create a custom view, if not you can use the second approach which is easier to maintain.

**further reading**

i would really suggest checking airflow documentation, there's a lot of useful knowledge there. i would also suggest reading “fluent python” by luciano ramalho, he talks a lot about python details that are very useful when creating this kind of solutions. the airflow documentation includes a section about the api: "apache airflow rest api documentation" i would suggest to read it carefully it's super useful when creating custom plugins or api endpoints.

let me know if you get stuck somewhere. i've probably tripped over a similar pitfall, and we can figure it out together.
