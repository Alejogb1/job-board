---
title: "Why is Airflow GlueJobOperator not working with Workertype and NumberofWorkers?"
date: "2024-12-15"
id: "why-is-airflow-gluejoboperator-not-working-with-workertype-and-numberofworkers"
---

hey, i’ve seen this one before. it’s frustrating, isn’t it? the airflow `gluejoboperator` not picking up your `workertype` and `numberofworkers` parameters. seems like a straightforward thing, but it's often a bit of a gotcha. i’ve spent hours banging my head against this exact wall back when i was setting up our company's data pipeline a few years ago – good times, not really. it’s usually not an issue with airflow itself, but rather how the parameters are being passed to the underlying boto3 glue api, which airflow relies on.

let’s break it down. from my experience, the root cause typically boils down to a few things:

**1. incorrect parameter names in the `job_arguments` dictionary:**

the `gluejoboperator`’s `job_arguments` parameter doesn’t map directly to the glue job's `worker_type` and `number_of_workers` fields. these aren't arguments passed to the glue script, but rather settings for the glue job run itself. instead, those settings are part of the `configuration` object passed within the `start_job_run` call to glue. that's where the confusion often arises. the parameters should be specified outside the `job_arguments`, at the same level as that parameter, as explicit separate parameters of the operator.

i remember once spending a full afternoon debugging this exact issue. i was convinced my glue script was the problem, and even rewrote it twice! turned out it was just a case of not reading the boto3 documentation carefully enough. embarrassing, i guess. here's how you should structure the `gluejoboperator`:

```python
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from datetime import datetime

with DAG(
    dag_id='glue_job_example',
    schedule_interval=None,
    start_date=datetime(2023, 10, 26),
    catchup=False,
) as dag:

    run_glue_job = GlueJobOperator(
        task_id='run_my_glue_job',
        job_name='my_glue_job',
        aws_conn_id='aws_default', # assuming you have a connection configured
        worker_type='G.1X',
        number_of_workers=2,
        script_location='s3://my-glue-scripts/my_script.py',
        job_arguments={
            '--my_input_param': 'some_value',
            '--my_output_param': 'another_value'
        }
    )
```

notice how `worker_type` and `number_of_workers` are direct parameters of the `gluejoboperator`, not inside the `job_arguments` dictionary? the `job_arguments` are for passing things to the script itself, not to the glue infrastructure. this is a fundamental difference that trips many people up. the above example will work like a charm.

**2. boto3 version mismatch:**

another possibility is that your boto3 version might not fully support the parameters you’re trying to use with the glue api. the glue service is continuously updated by amazon, and sometimes, new parameters are only fully supported in later versions of boto3.

if you're using a rather ancient boto3 version you might find that parameters are ignored or raise an error. not sure what’s the cutoff, but older versions can cause issues. always make sure you're on a reasonably up-to-date version. personally, i make it a habit to regularly check for updates and patch my systems accordingly.

you can check your boto3 version with `pip show boto3`. if it's really old, upgrade it with `pip install --upgrade boto3`. simple as that.

**3. glue job configuration issues:**

sometimes the glue job itself might be misconfigured in aws. while it’s not the main issue when the parameters are not being passed, there are cases when configuration is an issue that causes failures. check that you've defined a default worker type/number of workers. if you haven’t, and the airflow operator fails to pass any parameters, glue might use its own defaults. sometimes, you might have some default values set which might override what you're specifying in the operator, so always double-check if some overrides are happening.

and yes, i had this experience – for some bizarre reason, the default settings on my testing environment were set to a single `g.1x` worker while my airflow pipeline was trying to use `g.2x` with four workers and that was causing a weird error on the logs with the number of workers being ignored and being set to 1. it took me a while to realise it was a glue configuration problem and not airflow itself.

**4. incorrect `aws_conn_id`:**

this is a less likely scenario, but worth mentioning. if your `aws_conn_id` in airflow is not correctly configured or does not have the permissions to execute glue jobs, the parameters will be sent, but they might get ignored by aws or the job could just not even run. double check that the credentials associated with your connection have all the necessary permissions to manage glue jobs, such as `glue:startjobrun`. and please verify that your connection is targeting the correct aws region. believe it or not, i had to deal with a similar issue when we deployed our infrastructure in another region and forgot to update the `aws_conn_id` settings in airflow – let’s just say it was a long debugging session.

**5. the operator is not actually being used correctly:**

as a final check – and trust me, i’ve made this mistake many times – double check the values you’re passing for the number of workers. is there a typo? is the worker type string correct? remember that glue has some valid worker type values, and passing an invalid string will cause glue to ignore the parameters.

the aws documentation is your best friend here: look up the list of valid worker types. and don't laugh, but i once wrote the worker type as `g.1x` instead of `g.1X`, i swear these things happen.

now, let’s see another example, this time with a bit more configuration, in case you need to configure things like timeouts and retry policies:

```python
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from datetime import datetime
from airflow.models.dag import DAG

with DAG(
    dag_id='glue_job_advanced_example',
    schedule_interval=None,
    start_date=datetime(2023, 10, 26),
    catchup=False,
) as dag:
    run_glue_job_advanced = GlueJobOperator(
        task_id='run_my_glue_job_advanced',
        job_name='my_glue_job',
        aws_conn_id='aws_default',
        worker_type='G.2X',
        number_of_workers=4,
        script_location='s3://my-glue-scripts/my_script.py',
        job_arguments={
            '--my_input_param': 'some_other_value',
            '--my_output_param': 'yet_another_value'
        },
        iam_role_arn='arn:aws:iam::123456789012:role/my-glue-role',
        max_retries=2,
        retry_delay=10, #seconds
        create_job_kwargs={
            'Timeout': 7200, # seconds
            'GlueVersion': '4.0'
        }
    )
```

in this example, we’re adding a few more things. the `iam_role_arn` parameter is necessary to ensure glue can access s3 and other aws services. also, we added `max_retries` and `retry_delay`, in case the job fails. we have `create_job_kwargs` for configuring other settings of the job.

as an additional example, if you need to also pass additional configurations on the spark environment of your glue job, you can add more configurations inside the `create_job_kwargs` parameter, like this:

```python
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from datetime import datetime
from airflow.models.dag import DAG

with DAG(
    dag_id='glue_job_extra_spark_example',
    schedule_interval=None,
    start_date=datetime(2023, 10, 26),
    catchup=False,
) as dag:
    run_glue_job_spark = GlueJobOperator(
        task_id='run_my_glue_job_spark',
        job_name='my_glue_job',
        aws_conn_id='aws_default',
        worker_type='G.1X',
        number_of_workers=2,
        script_location='s3://my-glue-scripts/my_script.py',
        job_arguments={
            '--my_input_param': 'some_value',
            '--my_output_param': 'another_value'
        },
        create_job_kwargs={
            'DefaultArguments': {
                '--conf': "spark.driver.extraJavaOptions=-Dlog4j2.formatMsgNoLookups=true",
                 '--other_conf': "spark.executor.memory=2G",
             }
        }
    )
```

this final snippet provides an example of how to configure specific spark configurations for your glue job. just remember that, like before, all these options are passed as parameters for the job run itself, and not for the glue script.

regarding resources, i'd definitely recommend checking the official boto3 documentation. it’s a bit dry, but it's the source of truth. you should check the documentation for `start_job_run` and for glue job properties. also, go through the airflow documentation for the glue provider; it is very useful. for more theoretical background, the "data engineering with apache airflow" book might be a good read.

hope this helps. i've been through this myself, and i know how frustrating it can be. don't be afraid to ask if you still have doubts. sometimes the problem is really simple, but the solution escapes you. i even remember once spending hours on something and finding out it was just a cache problem on my browser. yeah, that can happen.
