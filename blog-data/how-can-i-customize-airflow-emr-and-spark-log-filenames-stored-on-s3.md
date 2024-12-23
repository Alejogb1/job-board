---
title: "How can I customize Airflow, EMR, and Spark log filenames stored on S3?"
date: "2024-12-23"
id: "how-can-i-customize-airflow-emr-and-spark-log-filenames-stored-on-s3"
---

Alright, let's talk about customizing log filenames when orchestrating pipelines with Airflow, EMR, and Spark, specifically when those logs are destined for S3. I've dealt with this quite a bit, particularly in projects where maintaining a logical and searchable log structure became critical for debugging and monitoring. It’s not always a straightforward configuration, but with a proper understanding of how these tools interact, it's certainly achievable.

The core challenge arises from the default naming conventions each service employs. Airflow, by default, generates log filenames based on the task id, dag id, and execution date. EMR, when launching Spark applications, primarily uses the application id and container id. When everything’s piped to S3 without modifications, you often end up with a chaotic, deeply nested directory structure that is difficult to navigate. The trick is to intercept these default naming schemes and introduce our customized logic. This often means dipping into the configuration parameters of each tool and, at times, leveraging environment variables and scripting.

Firstly, let's tackle the Airflow component. Airflow's logging system is quite flexible. We're mostly concerned with the task log filename patterns. You can configure how task logs are named by adjusting the `logging_config` parameter in your `airflow.cfg` file (or your respective configuration management system for Airflow). I’d highly recommend using a proper configuration management approach over editing the `airflow.cfg` directly in a production environment. The key here is the `log_filename_template` key. The default might look something like: `{{ ti.dag_id }}/{{ ti.task_id }}/{{ ts }}/{{ try_number }}.log`. I recall in one project, the initial approach yielded an almost unusable S3 bucket due to the timestamp component creating overly granular directories.

Here's a Python example demonstrating a modified configuration you could use, ideally within your Airflow configuration setup, not directly in the `airflow.cfg`:

```python
airflow_config_override = {
    'logging': {
        'logging_level': 'INFO',
        'log_format': '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        'log_filename_template': '{{ dag_id }}/{{ task_id }}/{{ execution_date.strftime("%Y-%m-%d") }}/{{ try_number }}.log'
        }
    }

```

This snippet demonstrates how to modify the log filename pattern to use just the date part of the execution date, which simplifies browsing. The `execution_date` object, accessed through jinja templating `{{ execution_date.strftime("%Y-%m-%d") }}`, offers the flexibility for formatting. You can further customize the structure as you see fit. For instance, adding a user-defined tag to a dag and then use it in the filename. This is how you manipulate Airflow's logging.

Next, let's talk about EMR's logging, specifically for Spark applications. EMR uses S3 buckets to store Spark logs. The default structure within this bucket is fairly deep; typically, it follows a pattern related to the EMR cluster id, the application id, and container id. While these identifiers are helpful for detailed debugging at the cluster level, they often complicate log analysis at the pipeline level. When working with Spark on EMR, you don't have direct control over the Spark application log filenames; however, you can influence where EMR places these logs by configuring the EMR cluster’s settings and S3 logging configurations. You also can customize logging on Spark application level by modifying `spark-defaults.conf`, and this approach is my preferred way to achieve customization.

Here's a Python example showing how you could configure your Spark session’s properties using `spark-defaults.conf` within your EMR cluster to organize the log file’s destination folder:

```python
spark_config_template = """
spark.history.fs.logDirectory    s3://your-bucket-name/emr-logs/spark-history/
spark.eventLog.dir              s3://your-bucket-name/emr-logs/spark-eventlogs/
spark.eventLog.enabled           true
"""
def configure_spark_logging(emr_cluster_id):
    """Configures Spark logging on EMR cluster."""
    # In real world this config should be managed in configuration files
    # or from a dedicated configuration service

    core_site_properties = {
        "Classification": "core-site",
        "Properties": {
            "fs.s3a.server-side-encryption-algorithm": "AES256", # example of adding encryption
            "fs.s3a.connection.maximum": "50",
            "fs.s3a.connection.timeout": "120000"
            }
    }
    spark_defaults_properties = {
        "Classification": "spark-defaults",
        "Properties": {
            "spark.history.fs.logDirectory" : "s3://your-bucket-name/emr-logs/spark-history/",
            "spark.eventLog.dir" : "s3://your-bucket-name/emr-logs/spark-eventlogs/",
            "spark.eventLog.enabled" : "true"
            }
        }

    # Example of using a boto3 client to modify EMR cluster configuration
    client = boto3.client('emr')
    client.modify_cluster(
        ClusterId=emr_cluster_id,
        Configurations=[core_site_properties, spark_defaults_properties]
    )

```

In this python example, I am adding two classification configuration parameters to the cluster `spark-defaults` and `core-site`. Core-site properties are used to fine-tune s3 access and avoid timeouts that might occur for slow connections. In the `spark-defaults`, I'm setting `spark.history.fs.logDirectory` and `spark.eventLog.dir` properties. I'm essentially configuring the output location to an easy-to-find location on the S3. While this won't directly modify the filename itself, it drastically improves the manageability of the logs. Remember, the goal is not just renaming the files but to have them organized in a way that makes sense for your analysis needs. This approach is superior to modifying the spark configuration at application submission.

Finally, and this is often overlooked, you may need to access the underlying Spark logs created by the driver and executor applications, these often have cryptic filenames. However, with a little customization, this can be improved. The output generated by EMR on the cluster instance is also collected by the EMR agent, and that also includes the Spark application logs. The naming for the logs in S3 in this case includes the container and executor ids. You will not be able to modify these names at the Spark level. However, you can intercept the log files in the EMR cluster before the emr agent copies them to S3, and rename them as required.

Here is an example of how you could modify the log name by intercepting the log output in the EMR instance and using `sed` and a custom bash script to create more meaningful file names:

```bash
#!/bin/bash

LOG_BASE_DIR="/mnt/var/log/hadoop-yarn/containers/"
S3_LOG_DESTINATION="s3://your-bucket-name/emr-custom-logs/"
JOB_NAME="my-spark-job" # Customize the Job name in the script. This could be set from an Airflow env variable

find $LOG_BASE_DIR -type d -maxdepth 4 -print0 | while IFS= read -r -d $'\0' dir; do

    if [[ "$dir" =~ "application_" ]]; then
        APP_ID=$(echo "$dir" | grep -oP 'application_[0-9]+_[0-9]+')
        for log_file in "$dir"/stdout; do #stdout is just an example, this can be all files or more specific ones like stderr
            if [[ -f "$log_file" ]]; then
              # Extract the container ID. This will be specific to EMR's log folder structure
              CONTAINER_ID=$(echo "$dir" | grep -oP 'container_[0-9]+_[0-9]+_[0-9]+_[0-9]+')
              # Construct the new filename format
              NEW_LOG_FILENAME="$JOB_NAME-$APP_ID-$CONTAINER_ID.log"

             aws s3 cp "$log_file" "$S3_LOG_DESTINATION/$APP_ID/$NEW_LOG_FILENAME"
            fi
        done

        for log_file in "$dir"/stderr; do #stderr log files.
           if [[ -f "$log_file" ]]; then
               CONTAINER_ID=$(echo "$dir" | grep -oP 'container_[0-9]+_[0-9]+_[0-9]+_[0-9]+')
               NEW_LOG_FILENAME="$JOB_NAME-$APP_ID-$CONTAINER_ID.err.log"

              aws s3 cp "$log_file" "$S3_LOG_DESTINATION/$APP_ID/$NEW_LOG_FILENAME"
            fi
        done
    fi
done
```

This bash script identifies relevant Spark logs under the yarn log folder. It extracts the application and container ids, and then constructs new log filenames and copies the logs to a specific S3 location. This script can be configured to run during EMR cluster startup or as a step after the Spark job completes. The script requires careful implementation, especially to avoid performance issues. Additionally, it’s essential to ensure that you have proper permissions for S3 access.

In summary, customizing log filenames in this environment involves multiple layers of configuration: Airflow’s templating for task logs, Spark’s configuration via `spark-defaults.conf` and EMR configurations, and lastly, some custom scripting within the cluster. It's a bit of a puzzle, but a structured approach greatly improves the usability of logs in S3.

For deeper understanding, I’d suggest reviewing: “Spark: The Definitive Guide” by Bill Chambers and Matei Zaharia for the Spark specifics, and the official Airflow documentation along with the AWS EMR documentation. Those are the authoritative sources you need for these kind of tasks. Don't underestimate the importance of practical experience and experimentation with these configurations. You’ll likely encounter edge cases, but with the knowledge of the tools, you can adapt your log configurations to fit your precise requirements.
