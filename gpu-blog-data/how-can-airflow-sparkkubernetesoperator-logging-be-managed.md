---
title: "How can Airflow SparkKubernetesOperator logging be managed?"
date: "2025-01-30"
id: "how-can-airflow-sparkkubernetesoperator-logging-be-managed"
---
Within the context of large-scale data processing, effective logging for Apache Airflow's SparkKubernetesOperator is paramount for debugging, performance analysis, and maintaining overall operational visibility. My experience managing hundreds of daily ETL pipelines involving Spark on Kubernetes has shown that the default logging configuration often falls short, requiring a more tailored approach. The operator by itself provides limited stdout/stderr capture; therefore, a holistic strategy must consider both the Airflow task logs and the Spark application logs within the Kubernetes cluster.

The primary challenge with the SparkKubernetesOperator logging stems from its abstraction layer. Airflow launches a Spark application as a Kubernetes pod, meaning the logs aren't directly accessible through standard Airflow logging mechanisms. Airflow captures the Kubernetes pod's stdout and stderr streams, providing initial insight into the Spark driver's launch process. However, this omits crucial details of the Spark job itself, including executor activity, transformations, and specific error conditions. To fully understand a Spark application's behavior, I've had to incorporate several techniques.

My go-to strategy involves a three-pronged approach. First, I ensure that the Spark application generates its logs in a structured and persistent manner. Second, I leverage Kubernetes sidecar containers to stream these logs to a centralized logging system. Lastly, I implement monitoring dashboards to track the overall application health and log patterns. These approaches move beyond the Airflow task logs, which only provide high-level launch information.

To illustrate, consider a basic Spark job executed through the `SparkKubernetesOperator`. Without custom logging configuration, the Airflow task logs will contain only the Kubernetes pod initialization and final status, with minimal information about the Spark application itself.

```python
from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import SparkKubernetesOperator
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id='example_spark_no_logging',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    spark_app = SparkKubernetesOperator(
        task_id='spark_pi',
        application_file='local://example_spark_app.yaml',
        namespace='default',
        do_xcom_push=True,
    )
```

The above code defines a simple Airflow DAG using the `SparkKubernetesOperator`. The `application_file` points to a Spark application configuration in YAML, a standard Spark configuration pattern. The important point is that this, without additional setup, will generate limited logs in the Airflow task context; mainly showing Kubernetes resources being created or destroyed. To capture more meaningful data, the Spark application configuration itself must be modified to route logs correctly.

The first significant improvement involves modifying the Spark configuration within `example_spark_app.yaml` to direct logs to a persistent storage system. I frequently use S3, but other options like GCS or Azure Blob Storage are equally suitable. By incorporating Spark's logging configuration options, we gain the ability to collect logs directly from the executors.

```yaml
apiVersion: "sparkoperator.k8s.io/v1beta2"
kind: SparkApplication
metadata:
  name: spark-pi
  namespace: default
spec:
  type: Scala
  mode: cluster
  image: "gcr.io/spark-operator/spark:v3.3.0"
  imagePullPolicy: Always
  mainClass: org.apache.spark.examples.SparkPi
  mainApplicationFile: "local:///opt/spark/examples/jars/spark-examples_2.12-3.3.0.jar"
  sparkVersion: "3.3.0"
  restartPolicy:
    type: Never
  hadoopConf:
    "fs.s3a.access.key": "{{ var.value.s3_access_key }}"
    "fs.s3a.secret.key": "{{ var.value.s3_secret_key }}"
    "fs.s3a.endpoint": "s3.amazonaws.com"
  driver:
    cores: 1
    memory: "1g"
    labels:
      version: 3.3.0
    env:
      - name: SPARK_LOG_DIR
        value: "s3a://my-bucket/spark-logs/{{ ds }}/{{ ts_nodash }}/driver"
      - name: SPARK_LOG_PROPERTIES
        value: "log4j.properties"
    configMaps:
      - name: spark-log-config
        mountPath: /opt/spark/conf/
  executor:
    cores: 1
    instances: 1
    memory: "1g"
    labels:
      version: 3.3.0
    env:
      - name: SPARK_LOG_DIR
        value: "s3a://my-bucket/spark-logs/{{ ds }}/{{ ts_nodash }}/executor"
      - name: SPARK_LOG_PROPERTIES
        value: "log4j.properties"
    configMaps:
      - name: spark-log-config
        mountPath: /opt/spark/conf/
```

Here, I've defined `hadoopConf` to access S3 storage, configured the `SPARK_LOG_DIR` environment variable to write driver and executor logs to S3 with a date-partitioned structure, and specified that a `configMap` containing the `log4j.properties` file is needed. The `SPARK_LOG_PROPERTIES` variable enables custom Log4j configuration. This configuration instructs Spark to write its logs directly to persistent storage, making them available for analysis regardless of the Kubernetes pod's lifecycle. The path in `SPARK_LOG_DIR` uses macros to create a unique folder for each run based on `ds` and `ts_nodash` from Airflow's context.

The crucial companion configuration, `log4j.properties`, is mounted into the Spark driver and executor containers from a ConfigMap called `spark-log-config`. Below is an example.

```properties
log4j.rootCategory=INFO, file
log4j.appender.file=org.apache.log4j.RollingFileAppender
log4j.appender.file.File=${SPARK_LOG_DIR}/spark.log
log4j.appender.file.MaxFileSize=100MB
log4j.appender.file.MaxBackupIndex=10
log4j.appender.file.layout=org.apache.log4j.PatternLayout
log4j.appender.file.layout.ConversionPattern=%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n
log4j.logger.org.apache.spark=INFO
```

This basic Log4j configuration defines the root logger as INFO level, designates the `file` appender as a `RollingFileAppender` to avoid overly large logs, and specifies the file output path using the `SPARK_LOG_DIR` environment variable we set earlier. The `ConversionPattern` defines the logging format. By combining the Spark configuration within the application specification with a suitable `log4j.properties` file, a more comprehensive logging solution is formed. This setup also ensures the separation of logs from the ephemeral Kubernetes pods, allowing access even after the pods have terminated.

This provides a good base for persistent storage of Spark logs. However, the logs do not go directly to our centralized log system, which requires an additional step. To achieve this, the implementation of a sidecar container which reads and forwards logs using fluentd or similar tools becomes vital. This can be added to the `SparkApplication` specification.

```yaml
  driver:
    #... previous config ...
    sidecars:
      - name: log-forwarder
        image: fluent/fluentd:v1.16.1-debian-1
        volumeMounts:
        - name: spark-logs
          mountPath: /spark-logs
        env:
        - name: FLUENT_LOG_ENDPOINT
          value: "{{ var.value.central_log_endpoint }}"
  executor:
    #... previous config ...
    sidecars:
      - name: log-forwarder
        image: fluent/fluentd:v1.16.1-debian-1
        volumeMounts:
          - name: spark-logs
            mountPath: /spark-logs
        env:
        - name: FLUENT_LOG_ENDPOINT
          value: "{{ var.value.central_log_endpoint }}"
  volumes:
    - name: spark-logs
      emptyDir: {}
```

In this setup, a sidecar container running Fluentd is added to both the driver and executor pods. The `emptyDir` volume acts as a common location for both the Spark application to write logs and the fluentd sidecar to read them.  The `FLUENT_LOG_ENDPOINT` environment variable can be an external service for collecting logs, which can be managed through Airflow variables. The Fluentd configuration within the sidecar will need to be set up to parse the log files and forward them to the desired endpoint. This approach allows for real-time log streaming to the centralized system.

To ensure that the S3 log output is not lost, a second volume mount to the `/spark-logs` directory would be necessary. This allows both the s3 driver and the sidecar to access log files. The Spark configuration must be modified to ensure the log files are written to this shared location rather than directly to S3. Then, the fluentd configuration can be adjusted to read these logs and push them to the destination, and an additional agent will handle the upload to S3 in the background. This setup allows both S3 storage for long term archival and a centralized location for real time monitoring.

Further enhancements I often implement include using Kubernetes' Resource Quotas to prevent uncontrolled resource usage, especially when running multiple Spark applications concurrently. Moreover, I always perform extensive testing of the logging pipeline, starting with isolated environments, ensuring robustness and stability prior to full deployment. I’ve found that without these measures, log collection can be incomplete, impacting the efficiency of troubleshooting and root cause analysis.

For a more robust logging and monitoring approach, I recommend exploring resources focused on effective Kubernetes logging patterns. Understanding the nuances of log aggregation tools, as well as techniques for creating monitoring dashboards and using metrics, is crucial. The official documentation for Apache Spark and the Kubernetes documentation provide detailed information regarding configuration options. Additionally, books and online courses focusing on cloud-native application development practices provide valuable insights into overall system monitoring strategies. These resources, when combined with practical implementation, are key to creating a manageable and observable Spark workload.
