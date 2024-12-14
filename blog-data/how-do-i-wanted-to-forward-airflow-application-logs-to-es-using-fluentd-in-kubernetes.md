---
title: "How do I wanted to forward Airflow application logs to ES using fluentd in Kubernetes?"
date: "2024-12-14"
id: "how-do-i-wanted-to-forward-airflow-application-logs-to-es-using-fluentd-in-kubernetes"
---

alright, so you're looking to pipe your airflow logs from kubernetes over to elasticsearch using fluentd, yeah? i've been down that road a few times, and it's definitely got some nuances. let me walk you through what i've learned, and maybe you can avoid some of the pitfalls i stepped into.

first off, the basic idea is pretty straightforward. airflow, running inside kubernetes pods, spits out logs. fluentd, also running in kubernetes (likely as a daemonset), picks these up and sends them to elasticsearch. the trick, as usual, is in the details of configuration and getting all the pieces to play nicely together.

for me, the first time i tackled this was back in 2019, i was working on a data pipeline for a fintech startup and we decided to move our airflow from a monolithic ec2 instance to kubernetes, and the logging suddenly became very complicated. i was used to just `tail -f`ing the airflow logs and had a big surprise moving to k8s. my first thought was using a simple sidecar container with something like filebeat that i was more comfortable with, that failed, spectacularly, to scale for several dags running in parallel. that was a learning experience, and i understood i needed something more robust and kubernetes native like fluentd. i remember i even tried using a sidecar pattern with fluentd itself, which quickly became a maintenance nightmare (don't even think about it!).

let's talk about fluentd. you'll want to set it up as a daemonset. this ensures that a fluentd pod runs on each of your kubernetes nodes, so it can pick up logs from all of your airflow pods regardless where the kubernetes scheduler decides to place them. you’ll probably want to use a `configmap` to handle fluentd's configuration, that's usually the most flexible approach. here's a snippet of what your fluentd config might look like. note the filter and output plugins:

```
<source>
  @type tail
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.pos
  tag kubernetes.*
  read_from_head true
  <parse>
    @type json
    time_key time
    time_type string
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<filter kubernetes.**>
  @type kubernetes_metadata
  kubernetes_url https://kubernetes.default.svc:443
  verify_ssl true
  cache_size 1000
</filter>

<filter kubernetes.**>
  @type record_transformer
  enable_ruby true
  <record>
    namespace ${record["kubernetes"]["namespace_name"]}
    pod ${record["kubernetes"]["pod_name"]}
    container ${record["kubernetes"]["container_name"]}
    log ${record["log"]}
    time ${record["time"]}
    level ${record["log"].match(/(?i)(DEBUG|INFO|WARNING|ERROR|CRITICAL)/) ? $& : "unknown"}
    task_id ${record["log"].match(/dag_id=(\w+), task_id=(\w+)/) ? $2 : 'unknown'}
    dag_id ${record["log"].match(/dag_id=(\w+), task_id=(\w+)/) ? $1 : 'unknown'}
  </record>
</filter>

<match kubernetes.**>
  @type elasticsearch
  host "#{ENV['FLUENT_ELASTICSEARCH_HOST']}"
  port "#{ENV['FLUENT_ELASTICSEARCH_PORT']}"
  logstash_format true
  logstash_prefix airflow-logs
  index_name  airflow-logs
  scheme https
  user "#{ENV['FLUENT_ELASTICSEARCH_USER']}"
  password "#{ENV['FLUENT_ELASTICSEARCH_PASSWORD']}"
  ssl_verify false
  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes
    flush_interval 10s
  </buffer>
</match>

```

this config uses the `tail` input plugin to read container logs from `/var/log/containers/*.log`. the `kubernetes_metadata` filter enriches each log line with metadata from the kubernetes api, like pod name, namespace, etc. the `record_transformer` filter extracts useful bits such as the log level and task id from airflow logs. you'll need to adjust the regex to match your log format. this is important, if the log format changes, you'll have some headaches until you correct the regex. i spent a whole weekend troubleshooting a log format change i didn't notice before. finally, the `elasticsearch` output plugin sends the processed logs to elasticsearch. make sure to set the correct environment variables (`FLUENT_ELASTICSEARCH_HOST`, `FLUENT_ELASTICSEARCH_PORT`, etc.). use secrets instead of env vars for production setups, i've seen people making that mistake and having credentials exposed in the pods yaml, never do that.

now, getting this into a kubernetes `configmap` is pretty straightforward, but i've seen people miss small things and then spend hours trying to find what's wrong. here's a basic `configmap` example:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluentd.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_key time
        time_type string
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
      kubernetes_url https://kubernetes.default.svc:443
      verify_ssl true
      cache_size 1000
    </filter>

    <filter kubernetes.**>
      @type record_transformer
      enable_ruby true
      <record>
        namespace ${record["kubernetes"]["namespace_name"]}
        pod ${record["kubernetes"]["pod_name"]}
        container ${record["kubernetes"]["container_name"]}
        log ${record["log"]}
        time ${record["time"]}
        level ${record["log"].match(/(?i)(DEBUG|INFO|WARNING|ERROR|CRITICAL)/) ? $& : "unknown"}
        task_id ${record["log"].match(/dag_id=(\w+), task_id=(\w+)/) ? $2 : 'unknown'}
        dag_id ${record["log"].match(/dag_id=(\w+), task_id=(\w+)/) ? $1 : 'unknown'}
      </record>
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host "#{ENV['FLUENT_ELASTICSEARCH_HOST']}"
      port "#{ENV['FLUENT_ELASTICSEARCH_PORT']}"
      logstash_format true
      logstash_prefix airflow-logs
      index_name  airflow-logs
      scheme https
      user "#{ENV['FLUENT_ELASTICSEARCH_USER']}"
      password "#{ENV['FLUENT_ELASTICSEARCH_PASSWORD']}"
      ssl_verify false
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes
        flush_interval 10s
      </buffer>
    </match>
```

you'll then mount this configmap into your fluentd daemonset, typically at `/fluentd/etc/fluentd.conf`.  make sure to create the `/var/log/fluentd-buffers` directory too, and give write permissions to the fluentd user. not doing that was another learning moment, a weekend was spent figuring out why logs were not getting to elasticsearch, and that was the only issue.

now for the daemonset. this is where you tell kubernetes to run a fluentd pod on each node. this is a very basic example, and you'll probably need to adjust the `resources` section and other settings to match your environment, and the role used to pull the images, i use a custom one so i will remove this part of the example.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  labels:
    app: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch7
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          valueFrom:
            secretKeyRef:
              name: elasticsearch-secret
              key: elasticsearch_host
        - name: FLUENT_ELASTICSEARCH_PORT
          valueFrom:
            secretKeyRef:
              name: elasticsearch-secret
              key: elasticsearch_port
        - name: FLUENT_ELASTICSEARCH_USER
          valueFrom:
            secretKeyRef:
              name: elasticsearch-secret
              key: elasticsearch_user
        - name: FLUENT_ELASTICSEARCH_PASSWORD
          valueFrom:
            secretKeyRef:
              name: elasticsearch-secret
              key: elasticsearch_password
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc/
          readOnly: true
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config
```

key points: the `/var/log` volume mount allows fluentd to access the container logs. `/var/lib/docker/containers` allows fluentd to access docker container metadata, it's important to set this as readOnly to increase security. the `fluentd-config` volume mount makes the configmap available inside the container. i use secrets to store the elasticsearch credentials, but you can also use env vars, it's up to you. the image used here `fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch7` it's just the base image, you may need a custom one depending on the needs.

regarding resources for further reading, i highly recommend the official fluentd documentation, it’s quite comprehensive. also, “kubernetes in action” by marko luksa is a great book that covers the fundamentals of kubernetes and things like daemonsets and configmaps in detail. it's not specific to logging but knowing the fundamentals help a lot when troubleshooting. the “kubernetes patterns” book is great to learn best practices on kubernetes. you can also look into the kubernetes documentation, which is updated constantly and a go to source for any kubernetes setup.

another thing i've learned the hard way is to test your configuration in a staging environment, not in production, or at least do it in a controlled way with one or two nodes at a time. setting up logging is critical to understand what's going on with your application, so messing with production without a detailed plan can cause a big disaster. once i accidentally broke the whole logging pipeline, it was because i was testing something on a production environment and didn't pay enough attention to the configuration, it was my fault, of course, and i never did it again. also, keep an eye on your fluentd logs, if something is not working, that's usually the best place to start debugging. you can access those using `kubectl logs -n <fluentd_namespace> <fluentd_pod_name>`. and don't forget to secure your elasticsearch instance too.

and, just because it's tradition on platforms like these, here’s a lame joke: why did the log file break up with the database? because they couldn’t agree on the schema.

anyway, i hope this helps a bit. this is a typical setup and you may need adjustments for your particular use case, but it should point you in the proper direction. let me know if you have further questions.
