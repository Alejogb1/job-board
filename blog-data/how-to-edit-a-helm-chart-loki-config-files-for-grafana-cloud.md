---
title: "How to edit a helm chart (loki) config files for Grafana Cloud?"
date: "2024-12-14"
id: "how-to-edit-a-helm-chart-loki-config-files-for-grafana-cloud"
---

alright, so you're looking to tweak a loki helm chart's configuration for grafana cloud, eh? been there, done that, got the t-shirt. or rather, got the sleepless nights and countless yaml file revisions to show for it. i've spent more time staring at indented blocks than i care to remember.

first off, let's be clear: you're not directly editing the loki chart that's running in grafana cloud's infrastructure. that's locked down. what you *are* doing is modifying the chart's values that *you* use when deploying loki *to* grafana cloud's managed loki offering. think of it like providing grafana cloud with your personal recipe for how it should set up loki.

most often, the way you do this is by using a `values.yaml` file, that’s passed when you actually install the helm chart. this file lets you specify configurations that override the defaults defined in the helm chart. when grafana cloud deploys loki, it uses these values. think of `values.yaml` as your instruction manual to the grafana cloud loki deployment.

the specific parameters you can tweak depend heavily on the version of the loki helm chart you’re using. helm charts have changed quite a bit over the last few years, and not always for the better if i'm being honest! i remember back when things were simpler, with fewer knobs to turn... ah, the good old days. but seriously, check the chart’s documentation for the exact settings available. that documentation will be specific to the chart’s version, which might differ from the one you are using.

typically, you'll find configuration sections for things like:

*   **loki:**  this is the main section. you'll usually see sub-sections within for things like ingesters, distributors, queriers, compactor, and table manager. those sub-sections have their own further sub-sections for parameters like storage config, chunk sizes, retention periods, and limits.
*   **storage:** specifies the location where loki will store its data. for grafana cloud, this will usually point to a specific bucket provided to you. you rarely touch these settings since grafana cloud will mostly manage this for you, but in some oddball cases you might need to modify them.
*   **limits:** sets limits on things like ingestion rate, query concurrency, and so on. these are crucial for preventing loki from crashing when it's under heavy load, which will happen, eventually. this is one of the main places you will be touching settings in `values.yaml`.
*   **compactor:** configuration related to how loki compacts its log data, basically the time where loki will merge small chunks into bigger chunks for better performance. you can fine tune its settings, but you don’t usually need to.
*   **query-frontend:** it handles the query load. this configuration can help to improve how fast your queries get their results.
*   **ingester:** settings about how loki ingest data, the main parameters are chunk size and time before flushing to storage.
*   **promtail:** in case you also install promtail as a daemonset in your cluster, this section would be relevant.

now, let's get to some concrete examples. assuming we are working with a fairly modern loki chart, here's a `values.yaml` example where i'm showing you a few parameters to edit to tune ingestion:

```yaml
loki:
  ingester:
    chunk_idle_period: 1h
    chunk_target_size: 1572864
    max_chunk_age: 2h
    lifecycler:
      join_after: 1m
  limits_config:
    ingestion_rate_mb: 20
    ingestion_burst_size_mb: 30
```

in that example, i've modified the settings to instruct loki to write chunks of 1.5 mb and flush them after 1h if they don’t fill, and set a maximum time of 2h after creation to flush them. i also increased the ingestion rate to 20mb and a burst of 30mb, which can help to handle periods with high traffic. these are parameters you may modify when troubleshooting ingestion problems. this is a small snippet but shows you the way you normally edit these values.

let's look at another example. suppose you need to modify the limits of your querier and query frontend. here's an example of how you can accomplish that through `values.yaml`:

```yaml
loki:
  query_range:
    split_queries_by_interval: 5m
    max_query_parallelism: 20
  frontend:
    max_outstanding_per_tenant: 200
    query_timeout: 3m
```

in this scenario, i am telling loki to split queries that take longer than 5m into smaller ones, allowing up to 20 queries in parallel. i also configured the frontend to allow up to 200 queries per tenant and to timeout queries taking longer than 3m. these settings are useful to avoid overwhelming loki during high query load.

one common point to adjust is the retention policy. if you don’t adjust this, your costs may rise since your data will be kept for a long time. here's how to do that:

```yaml
loki:
  compactor:
    working_directory: /tmp/loki/compactor
    retention_enabled: true
    retention_delete_delay: 48h
    retention_delete_worker_count: 10
    compaction_interval: 15m
    compaction_concurrency: 5
```

in this case, i'm telling loki to delete any log data older than 48 hours, using 10 concurrent workers to perform this task. the compaction interval also can be adjusted, but this value is already a good default in most cases. be very careful when you change retention periods, since if you reduce it too much you may lose important log data. i once changed it too low and had to troubleshoot an issue with basically no data available... not fun.

when you've made the changes to your `values.yaml`, you apply these values when installing or upgrading the helm chart. you generally use the command `helm upgrade --install loki grafana/loki -f values.yaml`. it’s also important to do it incrementally. make small changes and monitor if everything works as expected before making any other change. this is probably the most important tip i can give you to deal with helm charts. if you apply too many changes at once, it will be very hard to pinpoint where the issue lies if things break.

a tip: keep a close eye on the helm chart version and corresponding documentation. i’ve spent many hours troubleshooting weird issues only to discover they were due to a documentation error or subtle chart change. this is a frequent case for these charts since they tend to change often. one thing that works on one version might not work in another, it is maddening at times. it’s like they change stuff just for the sake of changing it! (but i’m sure they have good reason)

also, keep in mind that grafana cloud has its own internal configurations and limitations. you won't be able to tweak *everything*, and sometimes the changes you make might not produce the effect you expect if grafana cloud has some internal configurations that clash with yours. that’s something to consider when troubleshooting.

as for resources, i'd recommend checking the official helm chart documentation on the helm repository directly, and the loki documentation itself, usually found on grafana labs’ website. that’s the source of truth. there are also some great books on kubernetes and helm if you want to get deeper into how all of this works, and i'd recommend learning more about the theory of how loki works internally, that might also help you to understand why certain parameters do what they do. “kubernetes in action” by marko luksa is a great book to learn more about k8s in general. and the official grafana documentation and the official loki website are very useful for loki specific details.

so yeah, that’s how you edit the loki chart for grafana cloud. remember, it’s all about that `values.yaml` file. keep it versioned, test changes incrementally and you’ll be fine. and remember always the version you are working with, that’s crucial. hope this helps. and hey, if things are not working try restarting the pod… usually fixes the issue… (just kidding, kinda).
