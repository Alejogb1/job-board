---
title: "Why can't the Airflow HiveServer2Hook see query progress in logs?"
date: "2024-12-14"
id: "why-cant-the-airflow-hiveserver2hook-see-query-progress-in-logs"
---

ah, i see. the classic 'airflow cant see my hive query progress' conundrum. i’ve been there, and it's a frustrating one, it often comes down to a few specific areas. let's break it down like we're debugging a particularly stubborn piece of code, because that's pretty much what it is.

so, you're using airflow's `hiveserver2hook`, launching hive queries, and expecting to see progress updates in the logs, but instead, you're probably getting mostly silence, maybe just the initial query execution message and the final status. this usually isn’t an issue with airflow itself, but more about how hive and thrift are configured, and how airflow interacts with that setup.

first thing, lets get the lay of the land. `hiveserver2hook` uses thrift to connect to hiveserver2. this means airflow isn’t directly executing the query in the same way a `hive -f` call would. when you use the cli directly, hive is outputting detailed logging directly to your terminal. however, when using thrift through a hook like this, the logs aren’t automatically funneled to airflow.

let's look at what's actually happening under the hood. when airflow sends a query to hiveserver2 via thrift, the thrift connection gets a ‘handle’ to the running query. this handle allows us to check the status, grab results, and generally manage it. however, by default, the thrift interface doesn't automatically forward logging to the client. the server is writing log information, but its going into hive's server side logs, typically in the hadoop yarn logs. we dont want to go there. we want to get them directly in airflow.

the main issue usually boils down to the configuration of hive’s log settings, particularly how its configured with respect to thrift, and how we're handling it. hive itself has a lot of configuration knobs and whistles that are sometimes difficult to comprehend. it is important to understand that hive’s default logging setup doesn’t typically push query progress to the thrift interface.

i remember, back in my early days of dealing with big data, i spent almost a week on a similar issue. it was a particularly challenging weekend i spent debugging a pipeline at 3am on saturday, fueled by way too much coffee. we'd set up a new airflow cluster connected to an existing, very large hive cluster. we kept getting no progress logs, and we were stuck wondering why our jobs were taking so long and if they had stalled at any moment. the lack of visibility was just painful, you cant fix what you cant see. the answer was surprisingly simple once we went deep enough. we needed to tell the hive server to output this information back via thrift, and tell the client to be ready to read it.

so how can we fix it? this involves two main aspects: configuring hive correctly, and using the `fetch_logs` from airflow.

first, for configuring hive, we need to make sure that hive sends log messages through the thrift interface. this often involves playing with the `hive.server2.async.exec.threads` property. increasing that, and ensuring the property `hive.server2.log.operation.enabled` is true. these tell hive to send asynchronous operation log messages through thrift.

here's an example of how you would set these properties when starting the hive server, or in `hive-site.xml`

```xml
<property>
  <name>hive.server2.async.exec.threads</name>
  <value>20</value>
  <description>number of threads to use for async query execution</description>
</property>

<property>
  <name>hive.server2.log.operation.enabled</name>
  <value>true</value>
  <description>whether operation logging should be enabled.</description>
</property>
```

or if you are passing this as a startup option when launching the hiveserver, you do something like

```bash
hiveserver2 --hiveconf hive.server2.async.exec.threads=20 --hiveconf hive.server2.log.operation.enabled=true
```

after these changes in hiveserver, you are on the server side done. it is now your responsibility in airflow to tell your hive hook to fetch these logs.

now to the airflow part. the magic is in the `fetch_logs` method. after you execute a query using `hiveserver2hook.run`, you should call the `fetch_logs` method of that same object. this is where the magic happens. this is how airflow is told to fetch those log messages that have been configured to be sent by hive to begin with.

here’s an example of a custom operator where you will execute a hive query, and then fetch the logs. the `fetch_logs` method will get the operation logs and you can then output it to airflow’s logs.

```python
from airflow.models.baseoperator import BaseOperator
from airflow.providers.apache.hive.hooks.hive import HiveServer2Hook

class HiveQueryOperatorWithLogs(BaseOperator):
    template_fields = ['hql']

    def __init__(self, hive_cli_conn_id, hql, **kwargs):
        super().__init__(**kwargs)
        self.hive_cli_conn_id = hive_cli_conn_id
        self.hql = hql

    def execute(self, context):
        hook = HiveServer2Hook(hive_cli_conn_id=self.hive_cli_conn_id)
        hook.run(self.hql, handler = lambda x: self.log.info(x))
        self.log.info("Hive query executed")
        for log in hook.fetch_logs():
             self.log.info(log)

```

this custom operator is using the `HiveServer2Hook`, it then runs the `hql` query passed in with the init of the operator, and will be using the `handler` parameter of the hook to get the status log output from the initial query. after this the operator iterates through the results of `fetch_logs` and will write the rest of the query operation logs.

here is another code example showing the same idea in a shorter way using a python operator:

```python
from airflow.decorators import task
from airflow.providers.apache.hive.hooks.hive import HiveServer2Hook
from airflow.utils.log.logging_mixin import LoggingMixin

@task
def run_hive_query_with_logs(hql, hive_conn_id):
    log = LoggingMixin().log
    hook = HiveServer2Hook(hive_cli_conn_id=hive_conn_id)
    hook.run(hql, handler = lambda x: log.info(x))
    log.info("Hive query executed")
    for log_msg in hook.fetch_logs():
        log.info(log_msg)

```
this is the same principle. you'd just call this task from a dag.

a good thing to always check is the yarn logs for the map reduce jobs too. because if those fail, or take too long, you might not see anything either. this is unrelated to the thrift logs, and is more related to the actual execution of the mappers and reducers. those logs are often invaluable in debugging performance or errors.

if you still have issues, check for any firewall rules that might be blocking thrift messages, make sure your dns resolutions are working properly and ensure that hive is using the correct hostname and port for thrift. check that your jars are where they are expected to be. sometimes its that last dependency that is not in the correct path, that is creating issues. it might not be the code, it could be a server configuration setting.

now, about resources. instead of links, i recommend diving into the following:
1. **the official apache hive documentation:** the hive documentation is the ultimate source of truth for all things hive. the sections on configuration parameters, specifically those around thrift and logging are crucial. pay special attention to the properties i mentioned earlier.
2. **'hadoop: the definitive guide' by tom white:** this book is fantastic for a deep dive into hadoop, including hdfs and mapreduce, which are integral to hive’s backend processing. understanding these components helps diagnose issues that manifest as silent queries. also helps troubleshoot any potential yarn issues.
3. **thrift documentation:** if you really want to understand the nitty gritty of how data moves between hive and airflow, reviewing the thrift documentation will get you into the weeds of what is happening with this connection.

and a random joke for you; why did the sql database break up with the nosql database? it said, you have no structure!

remember that debugging these types of issues isn't always straightforward, but with patience and a systematic approach, you can get to the bottom of it. you need to first make sure that the server side has all configurations properly set, and then you need to make sure your airflow code is reading these configurations. happy coding!
