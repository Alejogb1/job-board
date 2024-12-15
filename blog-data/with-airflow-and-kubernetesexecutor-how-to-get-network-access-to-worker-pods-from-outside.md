---
title: "With Airflow and KubernetesExecutor: how to get network access to worker pods from outside?"
date: "2024-12-15"
id: "with-airflow-and-kubernetesexecutor-how-to-get-network-access-to-worker-pods-from-outside"
---

alright, so you're banging your head against the wall trying to get your airflow worker pods, running on kubernetes, to talk to the outside world. been there, done that, got the t-shirt (and probably a few gray hairs). it's a common problem and can feel like pulling teeth at first, but it's totally solvable. let's break it down.

first off, the default setup for airflow on kubernetes, especially using the kubernetesexecutor, is often pretty isolated. the worker pods are usually tucked away in their little kubernetes cluster network, not exposed directly to the internet. that's generally good for security, but not so great when your tasks need to reach out to external apis, databases, or other services.

my first encounter with this was back when i was setting up a data pipeline for a tiny startup. we were scraping data from several external websites, and everything worked like a charm in local development with the sequentialexecutor. the moment i moved it to a kubernetes cluster with the kubernetesexecutor, the workers went silent, as in no data was coming in. the logs were filled with timeouts, and i couldn't figure out why. it felt like i was running against a wall of invisibility, i was a little green when starting my devops journey.

after a good amount of hair-pulling and reading through countless kubernetes and airflow docs (and way too many stack overflow posts), i realized that kubernetes networking was the bottleneck. so, the fix is not as complex as the problem may seem, let's run through the usual suspects here:

the easiest approach is to ensure your kubernetes cluster's networking allows outbound connections. this means, your pods should have a route to the internet, and any firewalls (like kubernetes network policies) should not be blocking this traffic. sometimes the kubernetes cluster itself does not have a way out to the internet by default. in the past i had to configure an internet gateway on the cloud provider to allow traffic out. if it’s on premise or using another provider you have to ensure that your network has an outbound access.

however, assuming your cluster allows internet access, the most common issue is dns resolution. worker pods are created inside the kubernetes network and usually their dns servers are internal to the kubernetes cluster. if you are trying to access an external resource by name, like an api endpoint using “https://api.example.com”, you need to make sure that your pod’s dns servers can resolve the name. this usually comes down to your kubernetes cluster configuration, but sometimes it is not obvious, specially when using cloud providers. most cloud providers have a default dns configuration but the user has to ensure it works.

you can verify that inside the pod, you can use `nslookup` or `dig` commands, which usually are available in most container images, to check if the domain name can be resolved. if the domain does not resolve you have to configure or debug your kubernetes cluster's dns resolution.

but what if the resources you need are not reachable via the public internet? for example, internal databases or apis located within a vpn or a private network. you have two main choices:

1.  **configure a service mesh or a proxy:** this approach is the most robust and scalable, but also the most complex. this involves using tools like istio or linkerd to manage your service to service traffic. these tools allow for sophisticated routing, security, and monitoring. you would then use these services to connect to your internal resources, ensuring that all traffic is encrypted and authenticated. this is definitely more complex, you'll need to get your head around these tools if you are not familiar.

2.  **kubernetes services:** this is less complex, and more straight forward, it involves creating kubernetes services to expose the target resources within the kubernetes cluster. imagine the database you want to access is inside of a private network and your cluster has access to it, you will then have to create a kubernetes service pointing to the database, so your pods can access it as if they are in the same network. this approach is good when you are not looking for complex configurations.

    in the past i had to do this in a situation where one of my workers had to connect to another service that was part of a private cloud provider. the following example showcases how you would expose the database, assuming your cluster has access to it:

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: internal-database
    spec:
      ports:
      - port: 5432 # the port in your pod
        targetPort: 5432 # the port in the db
      selector:
        app: internal-db
    ```

    and then, from within the worker pod you can use `internal-database:5432` to access the database.

now, about the code snippets you wanted, let’s keep it simple, here is a dummy airflow dag to showcase the example with external api access:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests

def fetch_data_from_api():
    url = "https://api.example.com/data"
    response = requests.get(url)
    if response.status_code == 200:
       print(response.json())
    else:
      print(f"error: got status {response.status_code}")

with DAG(
    dag_id="simple_api_fetch",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    fetch_task = PythonOperator(
        task_id="fetch_data_task",
        python_callable=fetch_data_from_api,
    )
```

this is basic but illustrates a common use case. you'll need to ensure your kubernetes worker pod has network access to "api.example.com" to get this to work. this is the first thing to check. make sure your pod dns configuration allows for this to work.

here is another example that shows how to use kubernetes services for internal access:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host="internal-database", # kubernetes service name
            port=5432,
            database="your_db",
            user="your_user",
            password="your_password"
        )
        cur = conn.cursor()
        cur.execute("select 1;")
        result = cur.fetchone()
        print(result)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"error connecting to database: {e}")


with DAG(
    dag_id="simple_db_access",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    db_task = PythonOperator(
        task_id="db_connection_task",
        python_callable=connect_to_db,
    )
```

this snippet shows how to connect to a database using the kubernetes service created in the previous example, the `internal-database` string should resolve to the internal ip of the database.

and just for a little bit of fun, here is an example of a failed attempt, for the sake of educational value. imagine you didn't configure the database service and still try to access the private ip using an incorrect name (yes, it will not work) but you will see how this would look in code, this should also help debug things if you make a mistake:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2

def connect_to_db_incorrectly():
    try:
        conn = psycopg2.connect(
            host="10.0.0.10",  # assuming this is your db ip inside the vpn
            port=5432,
            database="your_db",
            user="your_user",
            password="your_password"
        )
        cur = conn.cursor()
        cur.execute("select 1;")
        result = cur.fetchone()
        print(result)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"error connecting to database: {e}")


with DAG(
    dag_id="incorrect_db_access",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    db_task_fail = PythonOperator(
        task_id="db_connection_fail_task",
        python_callable=connect_to_db_incorrectly,
    )
```

this last one is a lesson learned the hard way. hardcoding ip's is usually a bad idea, especially in dynamic environments like kubernetes where pods and services can get rescheduled and their ips change all the time, using the kubernetes service is the correct way to go. in the beginning i did this mistake way too many times. i also had problems with dns configuration, which can lead to very similar errors, specially when working with internal private networks. the good thing is that now you know it.

for deeper reading on this i recommend checking out "kubernetes in action" by marko luksa, it provides a good foundation for understanding kubernetes networking. also, for more advanced networking concepts, i would recommend "tcp/ip guide" by charles m. kozierok, even if it is old it is very relevant, and gives you the proper tools to deal with these kinds of issues. and, of course, the official kubernetes documentation, it has everything, just make sure you use the proper version of kubernetes.

so, that's the gist of it. getting those airflow worker pods to talk to the outside world isn't rocket science, it just needs a bit of understanding of kubernetes networking and dns configurations, and sometimes some luck too. but, hey, that's what makes this fun (kinda, haha!).
