---
title: "How do I configure Celery Flower in Airflow?"
date: "2024-12-23"
id: "how-do-i-configure-celery-flower-in-airflow"
---

,  I recall a project a few years back where we had a particularly convoluted DAG structure relying heavily on Celery, and getting Flower properly configured within Airflow became crucial for observability. It wasn't exactly straightforward, but with a bit of finesse, we got it humming. You're essentially looking at bridging two separate systems, and that's where things can get nuanced.

At its core, Celery Flower is an independent real-time monitoring tool for Celery tasks. Airflow, on the other hand, manages the scheduling and execution of those tasks via Celery. The key to successful integration is ensuring that Flower can effectively access the Celery broker and results backend being used by your Airflow Celery executor. Now, the first thing to consider is **how** you are running Airflow and Celery. Are they both within the same machine or across multiple containers? This impacts the configuration. Let me walk you through a typical setup and the rationale behind it.

Firstly, you'll need to ensure that Flower is installed. You can typically accomplish this through pip: `pip install flower`. Simple enough, but don't forget to do this within the virtual environment that contains both Airflow and Celery. Once installed, the actual configuration begins.

Here's the first code snippet, representing a foundational configuration approach when everything is in the same machine (a development environment, for instance):

```python
# airflow_settings.py (or airflow.cfg)

[celery]
flower_port = 5555 # or any unused port
flower_address = "127.0.0.1" # or specific IP if needed
flower_basic_auth = "user:password"  # For initial security
broker_url = "redis://localhost:6379/0"  # Your broker URL from Airflow

[flower_config]
address = 127.0.0.1
port = 5555
```

Now, what's going on here? We're adding configurations under two headers: `[celery]` and `[flower_config]`. Under `[celery]`, `flower_port` and `flower_address` defines how you access the Flower UI, locally. You specify the address and the port number you will use to open it in a browser. The basic authentication `flower_basic_auth`, provides some rudimentary protection and **should never** be used in a production environment, instead, you should use more secure authentication mechanisms like oAuth. Critically, the `broker_url` setting **must** exactly match the broker URL you have configured in your Airflow setup under the `[celery]` section, within `airflow.cfg`. This is where Flower determines which Celery broker to monitor. Note: In Airflow versions 2.0+, these settings are typically located within your `airflow.cfg` (or the equivalent configuration file).

You’ll need to ensure your celery worker process is running which you usually achieve with something like `airflow celery worker` and the flower instance itself must also be run using something like the command below:

```bash
flower --broker=redis://localhost:6379/0 --port=5555 --address=127.0.0.1 --basic-auth=user:password
```

The arguments passed here are pulled directly from the config file values, ensuring Flower connects to the correct broker. You can access the flower web UI by going to `http://127.0.0.1:5555`.
This setup works when your Airflow and Celery workers are on the same machine.

Now, a more complex scenario arises when your Celery workers are distributed across multiple machines or within a Kubernetes cluster. In this case, Flower needs access to the broker from outside the immediate Airflow server.

Here's the second code snippet illustrating that, incorporating a `CELERY_FLOWER_ADDRESS` environment variable which will be used in the `flower` start command:

```python
# airflow_settings.py (or airflow.cfg)

[celery]
flower_port = 5555
flower_address = "0.0.0.0"  # Allow external connections (be careful)
flower_basic_auth = ""  # Remove basic auth, use another auth mechanism for production
broker_url = "redis://your-redis-server:6379/0"  # Your actual redis server

[flower_config]
address = 0.0.0.0
port = 5555

```

In this scenario, we've set `flower_address` to `"0.0.0.0"`, which makes it accessible from any IP address, allowing you to view the Flower UI from a different host (remember security implications). Again, the `broker_url` now points to your shared message broker accessible by the celery workers and the Flower instance. Also note that the flower basic auth has been disabled. You must implement proper authentication using reverse proxies and a proper authentication mechanism such as oAuth. In this more complex example, we'll assume you have already deployed Airflow and the celery workers. You can start Flower from any location that has access to the message broker via `redis` or another protocol.

```bash
export CELERY_FLOWER_ADDRESS="http://your-flower-server-address:5555"
flower --broker=redis://your-redis-server:6379/0 --port=5555 --address=0.0.0.0
```

Here, we are setting an env var that we can use to configure Airflow’s `Monitoring->Flower` functionality if we wish. This makes the monitoring feature available on the UI.

Finally, let's examine how to integrate this directly into Airflow, particularly important for versions 2.0 and higher where the webserver supports this integration directly. This involves updating your Airflow configuration file.

```python
# airflow_settings.py (or airflow.cfg)

[webserver]
flower_url = http://your-flower-server-address:5555  # The address where flower is running
```

Within the `[webserver]` section, the `flower_url` directs Airflow to where Flower is accessible. Now, within the Airflow UI, the "Monitoring" menu will display the Celery Flower UI through an iframe. This integration simplifies accessing Flower directly from your Airflow instance.

A word of caution; be extremely careful when exposing Flower through `"0.0.0.0"`. It's critical to implement robust authentication and authorization in a production environment (like oAuth, reverse proxies, or VPN tunnels). The default basic authentication mechanism is unsuitable for real-world scenarios due to security vulnerabilities.

For further reading and a deeper dive into the specifics, I highly recommend reviewing the official Celery documentation, specifically the section on "Monitoring and Management". The book "Celery: Distributed Task Queues" by Marcin Nowak also offers a comprehensive guide, particularly when combined with "Programming with Celery" by Michael R. Smith. Finally, you may also find the official Airflow documentation section detailing “Celery Executor” of use.

In summary, configuring Celery Flower in Airflow requires precise setup, ensuring the broker URL is properly specified for both Celery and Flower, particularly when working across multiple servers. Remember to adjust addresses and ports according to your network settings, and always prioritize security with a good authentication mechanism, especially for production environments. This configuration has been refined by years of working with these tools, and I've found that sticking to these practices helps keep complexities at bay.
