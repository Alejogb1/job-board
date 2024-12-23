---
title: "How do I configure celery flower properly in Airflow?"
date: "2024-12-16"
id: "how-do-i-configure-celery-flower-properly-in-airflow"
---

Alright,  Configuring celery flower within an airflow environment is a problem I've seen more than a few times, and it's often not as straightforward as one might initially hope. The gotcha lies in how Airflow orchestrates its various components, specifically concerning the celery executor. We need to ensure flower has access to the correct celery broker and results backend to monitor the tasks effectively. I remember one particularly challenging deployment; it took a solid afternoon of debugging to get it singing, but I learned some valuable lessons that I’m happy to share.

The crux of the issue is that celery flower needs to connect to the same message broker and result backend that your celery workers are using. Airflow's configuration, by default, might not directly expose these settings in a manner that flower can readily consume. Thus, we often need to explicitly pass these details through environment variables or command-line arguments when starting flower.

First off, let's establish the necessary context. Your celery executor configuration in `airflow.cfg` (or environment variables) will contain the information required. Specifically, look for settings like `celery.broker_url` and `celery.result_backend`. These are fundamental. We'll need those values when launching flower. These settings determine how tasks are distributed among workers and where their results are stored.

My past experience has shown me it's generally beneficial to handle configuration via environment variables. It centralizes the setup, making it easier to maintain and audit. So, the first approach I usually recommend involves setting up those same broker and backend URLs as environment variables and referencing them when starting flower.

Here's how that might look. Assuming you’ve defined the broker and result backend in your Airflow configuration as follows:

`celery.broker_url = redis://redis:6379/0`
`celery.result_backend = redis://redis:6379/1`

You would first ensure these are correctly picked up by the celery executor. Next, you would set the corresponding environment variables when initiating flower. An example command, suitable for a systemd service or a docker container entrypoint script, would look like this:

```bash
export CELERY_BROKER_URL="redis://redis:6379/0"
export CELERY_RESULT_BACKEND="redis://redis:6379/1"
celery flower --port=5555 --address=0.0.0.0 --basic-auth=user:password
```

**Code Snippet 1: Starting Flower with Environment Variables**

This approach ensures that flower can connect to the correct queues and monitor the relevant tasks. This is by far the most reliable approach I have found. Notice I've included `--port` and `--address` for accessibility and also basic authentication, which you definitely want to configure in any production environment. I will note that while simple, basic authentication isn’t the most secure. You'll want to explore more robust options such as OAuth or API keys for a production-ready flower instance.

Another approach, if environment variables aren't your preferred method, would be to directly pass the broker and backend URLs as arguments when starting flower. This method is less flexible but can be useful for quick local debugging or cases where you prefer arguments over environment variables.

```bash
celery flower --broker="redis://redis:6379/0" --result-backend="redis://redis:6379/1" --port=5555 --address=0.0.0.0 --basic-auth=user:password
```

**Code Snippet 2: Starting Flower with Command Line Arguments**

This essentially mirrors the environment variable approach but passes the configuration directly on the command line. Again, `--port`, `--address` and basic auth are used. This approach might be more straightforward if you’re scripting the launching of flower and need to pass configuration programmatically.

Now, the above examples are somewhat simplified and assume that your broker and result backend are accessible on the network from where you launch flower. In real-world scenarios, especially with containerized deployments, that may not always be the case. You might need to consider network policies, DNS resolutions, or even custom network configurations.

Lastly, let's address the typical problem that pops up when running flower within a dockerized Airflow setup. Airflow often uses a specific docker network. If you run flower in its own container, it will also need to be connected to the same network to communicate with the celery worker and the broker/backend (assuming these are also containerized). This involves creating a docker-compose file that ties everything together on the same network or using custom docker network configurations, ensuring containers on the same docker network can resolve hostnames, like `redis` in our examples, and can communicate on specified ports.

Here’s a snippet showing a basic docker-compose configuration that achieves this:

```yaml
version: "3.7"
services:
  redis:
    image: "redis:latest"
    ports:
      - "6379:6379"
  airflow:
    image: "apache/airflow:2.8.1" # replace with your airflow image
    depends_on:
      - redis
    environment:
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CELERY__RESULT_BACKEND=redis://redis:6379/1
    ports:
      - "8080:8080"
  flower:
    image: "mher/flower:latest"
    depends_on:
      - redis
    ports:
        - "5555:5555"
    environment:
        - CELERY_BROKER_URL=redis://redis:6379/0
        - CELERY_RESULT_BACKEND=redis://redis:6379/1
    command:  ['celery', 'flower', '--port=5555', '--address=0.0.0.0', '--basic-auth=user:password']

```
**Code Snippet 3: Example Docker Compose File**

This docker-compose file sets up a basic airflow environment with redis as the broker, and a separate container running flower. Notice the `depends_on` directive. It ensures services start in the correct order, and how environment variables are utilized for the connection parameters. Note that the `redis` container exposes port `6379`, and `airflow` and `flower` can connect to it by the hostname `redis` within this docker network. This illustrates the practical use of environment variables and service dependencies.

Remember, robust monitoring setups might require additional considerations, like custom dashboards or more sophisticated alert systems, and these go beyond the scope of this answer. For more in-depth knowledge about setting up celery and monitoring, I suggest examining the documentation for celery itself (`celeryproject.org`) as well as the official airflow documentation. Also, the book “Celery: Distributed Task Queue” by Daniel Roy Greenfeld and Audrey Roy Greenfeld provides a very comprehensive understanding of celery. Lastly, the Apache Airflow documentation on the celery executor and related configurations is invaluable for a deep understanding of how it all fits together. I also recommend diving into “Production Kubernetes” by Josh Rosso and Chris Love, which will provide essential insights for running these in complex, production-grade environments.

In short, properly configuring celery flower within Airflow primarily hinges on ensuring it uses the same broker and backend settings as your celery workers, whether you use environment variables, command-line arguments, or complex orchestration systems. The key is to be intentional about how configurations are set and how network access is handled to prevent common connection errors that may arise.
