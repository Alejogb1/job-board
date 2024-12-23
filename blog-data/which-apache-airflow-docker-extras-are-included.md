---
title: "Which Apache Airflow Docker extras are included?"
date: "2024-12-23"
id: "which-apache-airflow-docker-extras-are-included"
---

, let's break down the Apache Airflow Docker extras. It's not uncommon to see confusion around this, and honestly, in my early days building data pipelines, I stumbled over this myself, leading to some frustrating deployment issues. Essentially, these "extras" are curated collections of python packages installed alongside Airflow when you use specific docker images from the official apache/airflow repository (or its derivatives). They aren't arbitrary; they're designed to streamline workflows for common use cases and minimize the ‘dependency hell’ that can arise when you're integrating various systems.

Instead of everything being a monolithic, bloated install, Airflow uses a neat extras mechanism, allowing you to tailor the docker image to what you specifically need. Think of it as a way to specify the "plumbing" your data workflows will use without being burdened by packages you won't touch.

Now, the extras aren’t just haphazardly bundled; they typically fall into categories like cloud providers (aws, google, azure), database connectors (mysql, postgres, etc.), data processing frameworks (spark, dask) and specific APIs (http, slack). When constructing your docker build or using a pre-built image you can choose to install these via specific tags using pip. If you’ve used virtual environments, the ‘extras’ concept should feel similar to the idea of installing dependencies with environment markers which makes understanding it even more straightforward.

Let's dive into some specifics. First, there's the core set, let's call it the 'base' setup. Even without specifying extras, you get the essential Airflow components and some minimal dependency requirements for basic scheduling and task orchestration. This alone usually doesn't cut it in complex scenarios, which is where the power of extras shines through.

Let me give you some concrete examples, drawing from my own experiences, and let’s include some snippets to illustrate how to actually use them.

**Example 1: Interacting with Amazon S3**

A while back, I worked on a project that involved loading data from an S3 bucket into a data warehouse. We quickly learned that the basic airflow image didn't include the `apache-airflow-providers-amazon` package, which contains all the necessary operators and hooks to work with AWS services like S3, SNS, SQS, and so on. Attempting to use the `S3KeySensor` or `S3Hook` straight out of the gate led to those infamous `ModuleNotFoundError` errors.

To correct this, we had to modify our dockerfile. In order to use this, we need to ensure that the package is installed into our docker image build, here's an example:

```dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install apache-airflow-providers-amazon
```

In this simplified Dockerfile, we take the official image as our base. The `RUN` instruction executes `pip install apache-airflow-providers-amazon` inside the docker image during build time. This adds the necessary packages for AWS integration. After building this docker image, you'd be able to use S3 related features in your DAGs such as `S3Hook` or `S3KeySensor`.

Now, in my experience you’d likely want a more modular approach using build-args:

```dockerfile
ARG AIRFLOW_VERSION=2.8.1
FROM apache/airflow:${AIRFLOW_VERSION}

ARG AIRFLOW_EXTRAS="amazon"

USER airflow

RUN if [ -n "$AIRFLOW_EXTRAS" ]; then \
      pip install "apache-airflow[${AIRFLOW_EXTRAS}]"; \
  fi
```
Here we introduce two build-args to make our build process more flexible allowing us to specify the base airflow version and the docker extras we wish to include during build. To use this we would run `docker build --build-arg AIRFLOW_EXTRAS="amazon,cncf.kubernetes" .`

**Example 2: Using the `http` extra**

Another project involved integrating with several third-party APIs via HTTP requests. Again, the default image did not have the needed http-related features. We found ourselves needing the `HttpOperator` and `HttpHook`, which are contained in the `apache-airflow-providers-http` package.

A simple way to add this, again during the docker image build time is:

```dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install apache-airflow-providers-http
```

Similar to the first example, this adds the provider, making http capabilities available within Airflow. However, this is not exactly how you would generally go about it. It's much more common to use the extra mechanism:

```dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install apache-airflow[http]
```
Or, using the same argument driven approach as above
```dockerfile
ARG AIRFLOW_VERSION=2.8.1
FROM apache/airflow:${AIRFLOW_VERSION}

ARG AIRFLOW_EXTRAS="http"

USER airflow

RUN if [ -n "$AIRFLOW_EXTRAS" ]; then \
      pip install "apache-airflow[${AIRFLOW_EXTRAS}]"; \
  fi
```
This snippet leverages the built-in mechanism for declaring extras, ensuring a cleaner and more standard approach.  You can see the structure here `apache-airflow[http]` means install Airflow with the `http` extra. This is the recommended way, it avoids having to manage multiple versions of the providers.

**Example 3: Data Transformation with Spark**

A more advanced scenario we encountered involved using Spark for complex data transformations. Again, the base image lacked all the needed elements, specifically the spark provider. We found ourselves needing to use the `SparkSubmitOperator` to orchestrate those spark jobs which meant that the `apache-airflow-providers-apache.spark` needed to be added.

```dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install apache-airflow-providers-apache-spark
```
Using the `pip install` approach can get you started quickly, but you’ll quickly find it inflexible as the number of needed extras begins to scale. You’ll also find yourself re-installing common extras such as sql, mysql, postgres, etc.. a lot, and having to manually maintain consistent versioning. It is much better to adopt a single line approach using the extras mechanism:

```dockerfile
FROM apache/airflow:2.8.1

USER airflow

RUN pip install apache-airflow[apache.spark]
```
Again, this installs the package needed to provide spark functionality for our workflows, but you’ll generally do this using a build argument to allow for easy configuration.

```dockerfile
ARG AIRFLOW_VERSION=2.8.1
FROM apache/airflow:${AIRFLOW_VERSION}

ARG AIRFLOW_EXTRAS="apache.spark"

USER airflow

RUN if [ -n "$AIRFLOW_EXTRAS" ]; then \
      pip install "apache-airflow[${AIRFLOW_EXTRAS}]"; \
  fi
```
This is now a common theme. Build-args allow us to easily and consistently specify dependencies to our docker image without having to remember the specific package names, or how to install them.

**A practical note about managing your setup**

Generally, you won’t be manually specifying single extras every time, you’ll typically accumulate a larger list over time, making build-args extremely useful. The complete list of extras is regularly updated within the Airflow documentation, and it’s definitely worthwhile to look there for the latest options and version compatibility information. Remember that provider versions and Airflow versions need to be compatible with each other. This is something you will almost certainly run into at some point in time, and debugging version incompatibility is something that’s worth avoiding if you can.

For a deep dive into this topic, I’d recommend examining the apache airflow documentation on “providers”. The official documentation will provide a complete reference and provide version information for all of the extras and their associated provider packages. Also, reading through the Dockerfile for the official Apache Airflow Docker images is extremely helpful to see how the extras mechanism is used in practice. And if you want to understand dependency management in python, I recommend reading through the 'Python Packaging User Guide' published by the Python Packaging Authority. They provide a comprehensive overview of dependency management in python, and they’re an invaluable source for anyone working with python projects.

In conclusion, the apache airflow docker extras are a crucial part of the airflow ecosystem. They're designed to provide a flexible and manageable way to install only the necessary dependencies for your workflow. It takes a bit of getting used to, and I've definitely spent time resolving issues caused by missing the correct extras. But by understanding how they work and using the official documentation you’ll be able to streamline your development and avoid many headaches along the way. By being proactive and using a well structured, argument-driven approach to your docker builds you can avoid a lot of time wasted later.
