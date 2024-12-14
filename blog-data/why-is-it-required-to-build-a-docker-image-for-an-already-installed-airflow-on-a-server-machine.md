---
title: "Why is it Required to build a docker image for an already installed airflow on a server machine?"
date: "2024-12-14"
id: "why-is-it-required-to-build-a-docker-image-for-an-already-installed-airflow-on-a-server-machine"
---

well, i've been around the block a few times with airflow, and this question hits a nerve. it’s not an uncommon point of confusion, especially when you’re coming from a more traditional server setup. let's break down why building a docker image, even when airflow is seemingly running fine on a server, often becomes necessary and beneficial, and, let me tell you, i’ve been there.

the surface level answer is: it's all about consistency and reproducibility, but there's more to it than that. think of your existing server airflow setup as a unique snowflake. it's got all its dependencies, python versions, libraries, packages, custom scripts etc., all living in its own little world. when you say it’s already installed, what that really means is it's working on *that* specific server *now*. what happens when you need to deploy to another server or another environment? perhaps to scale up to handle increased load or to introduce a test environment.

i remember this particularly painful experience about five years ago when I joined a startup. they had airflow running on a single, lovingly crafted ec2 instance. a single server, with a python version that was a few years old, libraries that were, shall we say, 'curated' over time and it was frankly a mess. when i tried to spin up a test environment to try something out i quickly realized that reproducing that specific setup was impossible. each time we tried it was like a 'will it work?' lottery. needless to say, things broke, a lot. we had hours of debugging just chasing down missing packages or wrong version conflicts and just wasted precious resources that we could use to improve the project at hand, instead we spent them re-inventing the wheel.

docker images aim to solve exactly this problem. instead of relying on that snowflake server, a docker image encapsulates your application along with all its required dependencies, into a single, portable package. it's like taking a snapshot of your perfectly working airflow environment, ensuring that it runs the exact same way every time, everywhere. no more 'works on my machine' nightmares.

here's the first reason, and it's a big one: environment parity. you can develop locally, test on a staging environment and deploy to production, all using the same image and this is something that a regular server set up can't guarantee because usually people will install things different on different servers. with docker, you have the peace of mind that if it works in your local container, it’ll very likely work in the production one. no surprises. and if it does not work it is way faster to troubleshoot and figure out what's going on with a single image with all the code inside.

also another benefit of docker is isolation. docker containers run in their own isolated environment, separate from the host operating system and from each other. this means your airflow setup won't interfere with other applications or libraries on the server, and vice versa. the days where a python upgrade on a single server would take down a whole stack are gone (or at least that's what docker aims to achieve). in a way, it is like having different little virtual machines but instead of a full virtualized system, you have containers that are way more lightweight.

now, let's talk about updates. upgrading airflow can sometimes feel like playing russian roulette. some old packages might not be fully compatible or, you might hit weird incompatibilities in the python version. the way people do it is by trying things on production because they are not set up to test in advance and then, when it breaks, they spend a lot of time fixing it. with docker, you simply build a new image with the updated airflow version, test it in a controlled environment, and then deploy the new image. if something goes wrong, you can easily roll back to the previous image. the rollback process takes a few minutes as opposed to hours if you had to debug everything on a single server install.

so, i've covered the "why", but how does it practically look? well, let's take a glance at some code snippets. these are illustrative examples, not production-ready recipes and, of course, you'll want to adapt them for your particular setup, *and always*, read the official documentation.

first up, a basic `dockerfile` to build an airflow image:

```dockerfile
from apache/airflow:2.7.1-python3.11

# install any additional python packages if needed
copy requirements.txt .
run pip install -r requirements.txt

# add your dags to the image
copy dags /opt/airflow/dags

# add custom configuration
copy airflow.cfg /opt/airflow/airflow.cfg

# if you need custom scripts
copy scripts /opt/airflow/scripts
```

this dockerfile starts from a pre-built airflow image. it adds your custom requirements, your dags, configuration and scripts. pretty simple, right? it is very powerful because you can customize it to your needs. of course, this might change on your use case, but it shows a good starting point.

second, you might need an example `requirements.txt`:

```text
apache-airflow-providers-postgres
pandas
requests
```

this lists the extra python packages required by the airflow environment that should be installed before executing any dags. it is a best practice to keep it short and only add packages that are needed for the whole project, and then install the ones that are only needed for specific dags directly on the dag definition, this helps with the build time and size of the docker image.

finally, some example of airflow configurations on `airflow.cfg`:

```ini
[core]
dags_folder = /opt/airflow/dags

[webserver]
web_server_port = 8080

[database]
sql_alchemy_conn = postgresql://airflow:airflow@host.docker.internal:5432/airflow
```

as you can see, this is pretty self explanatory, you can customize everything. this example shows how to change the dags folder and the database connection. you might need to adjust these configurations to match your own environment.

let’s not forget about scaling. if you're running airflow on a single server, what happens when your job load increases? with docker, you can easily scale up your airflow deployment by running multiple containers. you can use container orchestration platforms like kubernetes to manage these containers, providing an easy way to scale out your workflow execution.

i've seen way too many projects held back by fragile, hard-to-reproduce deployments. spending a bit of time up front to build a solid docker image for your airflow setup is worth it. it’s all about efficiency, scalability, and less late night debugging sessions. it makes your life way easier down the line.

i think the best resources to dive deeper are the official docker documentation, as it is a vast subject, start with the docker compose. then check the kubernetes documentation if you need to scale to more than one instance. the airflow documentation also provides very good examples for building images. i also recommend 'kubernetes in action' by marko luksa to understand what a cluster looks like and how to manage it. do not trust blog posts, always go to the source! there is a lot of outdated information in the internet and, always refer to the latest versions available.

i hope this explanation clears things up a bit. if you have further questions, throw them my way. i'm happy to help. just not with any questions about why my last attempt at building a dag to make coffee resulted in a full-blown espresso machine malfunction. some secrets, even for me, are best left buried.
