---
title: "How to run python in singularity on ubuntu?"
date: "2024-12-15"
id: "how-to-run-python-in-singularity-on-ubuntu"
---

hey there,

so, you're looking to run python within singularity on ubuntu, got it. this is a pretty common thing, and i've definitely banged my head against a wall a few times figuring out the best way to do it. i’ve been through the wringer with containerization and python, so i can share what worked for me over the years. think of it as a fellow coder trying to save you some pain.

first off, let's talk about singularity itself. it's a containerization platform, like docker, but designed more for high-performance computing and scientific workloads where security and reproducibility are key. it basically lets you package up your application and its dependencies into a single container image. this makes it easier to run the same software consistently across different environments, which, let's be honest, is a blessing in disguise given python's, shall we say, 'interesting' relationship with dependency management.

now, about python specifically. you can approach this in different ways. you could use a pre-existing image with python already installed, or you could create your own image from scratch. i've done both, and each has its pros and cons. starting with a base image, let's say, an ubuntu one, is often simpler. it's like having a ready-made foundation to build on. this is what i often recommend for beginners. however, sometimes, creating your own from scratch is needed when you need more control or need to keep the image minimal.

let's dive into a basic scenario using an ubuntu base image. for this, we’ll make use of a singularity definition file, which we’ll call `singularity.def`. this file is the blueprint for building our singularity image.

here's a very basic example of a `singularity.def` file you can use:

```singularity
Bootstrap: docker
From: ubuntu:latest

%post
    apt-get update
    apt-get install -y python3 python3-pip

%runscript
    python3 "$@"
```

let me break this down line by line. `bootstrap: docker` specifies that we're using a docker image as the base. `from: ubuntu:latest` indicates that our base image is the latest version of ubuntu from docker hub. the `%post` section contains commands that are executed during the image build process. here, we are updating the package list and installing python3 and pip. this is essential since you want to run python. finally, the `%runscript` defines the command executed when you run the container. in this simple case, we are just passing arguments to `python3` which makes the container run your python script.

to actually build this image, you'd use the command `sudo singularity build my_python_image.sif singularity.def`. note the `sudo`, you often need root to build the container. `my_python_image.sif` will be the name of the resulting singularity image. then you can run python scripts inside it with `singularity run my_python_image.sif your_script.py`. if you want more control over your python environment we can use virtual environments, now you can be sure that each project will use specific versions of python. this brings us to a slightly more advanced definition file:

```singularity
Bootstrap: docker
From: ubuntu:latest

%post
    apt-get update
    apt-get install -y python3 python3-pip
    python3 -m venv /opt/venv
    /opt/venv/bin/pip install numpy pandas

%runscript
    /opt/venv/bin/python3 "$@"
```

notice the `%post` section change? first, we create a python virtual environment in `/opt/venv`, then we install some common data science python packages using pip, numpy and pandas. the `%runscript` now uses the virtual environment python, making sure we are executing in our contained enviroment. to use it is the same `sudo singularity build my_python_image.sif singularity.def` and `singularity run my_python_image.sif your_script.py`. this is much better, since you’ll have a controlled python environment isolated from the rest of the system, which is often desired.

but, wait, there’s more! what if you need custom packages that are not on pip? for instance, libraries that are compiled and provided by your specific institution? well, we can also accommodate that. let’s say you have a custom library, and you need to copy it to the container. then, here’s how the definition file looks:

```singularity
Bootstrap: docker
From: ubuntu:latest

%files
    my_custom_lib.so /opt/

%post
    apt-get update
    apt-get install -y python3 python3-pip
    python3 -m venv /opt/venv
    /opt/venv/bin/pip install numpy pandas
    export PYTHONPATH=$PYTHONPATH:/opt
    echo "/opt" >> /opt/venv/lib/python3.10/site-packages/my_custom_lib.pth
%runscript
    export PYTHONPATH=$PYTHONPATH:/opt
    /opt/venv/bin/python3 "$@"
```

here we use `%files` to copy a custom library (`my_custom_lib.so`) into `/opt` within the container. the `%post` section includes additional actions: we are appending the path to the copied library to the `PYTHONPATH` environment variable (a common way python finds libraries) and also adding a custom path file that includes `/opt` directory to ensure the custom library is included inside the virtual environment. same as before the `%runscript` sets up the path variable again. now your python script inside the container can import and use the custom library. for this to work you’ll need `my_custom_lib.so` on the same directory as the `singularity.def` file.

a common error i’ve seen is forgetting to add the execute permissions to the file `your_script.py` when running it inside the container, that will throw a “permission denied” error. other common issue is to forget to create a virtual environment and then python package versions can mess things up.

now, i know what you are thinking, “wow this is awesome”, and i'm not sure why you’re even talking to me, since you probably can now implement all these by yourself and write much better answers than i do. but seriously, these were some of the issues i had in the past, and these tricks were how i got through it.

regarding resources, i’d recommend the singularity documentation page directly, is the most up-to-date material. for learning more about python environments and project management, i’d recommend the book "python for data analysis" by wes mckinney. it has very detailed explanations of how to do things properly. this way you will be better equipped to handle more complex projects.

remember that practice is key. the more you do it, the better you'll get at it. keep experimenting with different base images, different package combinations, and different options to better familiarize yourself with it.
