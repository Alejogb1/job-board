---
title: "Why is an Environmental variable is causing an issue when using singularity to run a docker container on a supercomputer?"
date: "2024-12-14"
id: "why-is-an-environmental-variable-is-causing-an-issue-when-using-singularity-to-run-a-docker-container-on-a-supercomputer"
---

alright, let's break down why an environment variable might be causing headaches when you're running a docker container inside singularity on a supercomputer. it’s a pretty common pitfall, and i've definitely spent more time debugging this than i'd like to remember. it usually boils down to how environment variables are handled at different layers of this tech sandwich: the supercomputer's shell, singularity, and finally, the docker container.

first off, think of environment variables like global settings for your system. when you log into a supercomputer, you're usually starting within a specific shell environment (like bash or zsh), which is configured with certain variables. these can point to crucial things, like where executables are located (`$PATH`), locations of libraries (`$LD_LIBRARY_PATH`), or specific preferences for tools. when you launch singularity, it creates an isolated space for your container to run. however, it doesn't automatically inherit *all* the environment variables from your shell. some, like `$PATH`, are usually passed through, but others, particularly those related to the cluster's specific setup, may not be. this is where problems start to bubble up.

the singularity container, by default, will try to use its own clean environment, sometimes it does try to inherit from the host, sometimes it does not it depends on the settings that were passed during the container build or the run. if a docker container relies on an env var that isn't set inside singularity it might act weirdly or fail. i saw this a lot back when i was working on a genomics pipeline, the container was crashing with an error complaining about a missing dependency but the dependency was present. after hours i realised that the variable pointing to the modules directory was not being passed to the container and it had to be passed manually.

then we've got the docker container itself. it might have its own expectation of environment variables, often defined in its `dockerfile`. if these variables are different from what's available at the host, or what singularity passes, boom. conflict. for example a very common issue is when a docker container expect a specific user or UID to do something within the container but the host environment does not set that same UID and it tries to access protected files.

here is the common scenario i’ve encountered the most.
say on the host you have a crucial variable defined:

```bash
export MY_CLUSTER_LIBRARY=/opt/supercluster/lib
```

and you then run a singularity container based on a docker image that expects that variable in order to execute correctly.

```bash
singularity exec docker://my-docker-image:latest my_script.sh
```

but, the `my_script.sh` inside docker might complain about missing libraries or modules. and even if you set the variable before, singularity doesn't automatically pass it through, unless explicitly told to.

so, here is a example of how you can tackle this:

**method 1: passing the variable directly to singularity**

instead of relying on environment inheritance, the most reliable method is to explicitly pass the variable when you run the singularity command. so the previous example would change into this:

```bash
MY_CLUSTER_LIBRARY=/opt/supercluster/lib singularity exec --env MY_CLUSTER_LIBRARY docker://my-docker-image:latest my_script.sh
```

the `--env` flag lets you add specific variables to the container environment. it's cleaner and ensures that singularity gets the variable you want.

this was one of the first lessons i learned when i was still very naive. i was building scientific workflows with containers using `singularity` on an HPC environment. my early containers kept crashing with bizarre errors related to missing python modules even after they were `pip install`ed, and for the longest time i thought it was a bug with the container image, but in reality i needed to pass the PYTHONPATH variable to the singularity container. let's face it containers sometimes feel like a black box until you find that small detail you forgot.

**method 2: use a bind path to pass a .env file**

if you have a lot of variables or variables that are too sensitive to be exposed to the command line you can also use a `.env` file. this is very useful when you have different configurations for different scenarios. to use this you first have to create the `.env` file.

```bash
# example of env file
MY_CLUSTER_LIBRARY=/opt/supercluster/lib
MY_ANOTHER_VAR=some_other_value
```

then you just execute the singularity container, remember that the `.env` file has to be visible from within the container. i.e you have to bind it to the container.

```bash
singularity exec --bind /path/to/my/.env:/my/.env  docker://my-docker-image:latest my_script.sh  
```

then within the `my_script.sh` you load the variables using the `source` command

```bash
source /my/.env

# rest of the script
```

i found this particularly helpful when dealing with multiple project configurations where each required slightly different library locations or when i need to pass credentials in a secure way without exposing them in the cli. this kept our code clean and our sanity mostly intact. i say mostly, because sometimes the problem is not that the env var is not there but is being overriden. and that can get you really confused.

**method 3: using singularity's 'contain' flag**

finally there's another case where variables are a problem. the container might be configured to run with a specific user, usually root within the container, but your actual user on the supercomputer is a different one. this sometimes lead to weird permissions issues when accessing shared filesystems within the supercomputer. to fix that you can force singularity to use the same user and group with the `--contain` flag, this is useful when the user configuration is important to get a job running:

```bash
singularity exec --contain docker://my-docker-image:latest my_script.sh
```

this method has to be used with a little caution because you can over do it. if the container needs to be run by `root` this will cause the container to run as a `root` user in the host, and this can open security holes if you do not know what you are doing. you can also use other `contain` flags such as `--containall` this one is even more aggressive and it will not pass any environment from the host. so be careful with it. it's good to remember to never trust containers blindly, always be aware what you are telling them to do.

now, what's the reason for all of this complexity? it's all for security and portability. containerization is about creating self-contained environments that can run anywhere with the same results, without depending on the host. so to ensure this is always the case they try to isolate from the environment as much as possible.

if you want to really understand the nitty gritty, i'd recommend going through the singularity documentation on environment handling (i do not like to link directly to web pages since they change a lot, but if you type singularity environment variables documentation on google you can usually find it pretty easily, this is the same for docker). the docker documentation regarding how to handle environment variables is also a must (again same strategy to find the right documentation).

and finally, i also found a paper from a few years ago by the singularity team about the principles behind containerization for scientific computing. i believe it was called 'singularity: containerizing hpc for scientific workflows'. i've always found that paper very useful when trying to understand the philosophy behind singularity. it helped me understand why things are the way they are.

debugging this kind of issue sometimes feels like trying to find a single misplaced sock in your entire laundry basket, but at least now you have more tools to approach the problem, and hopefully, my past pain can save you some time. remember to always check all layers, and use the right methods to pass variables.
and last tip, if it takes more than 1 hour, it's probably not your code, is your environment.
