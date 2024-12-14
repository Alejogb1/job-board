---
title: "Why am I getting a tensorflow.python.framework.errors_impl.InvalidArgumentError: 'visible_device_list' listed an invalid GPU id '1' but visible device count is 1?"
date: "2024-12-14"
id: "why-am-i-getting-a-tensorflowpythonframeworkerrorsimplinvalidargumenterror-visibledevicelist-listed-an-invalid-gpu-id-1-but-visible-device-count-is-1"
---

well, let's get into this `tensorflow.python.framework.errors_impl.InvalidArgumentError`, it's a classic and i've seen this exact error more times than i care to remember. the core problem, as the message hints, is a mismatch between what you're telling tensorflow about your gpus and what it actually detects or is allowed to see. basically, you're trying to use a gpu id that tensorflow isn’t aware of or can’t access.

i remember when i first bumped into this, i was trying to parallelize training across two gpus on a workstation i had cobbled together. i was so confident my setup was perfect, having triple-checked my nvidia driver installation and cuda versions. i configured tensorflow to use gpu ids 0 and 1, because i had verified with `nvidia-smi` they were available. however, tensorflow kept throwing this error and only using gpu 0. after hours of head-scratching, i realized my mistake: i had forgotten that for certain libraries and configurations (particularly when using virtual environments or docker), the visibility of gpus can be controlled through environment variables. it wasn’t that the gpu wasn’t there, it was that tensorflow was only permitted to see the first one.

so, lets break down why this happens and how to fix it using a stackoverflow like approach:

**common causes and solutions:**

1.  **incorrect `visible_device_list`:** this is the most straightforward reason. as the error suggests, you are explicitly telling tensorflow to use a gpu id (in your case, id `1`) but this id isn't in the set of gpus tensorflow can 'see'. this can happen if you've incorrectly set the `CUDA_VISIBLE_DEVICES` environment variable, or if you've programmatically specified a device list that doesn't match the actual available gpus.

2.  **environment variable conflicts:** the `CUDA_VISIBLE_DEVICES` environment variable is a frequent culprit. it controls which gpus are exposed to the cuda runtime (and thus, to tensorflow). if this variable is set to something other than what you expect, tensorflow will be confused. for instance, `CUDA_VISIBLE_DEVICES=0` means tensorflow will only see the first gpu (gpu id `0`), even if there are others present. this can happen with jupyter notebooks or when you launch programs inside shells with this env variable defined before.

3.  **docker container issues:** if you are running your code inside a docker container, the gpus available inside might not be the same as those available on the host machine. if you did not specify `nvidia-docker` or some similar tool to expose host gpus to the container, tensorflow may simply not see them. usually people forget to pass `--gpus all` on docker run and later are confused, and sometimes if you specified `CUDA_VISIBLE_DEVICES=0` in the host before docker launch it will also propagate inside the container so you need to unset it, otherwise, even if you expose gpus to the container, the container environment will only see `0` and if you specify in tf to use `1`, you will see the same error.

4.  **driver problems:** although less common, it's also possible that a problem exists with your nvidia drivers or cuda installation that is preventing tensorflow from properly detecting gpus. but, honestly if `nvidia-smi` is working, this is less likely.

**code examples and fixes:**

*   **checking available gpus:** let's start with a quick way to check which gpus tensorflow actually sees:

    ```python
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"physical gpus: {physical_devices}")
    visible_devices = tf.config.get_visible_devices('GPU')
    print(f"visible gpus: {visible_devices}")
    ```

    this snippet will print the physical gpus available and the gpus visible to tensorflow. note that for the second print it also shows the indexes and it is also possible that no gpus will be visible if the cuda configuration is not valid, or for the reasons specified above in docker etc. compare this output with `nvidia-smi` output. if the output is inconsistent then it's a configuration issue as discussed above.

*   **explicitly setting visible gpus:** now, let's say you want to explicitly use gpu 0 and 1 (if you have it and are visible):

    ```python
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            # set visible gpus if more than 1 available
            if len(gpus) > 1:
              tf.config.set_visible_devices(gpus[:2], 'GPU')
            else:
              tf.config.set_visible_devices(gpus[0], 'GPU')

            # verify
            visible_gpus = tf.config.get_visible_devices('GPU')
            print(f"visible gpus after setting: {visible_gpus}")
        except RuntimeError as e:
          print(f"RuntimeError: {e}")
    else:
        print("no gpus found, working on cpu only")

    # rest of your code
    ```

    this code checks if there are gpus available, and if so it explicitly set gpus 0 and 1 as the visible devices, then it re-prints them to make sure, and does nothing if no gpus are found so the code does not crash. note the `try`/`except` block is important in case the code runs into issues, so you don't miss the problem. you can modify the `[:2]` slice to select the desired number of gpus or the exact gpus you need. remember that gpu indexes start at 0. if you only have one gpu, do not select `[:2]`.

*   **setting `CUDA_VISIBLE_DEVICES` environment variable:** a solution when the above approaches do not work is to set the environment variable. if you're using a shell, you might set it before running your python script:

    ```bash
    export CUDA_VISIBLE_DEVICES=0,1 # if you want to see 2 gpus, if you want just one use 0
    python your_script.py
    ```
    or if you want to do it programmatically at the beginning of your python script, you can do the following:

    ```python
    import os
    import tensorflow as tf

    # set environment variable before doing anything else
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # or just '0' if only one gpu

    # check if env var is correct, this is only for debugging
    env_var = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES set as: {env_var}")

    # rest of your tensorflow code here
    #check gpus available
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"physical gpus: {physical_devices}")
    visible_devices = tf.config.get_visible_devices('GPU')
    print(f"visible gpus: {visible_devices}")
    ```

    this method is particularly useful for docker containers or for controlling gpu visibility on a per-process basis. note that there is an order of precedence, and this usually takes priority over the tensorflow code configuration above, but it is always a good idea to also configure it correctly in your tensorflow program as well. so use both methods together.

**recommendations for deeper understanding:**

*   **tensorflow documentation:** the official tensorflow documentation on gpus and distributed training is essential reading. i suggest specifically looking at the sections covering device placement and configurations of gpus: search for topics related to `tf.config` and specifically, `tf.config.set_visible_devices` and `tf.config.list_physical_devices`.
*   **cuda documentation:** while not directly tensorflow related, understanding how cuda exposes devices is fundamental. nvidia's official cuda documentation explains how the `CUDA_VISIBLE_DEVICES` variable and the cuda runtime work. also study the usage of `nvidia-smi` command line to understand the device structure.
*   **"deep learning with python" by françois chollet:** while not exclusively about the details of gpus, this book provides a great foundation for understanding how tensorflow works, and has a good section about distributed training. also it provides very clear examples.
*   **"programming in cuda" by michael j. flynn:** this is a very good book that is more lower level, that can help you understand in details the inner workings of cuda and how it handles gpus. it also contains tips that might help you understand potential configuration problems.

**debugging approach:**

1.  **start with basic checks:** is `nvidia-smi` working and showing the gpus correctly? is your cuda version compatible with your tensorflow version?
2.  **check the output of `tf.config.list_physical_devices('GPU')`:** does tensorflow detect your gpus at all?
3.  **check the output of `tf.config.get_visible_devices('GPU')`:** what gpus are visible to tensorflow?
4.  **check `CUDA_VISIBLE_DEVICES` environment variable:** ensure it's set correctly if you're using it, and make sure it is not interfering with the `tf.config` approach.
5.  **isolate:** if in docker, try running outside the container to isolate issues. run a simpler program first, instead of your complex models.

i know it's painful to debug these kinds of issues, but it's always a configuration problem. and the famous saying is: *"there are only 2 hard problems in computer science: cache invalidation, naming things, and off-by-one errors"* hopefully this approach helps you out. keep it techy.
