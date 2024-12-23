---
title: "How to do Docker Custom Entrypoint While Still Debugging?"
date: "2024-12-15"
id: "how-to-do-docker-custom-entrypoint-while-still-debugging"
---

so, you're looking to hack around docker entrypoints and still have your debugger work, right? been there. let's unpack this. it's a common headache, and there isn't a single silver bullet, more of a collection of practical techniques that depend a little on your specific situation.

the core problem is that your `entrypoint` script in docker replaces the usual command-line invocation that your debugger is expecting. the debugger needs to attach to a process. when the entrypoint runs its thing it can spawn subprocesses, or change how the python interpreter is invoked, so the debugger can't just attach to it. the `docker run` command, which usually starts your application, is bypassed. the `entrypoint` script, you set in your dockerfile, takes over. that’s the catch.

i've personally spent countless hours staring at seemingly random error messages because of this. there was this one project where the `entrypoint` was a tangled mess of bash commands that not even the original developer could explain fully (we all have those stories, i assume). i had to basically reverse engineer the darn thing to figure out why my debug session wasn't connecting to the correct process. it felt like finding the proverbial needle in a haystack... made of needles.

first off, a very basic but often missed trick: simplify the `entrypoint`. instead of having a giant, complex shell script, try to make it as minimal as possible. ideally, it should just set up a few environment variables and then call your main application script. the cleaner the `entrypoint`, the easier it is to reason about.

example: if your dockerfile has:

```dockerfile
from python:3.9-slim

# ... other instructions...

copy ./my_app /app
workdir /app

entrypoint ["/app/startup.sh"]
```

your `startup.sh` might look like this mess:

```bash
#!/bin/bash
set -e

echo "setting up stuff..."
# some complex initialization here
some-complex-setup-command --option1 --option2 value 
another-command
export ENV_VAR="value"
echo "starting the app..."
exec python my_app.py
```

this is just a recipe for pain. instead, try the following. in your dockerfile:

```dockerfile
from python:3.9-slim

# ... other instructions...

copy ./my_app /app
workdir /app

entrypoint ["python", "my_app.py"]
```

and then move the initialization logic to a python script that gets executed before the main app:

```python
# /app/setup.py
import os
import subprocess

def run_setup():
    print("Setting up things")
    subprocess.run(["some-complex-setup-command", "--option1", "--option2", "value"], check=True)
    subprocess.run(["another-command"], check=True)
    os.environ["ENV_VAR"] = "value"
    print("Setup complete")

if __name__ == "__main__":
    run_setup()

```

and make my_app.py call setup.py

```python
# my_app.py
import setup
if __name__ == "__main__":
    setup.run_setup()
    print("starting my actual application")
    # main application logic here
```
this is a much simpler setup.
with this structure your debugger can simply attach to the python process in `my_app.py`, and you control the initialization steps in your `setup.py` script. this might sound almost too simple to help but it really does most of the time.

another helpful strategy, especially with more complex apps or if you just don't want to make changes to the application code, is to use an alternative `entrypoint` when debugging. you can accomplish this with the `docker run` command overriding the entrypoint value from the dockerfile. that is handy when you need to preserve the entrypoint logic, especially when you work with more than one debugger.
example:

```bash
docker run --rm -it  --entrypoint python my_image my_app.py
```

this bypasses your defined `entrypoint` and directly starts your application, so your debugger attaches directly to it. remember to remove the `-d` (detached mode) flag if using a debugger from an ide, as it needs the container to be running interactively. the `--rm` flag makes docker remove the container after you stop it, avoids clogging up your system.
remember you can use the `-e DEBUG=true` for some control of what the application does under debug, you can have a condition on `if os.environ.get('DEBUG')` in your application code.

now, if you’re using more advanced tools like pdb or other debuggers that require network connections, things get a little more involved. you will have to open the proper ports. when you debug using your ide debugger, like in vscode or pycharm it can use a remote debugger server that connects to your app through the debugger's defined port. you'll have to expose these ports in your dockerfile and map them when you run the container. for instance:

```dockerfile
# ... other stuff

expose 5678 # port for pycharm remote debugger
```

and then, when running:

```bash
docker run -p 5678:5678 --rm -it my_image
```

this maps the docker container's port 5678 to your local machine. the debugger can connect to this port and control your application while it runs in the docker container. this is not that different from running a debug session in any remote host, only difference you need to build your image beforehand and run it. that said, this approach has allowed me to debug fairly complex microservice architectures where you run multiple services inside docker containers while debugging. there was one service where the issue was in how it handled concurrency so using a debugger in docker to debug the specific timing issues was very important, it also had many configuration flags that we did not fully understand the implications of, the debugger allowed us to easily try the different settings and see the results in real time.

i've found that a good strategy is to use a combination of these methods depending on the scenario. simplify your `entrypoint` where possible, use alternative entrypoints for simple debug sessions and always use exposed ports if using a remote debugger.

if you’re looking for more depth on this, i recommend reading more about docker internals and the way process management works in linux containers in general. the docker documentation is , but a deeper dive into containerization concepts really helps understand the limitations and capabilities of the tech. reading academic papers on containerization can be very useful for that, specifically papers on cgroups and namespaces. i am a big fan of os books, particularly ones that cover how process management and file systems work. i have this one in my library *'operating system concepts by silberschatz, galvin and gagne'* which is not specific to containers but helped me understand how processes work, which has been very helpful. also any good advanced linux user guide can give you the basis of what is going on under the hood.

debugging docker containers, as is often the case with technical problems, takes a lot of practice, so keep at it. and remember the only people that don't make mistakes are those who don't touch code. so keep practicing and happy debugging.
