---
title: "Is there a way to use a python debugger in airflow dags?"
date: "2024-12-15"
id: "is-there-a-way-to-use-a-python-debugger-in-airflow-dags"
---

yes, absolutely. it's not always straightforward, and i've definitely banged my head against the wall on this one more than a few times, but debugging airflow dags with a python debugger is doable. the key is understanding where your code actually runs and how airflow's execution environment is set up.

first off, forget about trying to just drop in a `pdb.set_trace()` at the top of your dag file and hoping for the best. that's a rookie mistake, and trust me, i've made it. that code will execute, yeah, but it’ll be in the airflow scheduler process, not the process that's actually executing your tasks. that’s a lesson i learned the hard way when i spent hours wondering why my breakpoints weren't getting hit. my first serious foray into airflow was back in 2018, working for a small startup that was trying to do some pretty serious data transformations. we were running our airflow instance in a custom docker container on an aws ecs cluster, a real pain in the neck. i had this monstrously complex dag, it was literally a spaghetti of operators and dependencies. and i kept hitting these really peculiar bugs that were just not making sense. nothing was breaking spectacularly, things were simply outputting incorrect results.

so, the general rule is, you need to put your breakpoints in the *actual* task code, meaning the python callable within your airflow operators. that’s where the work is being done and where you need to debug. 

the most basic way to debug is using `pdb` directly, which ships with python. this works well when you can run your tasks locally.  here's how that would look in a simplified airflow python operator example:

```python
from airflow.decorators import task
import pdb

@task
def my_task(some_param):
    result = some_calculation(some_param)
    pdb.set_trace()
    print(f"the result was {result}")
    return result

def some_calculation(param):
    #this is where all the magic happens
    return param * 2 # simplified magic for example purposes
```

now, if you trigger this task from your airflow ui or via command line, and the executor you use is the localexecutor, you’ll drop into the pdb debugger when `pdb.set_trace()` is reached. you can inspect variables, step through your code, everything that the `pdb` offers. be aware you might have to do some `import` statements within the pdb debugger as it can be in a different python virtual environment than the one where you run airflow locally.

this is great for local development and testing but not so handy when airflow is running remotely, which is, let's face it, most of the time. that setup is where it starts to get trickier and requires a little more tooling. for remote debugging, i find the best approach is to use a remote debugger like `debugpy`, which is a module developed by microsoft specifically for python. you need to install it in your environment, usually done with `pip install debugpy`.

the idea is, you inject a debugpy hook at the beginning of your task’s code and then you can attach a debugger like vscode or pycharm remotely to that hook. here's a sample of code injection using `debugpy`:

```python
from airflow.decorators import task
import debugpy
import time

@task
def my_remote_task(some_param):
    debugpy.listen(("0.0.0.0", 5678))
    print("waiting for debugger to attach")
    debugpy.wait_for_client()
    print("debugger attached, continuing")
    
    result = some_remote_calculation(some_param)
    print(f"remote result was {result}")
    return result

def some_remote_calculation(param):
    #this is where all the magic happens
    time.sleep(2) # just to simulate a remote process running for 2 seconds
    return param * 3
```

the above code makes your task hang and await until you connect your vscode debugger to it. be aware that you need to open port 5678 (or whatever port you configure) in your firewall. you'll then configure your ide to attach to that ip and port combination.

now there’s a catch, or two actually. first, you need to make sure debugpy is installed in the python environment where your airflow tasks run. this usually involves baking that into your docker image or making it available to your worker nodes. and second, your network needs to be configured to allow your local machine to communicate with the remote process on the specified port. this can be painful in a complex networking setup. when i was working on that startup i mentioned, our airflow cluster was behind a vpc, and i had to set up port forwarding to be able to debug tasks.

a slightly more sophisticated method that avoids the potential firewall configuration hassle, is to use the `debugpy.connect` method, where the remote task starts a connection to a pre-running listener in your ide, but that has additional complexity related to your ide and debugger support, and it’s not always ideal for cloud-based environments where you don't have a fixed ip.

also, a word of caution, when debugging remote systems always think of your security, using port forwarding or listening with open interfaces is an open door for possible security attacks so take care of your network security when debugging remotely.

a better approach, in my experience, when dealing with larger more complex setups, is to make sure your functions are easily testable in isolation before they are integrated into the airflow pipeline. for example, writing unit tests that you can run in an ide or the terminal, will save you a lot of debugging hours.

i learned this particular lesson when, once i had all the debugging setup and everything working, i realized i spent way too much time staring at breakpoints instead of writing well-isolated testable code in the first place. that's when i started thinking, maybe i should read a good book on software architecture, perhaps something like "clean architecture" by robert c. martin. i also found that “test-driven development by example” by kent beck also helped me significantly to move to this approach. it was a hard lesson for me since i started on the "test after the code is written" approach.

the key is that by running your code outside of the airflow environment, you can debug issues faster without all the overhead of setting up a debugger and waiting for an airflow task to be scheduled.

a less popular but interesting method is to use python logging module to extensively log the execution path and variable values. that will give you some perspective on your code behavior without needing to halt the execution, like in the breakpoint methods. this approach is less interactive but very handy for production environments, or even to debug in test environments. this approach requires that you configure your logging level, that you log only necessary information and do not overwhelm the logs.

 here's an example of logging used within a python task operator:
```python
from airflow.decorators import task
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def my_logged_task(some_param):
   logger.info("starting my_logged_task")
   random_val= random.randint(0,1000)
   logger.info(f"random number: {random_val}")
   if some_param % 2 == 0:
     result = some_logged_calculation(some_param, random_val)
     logger.info(f"result: {result}")
     return result
   else:
      logger.warning(f"skipping because of even parameter number")
      return None

def some_logged_calculation(param, rand):
  #this is where all the magic happens
  logger.info("starting some_logged_calculation")
  intermediate= param * 2 + rand
  logger.info(f"intermediate value: {intermediate}")
  return intermediate * 3
```
the code above will log to the console some informative data about the execution path and the variable values, when using the local executor you can find those logs in the console. but when used in a real cluster you can find those logs in the airflow web ui in the task log tab. this is not as interactive as a debugger but is faster and more lightweight, and it works on real production systems.

in summary, yes you can debug airflow dags with python debuggers. the best approach depends heavily on your setup and what exactly you're trying to accomplish. if you're just developing locally, `pdb` will get you a long way. for more complex remote or cloud systems, a solution with debugpy and using the correct debugging tools of your IDE can be very helpful, also remember to use isolated, testable functions to prevent further debugging sessions and to be more productive. also make sure that you log as much as you need to diagnose the issues and that you do not overwhelm the system with unnecessary logs that will only slow it down.

and one last thing, make sure to take regular breaks during debugging, because prolonged exposure to code can lead to bugs. it is a very well known scientific fact, that if you look at your code for too long it will produce even more issues... the real trick is to look at someone else's code and see if you can spot the bugs there instead, and then apply that knowledge to yours... haha just kidding... or am i?
