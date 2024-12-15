---
title: "How to install Linux Packages in Kubernetes pods?"
date: "2024-12-15"
id: "how-to-install-linux-packages-in-kubernetes-pods"
---

alright, so you’re asking about installing linux packages *inside* kubernetes pods. this is a fairly common task, but it’s not exactly a ‘default’ kubernetes workflow, if that makes sense. let me break down my experiences, the ways i’ve tackled it, and give some examples.

the core issue is that pods in k8s are designed to be ephemeral and immutable. they spin up based on a defined image, and if that image doesn't have the packages you need pre-installed, you’re not gonna magically get them during runtime. the whole idea is that you modify the image and redeploy if you want a new setup, not change things on the fly. however there are workarounds.

firstly, let’s talk about the 'why' you'd even do this in the first place. honestly in 90% of the cases it is a bad idea and you should instead modify your container images, but i'm guessing you have that 10% case. i had a situation once, back when i was managing this really janky, old app. it depended on a proprietary library that i couldn't easily package into a docker image. so we had to install it post-deployment inside of the pod, what a nightmare. this was before we really understood the 'immutability' of containers and the problems this creates. we were younger, and less cynical about how things can go very wrong.

the standard method, the one i recommend the most, is to bake the necessary packages into your docker image during the build process, that is if it is at all possible. if not, or if you have dynamic dependencies, there are a few common strategies you can employ.

**1. using an init container:**

this is generally the preferred approach for non-dynamic dependencies. you can have an `initcontainer` that runs before your primary container starts. this init container can execute commands to install packages. this is usually the best if you must do it this way for a number of reasons. for example, you would not mix package management with running your app, which should be a golden rule, or a red flag that something could go very wrong. let me show you what that looks like:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-package-install-pod
spec:
  initContainers:
    - name: package-installer
      image: ubuntu:latest
      command: ['apt-get', 'update']
      args: ['&&', 'apt-get', 'install', '-y', 'your-package'] # change your-package
  containers:
    - name: main-app
      image: your-main-app-image:latest # your main app image
```

here, i’m using a basic `ubuntu` image for the init container and then just running the `apt-get` commands to install the needed package. just replace `your-package` with whatever package you need, and `your-main-app-image` with your app image. this method keeps the package installation isolated. once the package is installed and the `initContainer` is done, your main container can start up and have access to the new package, because it shares a common file system with the others in that pod.

**important note**: you need to pay attention to the `shareProcessNamespace` field if you want your main container to directly see the results of processes started by the init containers. normally, only file system changes are visible, not the process state.

**2. using a sidecar container:**

sidecar containers run alongside your main app container. while they’re not typically used for installing packages, they *can* be used for situations where you require a helper service or a specialized process that handles some operations *after* the main container is running. i rarely have done this because in that case i'd usually use init containers instead, however you can install packages during its setup.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-sidecar-package-pod
spec:
  containers:
    - name: main-app
      image: your-main-app-image:latest
    - name: package-installer-sidecar
      image: ubuntu:latest
      command: ['/bin/bash', '-c']
      args: ['apt-get update && apt-get install -y your-package && while true; do sleep 60; done'] # change your-package
```

the major difference is that this container runs alongside your main app. so it will always be there running, in this case doing nothing but waiting every minute, once it has done the package installation. while this is technically a working example, this method feels less 'clean' than init containers. plus you have a forever-running process eating up resources. i’d only resort to this if the dependencies are complex and dynamic during the application lifetime, or in special scenarios like log collection or monitoring setups. not my favorite, but it exists.

**3. using a custom entrypoint (discouraged):**

this is the method i used back in the day with the old janky app, and i’ll put a disclaimer on it: this is not something i recommend, it's messy, and you are much better using init containers. here, you modify your application image to include a custom entrypoint script that installs packages before it starts the actual application. let me show an example of how it looks like:

*dockerfile:*

```dockerfile
from your-main-app-image:latest

copy entrypoint.sh /

entrypoint ["/entrypoint.sh"]
```

*entrypoint.sh:*

```bash
#!/bin/bash

apt-get update && apt-get install -y your-package  # change your-package

exec "$@"
```

this works, but it intertwines package management with your app’s startup, making it less modular, harder to debug and more difficult to maintain in the future. the code above is an example that assumes your main app command is passed as arguments to the `entrypoint.sh` script, this way the original command of your app will get executed. this means you will not have to change the pod manifest. but, as i mentioned, doing this means you have to be more careful with this way of running the container.

**why are these methods a bit bad?**

well, modifying the pod itself after startup is generally against the principle of kubernetes being an infrastructure that guarantees the stability and state of the application. every time that you modify the pod after the fact, you are breaking the implicit guarantee that the infrastructure gives you of state stability and idempotence (the same state every time). that's why you should aim to build everything you need into the image, and avoid any kind of modifications after its run.

the methods i described are generally used for non-standard or complicated scenarios. if you are installing standard packages, then please, build them into the image. this way, things are much cleaner and predictable. you can get away with those methods but it will cost you a lot of debugging and headaches down the line. especially if you are new to the ecosystem.

**resources i’d suggest:**

instead of giving you specific links, i would suggest that you take a look at kubernetes documentation itself, which is fantastic. specifically, the section on pods and container lifecycle, which explains things in more details that what i can describe here. for a deeper understanding of docker and image layering, you can also refer to "docker deep dive" by nigel poulton. it explains, in simple terms, how images are structured and built, which is an essential skill when working with containers and k8s. and a last book i would suggest is "kubernetes in action" which explains in great depth the workings of kubernetes, and how pods are managed inside.

one final note, and this is my attempt to be funny, don't make your deployment process a surprise party, install packages in the build phase, or at least in an isolated init container. it's just better that way, unless, you are into some sort of troubleshooting masochism.

hope this helps, let me know if you have more specific questions.
