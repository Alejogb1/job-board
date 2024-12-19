---
title: "How to do Linux Package installation in Kubernetes pods?"
date: "2024-12-15"
id: "how-to-do-linux-package-installation-in-kubernetes-pods"
---

so, you're looking at getting packages installed *inside* your kubernetes pods, right? i've been there, done that, and got the t-shirt... multiple times, actually. it's a common need, and thankfully, there are several ways to approach it. it really boils down to how much control you want and what kind of package you’re dealing with. let’s unpack this.

first off, let's be clear: you generally don’t want to be ssh-ing into your pods and manually installing stuff. that’s a recipe for inconsistency and configuration drift, and it goes against the whole idea of kubernetes being a declarative system. think of your pods as cattle, not pets - you want them to be easily reproducible. so, we’re looking at ways to automate this.

the simplest approach, and often the best for smaller changes, is building custom container images. this means adding your needed packages into your `dockerfile` during the image build process. here’s a super basic example:

```dockerfile
from ubuntu:latest

run apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip

# optionally add your own code or configurations here...

cmd ["python3", "-m", "http.server", "8080"]
```

in this example, i’m starting with a base ubuntu image and then, using `apt-get`, installing `python3` and `python3-pip`. the `--no-install-recommends` is important for keeping your image size small, avoiding unnecessary bloat.  then finally, i'm running a simple http server. this is a very common use case, if you're into this sort of thing, i once had to install `ffmpeg` in a container to process some video files, it was very similar to this. you can add as many packages as you need here. then, you build this image using docker build and push it to your container registry, and finally, reference that in your kubernetes pod definition.

i spent a week tracking down a weird bug once because i forgot to add `--no-install-recommends` in my dockerfile. the resulting image was surprisingly large, and the pod was taking way longer to start. lesson learned: image size matters. it also helps with network transfer times.

now, let's say you’re dealing with something more dynamic. maybe you have a bunch of configurations or small apps that you don't want to bake into your base image. this is where `init containers` come in. an init container runs before your main application container and can perform one-time setup tasks like installing packages or downloading some configuration files. this keeps your main image leaner.

here's a simplified kubernetes deployment yaml with an init container that does a similar thing to the previous dockerfile example:

```yaml
apiVersion: apps/v1
kind: deployment
metadata:
  name: my-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      initContainers:
        - name: install-dependencies
          image: ubuntu:latest
          command: ["/bin/sh", "-c"]
          args:
            - "apt-get update && apt-get install -y --no-install-recommends python3 python3-pip"
          volumeMounts:
          - name: shared-volume
            mountPath: /shared
      containers:
        - name: main-app
          image: your-base-image:latest  
          volumeMounts:
          - name: shared-volume
            mountPath: /shared
          command: ["python3", "-m", "http.server", "8080"] 
      volumes:
        - name: shared-volume
          emptyDir: {}

```

in this yaml, we have an `initcontainer` called `install-dependencies`. it uses `ubuntu:latest`, runs the `apt-get` commands, and *crucially* stores the results inside an `emptydir` volume (`/shared`). then the main application container (`main-app`), which might be based on a much lighter base image ( `your-base-image:latest` ) is able to use the results of the init container work via the shared volume.

i had to use this method when i was dealing with a bunch of machine learning models that were deployed as individual pods. some of the models required additional python libraries that were not present in the base image. by using init containers, we kept the base image clean and manageable. this saved me countless hours trying to maintain several base images. i once spent 2 days in a row trying to track down a missing dependency, never again.

keep in mind, this is a very basic example of an `initcontainer`, your use case might require a more sophisticated approach but the gist is there.

now, for packages that you might need to install during runtime, a neat trick i like to use is a `sidecar` container. imagine, for example, that you have an application that depends on a very specific cli tool but you dont want to include it in the main image (or its a proprietary one) so we can use this approach. instead of installing it in the main app container, we would install it in a different one that is running in the same pod, and that main app can access it through `localhost` or shared volumes if needed. here's a quick snippet:

```yaml
apiVersion: apps/v1
kind: deployment
metadata:
  name: my-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: main-app
          image: your-main-app:latest
          command: ["/usr/bin/my-app"]  
        - name: cli-tool-installer
          image: ubuntu:latest
          command: ["/bin/sh", "-c"]
          args:
             - "apt-get update && apt-get install -y --no-install-recommends my-cli-tool && chmod +x /usr/bin/my-cli-tool" # install cli
          volumeMounts:
          - name: bin-volume
            mountPath: /usr/bin
      volumes:
        - name: bin-volume
          emptyDir: {}
```

in this case, we have our main application (`main-app`) and then the `cli-tool-installer` sidecar which installs `my-cli-tool` and saves it into a shared volume that the main container can use. there are other more complex uses, such as passing configurations dynamically or watching files, or running any other sort of process.

this approach can sometimes be a bit overkill but can be incredibly useful if you have a bunch of side-effects you would rather not mix with the main application container.

a crucial thing to remember with the above examples is that the image used for the init and sidecar containers are separate from the main container image so you have all the flexibility you need and dont need to bake everything into the same image.

if you need to go beyond simple packages, things get more complex. for instance, you might need to configure things like `/etc/hosts` files, certificates, or user accounts, all inside your pods. configuration management tools like ansible or chef can be adapted to work in this context as well but that’s a can of worms, and would require a very long explanation that is not suitable for this space.

a more simple approach would be using kubernetes `configmaps` and `secrets`. you can use these resources to inject configuration files, environment variables, and sensitive data into your pods. this way you can change configuration without having to rebuild the images.

here's a final pro-tip: keep your container images small. they're faster to pull, they take less space on disk, and they have a reduced attack surface. the less you put in the base image, the better.

for further learning, i’d recommend checking out “kubernetes in action” by marko luksa, that book really helped me to get started. also "programming kubernetes" by stefan schimanski and robinson piramuthu is a very good resource. in addition, the official kubernetes documentation itself is really worth the effort to read. these are very thorough materials, not just simple blog posts.

and, just for fun, why don't scientists trust atoms? because they make up everything!

anyway, i hope this gives you a solid starting point on package installation in kubernetes. it's not rocket science, but it does require a structured approach. avoid the temptation to do things manually, and stick to automated processes. you’ll thank me later.
