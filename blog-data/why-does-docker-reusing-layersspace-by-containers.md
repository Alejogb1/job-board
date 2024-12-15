---
title: "Why does Docker: Reusing Layers/Space by Containers?"
date: "2024-12-15"
id: "why-does-docker-reusing-layersspace-by-containers"
---

alright, let’s talk about docker and its layer caching mechanisms. it's a pretty fundamental concept, and if you’ve been working with docker for a while, it’s probably something you’ve taken for granted, but it’s worth breaking down. at its core, docker's image layering is all about efficiency, both in terms of storage and build speed. 

let's start with the storage. think of a docker image not as a single monolithic file, but as a series of read-only layers stacked on top of each other. each layer represents a change, a step in the instructions from your `dockerfile`. so, if you start with a base image like ubuntu, that's your first layer. then, maybe you install some dependencies using `apt-get`. that install action will create a new layer. every `copy`, `add`, `run` command, all these guys result in a new layer. this is important, because each of these is cached.

now, here’s where the reuse part comes in. when you build a new image, docker checks if any of the steps in your dockerfile are identical to steps in previously built images. if it finds a match, it simply reuses that layer instead of rebuilding it. imagine you have 10 different dockerfiles, all starting with the same base ubuntu image and installing a set of common python libraries and you are developing in a project with many small microservices, all based in python. instead of downloading and installing that base ubuntu and all the libraries 10 times docker only does it once. the other nine images use references to the already existent cached layers. this makes sense, doesn't it? it’s why subsequent builds can be so much faster. the same happens when containers are built, docker can reuse the layers from the image.

it’s a similar thing with containers. when you run a container, a thin, writable layer is created on top of the image layers. this is where any modifications made by the running container go, like files you create or changes to existing ones inside the container. when you stop and restart that container or create a new one from that same image, it will reuse the same read-only layers of the image, and create new thin read-write container layers. this way of reuse is one of the main reasons why docker can run so quickly and create containers in seconds. this reuse extends to other things such as volume mounts too.

when i first started using docker, this was not obvious. i remember working on a project where i was making tiny changes to a large application, each time i was building a new docker image it took ages, because i was always doing a clean build. after a bit of trial and error, and tons of reading about docker images, and layers, i learned how to order the steps in my dockerfiles. specifically, steps that rarely changed, at the beginning. and the ones that changed a lot, towards the end, that simple reordering was a game-changer in that project. it reduced build times from several minutes to seconds. for example moving install dependencies step to the top, because they do not usually change that much.

here's a little practical example in a dockerfile:

```dockerfile
from ubuntu:latest

run apt-get update && apt-get install -y python3 python3-pip

copy requirements.txt .
run pip3 install -r requirements.txt

copy . .
run python3 app.py
```

in this dockerfile, the `from ubuntu` and `apt-get` steps will be cached. as long as you don’t change the ubuntu base image or add anything to the `apt-get` install. similarly, the `pip3 install` step will be cached. if you change your requirements, only that layer will be rebuilt, the previous layers are reused. the last two steps `copy . .` and `run python3 app.py` will be rebuilt on every change in the source code of the project. if your `requirements.txt` file stays the same and your dependencies do not need updating, the next time you build you will see that the install layer will be taken from cache.

now, here's a different, perhaps less ideal, way of building the same image:

```dockerfile
from ubuntu:latest

copy requirements.txt .
run apt-get update && apt-get install -y python3 python3-pip && pip3 install -r requirements.txt

copy . .
run python3 app.py
```

here, we moved the copy command before the `apt-get` and `pip3` instructions. the issue with that is that on every change you make in your `requirements.txt` file, because the copy instruction is before the other ones, the whole layer that comes after the copy will be invalidated, and that means that docker will need to reinstall all the packages all over again. this shows how important is to carefully decide which steps are first in the dockerfile to take full advantage of caching.

let me give you one more example. imagine you have a simple nodejs application:

```dockerfile
from node:16

workdir /app

copy package*.json ./

run npm install

copy . .

cmd ["npm", "start"]
```

in this `dockerfile` we copy the package files and then we do `npm install` so when we change the source code of the application, only the last copy and cmd instructions are recalculated and npm is not install it again.

so in summary, the key takeaway is that docker caches layers based on the commands in your dockerfile, when a step does not change, its associated layer is used from the cache, instead of being recalculated. the most important rule of thumb is that the layers that change more frequently should be placed lower in your dockerfile so you can maximize layer reuse.

now, i’ve seen developers get into issues with caching. sometimes, people try to change stuff in a layer that is being cached, and then wonder why it’s not working. or even worst some people try to clean or delete stuff that was put in a layer that is being cached, that is not how it works. the layers are read-only. sometimes, you want to force a rebuild, but you do not know how, and you are having a bad day, when you could have been working on more important things (this is a joke). for these cases docker has an option to avoid using cache during builds.

there are times when this layer caching can cause unexpected results if you are not aware of how it works. you might change a dependency or configuration and then not see the changes when building. sometimes you have to clean cache, or better yet, use the option `--no-cache` when you are building your image. it forces docker to recalculate all the steps instead of using cache layers.

in general, layer caching is crucial for any project using docker, and it is important to understand how it works. it's not only a nice-to-have, it’s a fundamental aspect of the platform and it really reduces build times and disk usage.

if you want to dive deeper into this, i would recommend looking at the official docker documentation. specifically, the section on understanding images and layers is where you will find all the nitty gritty details. also, the book “docker deep dive” by nigel poulton is a good read, if you want to get a full grasp of docker concepts. also, the research paper “the docker runtime” has a great in-depth description of how images and containers work.
