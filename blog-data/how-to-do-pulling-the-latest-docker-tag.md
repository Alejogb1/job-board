---
title: "How to do Pulling the latest docker tag?"
date: "2024-12-15"
id: "how-to-do-pulling-the-latest-docker-tag"
---

alright, so pulling the latest docker tag. that's a classic, something i've definitely spent more hours than i'd care to calculate on. it sounds straightforward, but as with most things in devops, there are a couple of wrinkles.

first off, the core issue is that "latest" is a moving target. it's not really a tag, it’s more like a symbolic link, it always points to the most recently pushed image for that repository when no specific tag was specified. this makes it really bad for production systems, or anywhere you need repeatable deployments. i learned that the hard way a few years back. i had this system that was updating daily based on the `latest` tag and one night my deployment broke at 3 am. it was like pulling teeth to get it working again. this happened because there was a new push to `latest` with an updated application, but it was different enough to cause a downstream crash on one of the systems that was using that image.

i'd been careless, thought i could get away with it but, naah, it was a painful lesson. always specify a specific tag or a sha hash. the "latest" is only good for early stages or experiments.

so, let’s talk about how you actually do it, and what to keep in mind.

the most basic way, which most probably already know, is with the docker pull command.

```bash
docker pull your_image_name:latest
```

this will indeed pull whatever image is currently tagged as `latest` on docker hub or your private registry. but if you are not careful you might run into a local cache problem if the image has not been modified and your machine will just use the existing cached version. if you really want to get the most up to date version for sure you need to add the `--no-cache` flag:

```bash
docker pull --no-cache your_image_name:latest
```

now, while this gets the job done, it leaves you with the same old `latest` problem. it doesn’t actually tell you what version you just pulled. that's not ideal, especially if you are trying to keep track of which version of your application is running in your environments.

if you want to go a step further you should use the docker inspect to find out the sha id of the image. then use this sha id for the deployment.

```bash
docker pull your_image_name:latest
docker inspect your_image_name:latest --format "{{.Id}}"
```

the output will be something like `sha256:abcdefg123456...`, and now you have a unique identifier for this particular version of the image. now you can use this id to pull the image too. you don’t need to pull again, you already have the local version, but if you want you can do this:

```bash
docker pull your_image_name@sha256:abcdefg123456...
```

this will pull the specific image version identified by that sha id, it will check if you have it locally, and use it if the sha is correct or else download it. this is way better than using `latest` for production. if you are using a git based pipeline like gitlab or github actions, they typically have functionalities to get these id’s from the container registry as part of their pipelines.

now, here is the thing, i usually don't pull the image directly on my production systems. i almost always use some sort of container orchestration tool, like kubernetes or docker swarm. in those environments, you typically configure a pod or service to use a specific tag or sha id. so you rarely will pull images directly into a production system.

instead, your pipelines will do the pulling. the pipeline will rebuild the docker image, and tag it properly, then push to your docker repository, and then update your kubernetes or docker swarm configuration to use the new image's tag or sha id. this is how you automate deployment and avoid manually dealing with images on live systems. the way i see it that’s the only correct way to deploy a production system.

another thing i always recommend is to never tag with `latest` after development unless you have very good reasons to do it. it’s a trap that has bitten a lot of people, me included. instead, use a versioning scheme that makes sense for your application, like `v1.2.3`, or commit hashes. this makes it much easier to track versions and roll back changes. always prefer an immutable identifier to track your images and use `latest` just for dev iterations.

a trick that i sometimes find useful is to use labels. you can use labels to add all kinds of extra information about your image. for example, you can add the git commit sha that the image was build from or the build number, so you know which git repo commit is using or which build trigger it.

```bash
docker build --label "git-commit=$(git rev-parse HEAD)" --label "build-number=123" -t my-image:v1.0 .
```

these labels can be checked later with docker inspect and can help in auditing and debugging if necessary. it's all metadata but metadata can be quite valuable. i’ve once spent a whole afternoon figuring out which version was running on a client's system just to discover that it wasn’t matching the build in the repo. a simple label would have avoided that long day.

about resources, for more in-depth stuff about containerizing i'd recommend the "docker deep dive" book by nigel poulton and the "kubernetes in action" book by marko luksa. also, if you are planning to use kubernetes for deployment, then reading the kubernetes documentation is crucial as its really complete and updated. these are good resources to learn how container technology works at a fundamental level.

now i'm going to take a quick break to drink a glass of water, my keyboard is getting a bit warm after this.

anyway, i hope this helps to clarify how to properly pull the latest docker tag and the problems it might introduce if not done correctly and also explain how to avoid those issues. the key takeaway here is that “latest” is really a moving target. you need to use proper tagging or sha id for anything that's even slightly important. and remember, it’s not about always pulling latest, but about pulling the image version you want with confidence.

do not hesitate to ask if you have more questions!
