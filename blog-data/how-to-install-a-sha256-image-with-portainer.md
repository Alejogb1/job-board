---
title: "How to install a sha256 image with portainer?"
date: "2024-12-15"
id: "how-to-install-a-sha256-image-with-portainer"
---

alright, so you're trying to deploy a container image using its sha256 digest with portainer, and you're running into some trouble, eh? i get it. been there, done that, got the t-shirt… and probably a few error logs burned into my brain. it's one of those things that *should* be straightforward, but can trip you up if you don't know the subtle dance steps. let me share my experience with this.

i remember way back, when docker was still kind of finding its legs, i spent a whole weekend chasing my tail trying to figure out why one of my builds wasn't behaving like the previous version. turned out, i had been pulling images by tag, which had been updated underneath me. lesson learned: always, always, always use digests for anything production-related. it eliminates a bunch of the "it worked yesterday!" type of head-scratching.

portainer, for all its goodness in making docker management more accessible, does have some quirks in how it handles image digests, particularly when you're not using a public registry like docker hub. the ui is helpful for day-to-day stuff, but sometimes it can get in the way when you need to drop down to the nuts and bolts. you often need to go through the api or the cli because of it. i like the fact that it centralizes access to my different docker servers, but not so much when i need to get specific with my configuration.

when pulling an image by sha256, you're basically telling docker, "i want *this* exact version of the image, no ifs, ands, or buts". it’s the fingerprint of the image and that can avoid a lot of issues that tags sometimes introduce due to its mutability. you’re avoiding the issue of potentially getting something different when you expect something specific.

so, here's the rundown of the way i've made this work consistently, broken into a couple of approaches, both from a cli way because i find that more reliable in my day-to-day.

**approach 1: using `docker run` with a pre-existing network**

this is probably the quickest way if you're comfortable with the docker cli and just want to get things up and running fast, skipping the portainer middleman for a moment. if you already have a network you want your container to use, this should be your first pick. here’s the basic structure:

```bash
docker run -d \
    --name my-container \
    --net my-network \
    -p 8080:80 \
    sha256:your-sha256-digest@your-registry/your-image
```

*   `-d`: runs the container in detached mode.
*   `--name my-container`: sets the name of your container. feel free to change this to something more descriptive
*   `--net my-network`: specifies that you want this container on a given network
*   `-p 8080:80`: maps port 80 of the container to port 8080 on your host. this part depends on your needs, feel free to change accordingly
*   `sha256:your-sha256-digest@your-registry/your-image`: is the most important, it's the full image digest with its registry. so for instance it would be something like `sha256:abcdef1234567890abcdef1234567890@my-private-registry.com/my-image`

the key here is the full image name: `sha256:digest@registry/image`. make sure that you include the `sha256:` part and the full path to the image. if it's docker hub, the registry can be omitted, but in my experience, it's better to be explicit about the registry.

**approach 2: using `docker create` then `docker start`**

this is the way if you need to do more config before starting the container or maybe need to use docker compose to do a more complex setup:

```bash
docker create \
    --name my-container \
    --net my-network \
    -p 8080:80 \
    sha256:your-sha256-digest@your-registry/your-image
```
after this, you'd run:
```bash
docker start my-container
```

what's different here? instead of directly running the image, we create the container first. this can be helpful if you need to add additional configurations or if you need to do this in a more programmatic way using docker compose for instance.

**approach 3: using portainer cli**

portainer's cli isn't as widely advertised, but sometimes it's the easiest way to work with things. so this will be like approach 1, but doing this programmatically through portainer using its built in cli:
```bash
portainer container create \
  --name my-container \
  --network my-network \
  -p 8080:80 \
  sha256:your-sha256-digest@your-registry/your-image
```
then you have to start the container like so
```bash
portainer container start my-container
```

you must have the portainer-cli already set up, this way is useful to make programmatic deployments, and i find this more convenient when working with remote servers that have limited shell access and you want to avoid going to the web ui because its slow or laggy. it’s also convenient for doing batch deployments from scripts. this approach is not very different from approach number 2 but it uses the portainer built in api.

**common pitfalls to avoid:**

*   **incorrect sha256 digest:** double check that the digest you're using is the *exact* one you intended to use. a single typo and you might end up with a completely different image (or no image at all, which is better than the wrong image)
*   **missing registry:** make sure you include the registry url if it's not docker hub.
*   **network issues:** containers not being in the same network is a classic.
*  **port conflicts:** check if there is not any other service that already uses that port, specially if you are using standard ports like 80, 443, or 8080. sometimes you can be banging your head against the wall trying to debug it when all you need to do is change the port number of your container.

now, as far as resources, i'd recommend you to skip the generic google search and take a look into some of the more canonical materials out there:

1.  "docker deep dive" by nigel poulton. a very solid book to go deep into docker.
2.  the docker official docs, of course. they're actually pretty good now. not like back in my early docker days.
3.  various articles about docker image building, focusing on the immutability of digests and the importance of knowing how to generate them correctly. check online the kubernetes documentation on image pull policy as well.

and here’s the joke i promised: why was the docker image so good at keeping secrets? because it had really good layers! haha… i know, bad one.

anyway, i hope this sheds some light on the issue. the main thing is to always double check your digests, registries, and network configurations. once you get the hang of this, you'll be deploying containers with a sha256 like a pro. if you've got any more issues, feel free to ping. we've all been there, so no worries.
