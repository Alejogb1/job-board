---
title: "How to push a lando image to a Github container repository?"
date: "2024-12-14"
id: "how-to-push-a-lando-image-to-a-github-container-repository"
---

alright, so you're looking to push a lando-built docker image to a github container registry. yeah, i’ve been there, done that, got the t-shirt – and probably a few error messages along the way. it's a pretty common workflow these days, and honestly, it can feel a little convoluted the first couple of times. let's break it down, though. i’ve had my fair share of head-scratching moments with this, so i hope my experience helps smooth things out for you.

first things first, lando makes building your docker images pretty straightforward, right? but the 'pushing' part, that's where the github container registry comes into play, and it's got its own quirks. before jumping into the nitty-gritty, let's make sure our foundations are solid.

you need to be logged into your github account both locally and have the right permissions for your repository and container registry. for local login, you usually do a `docker login ghcr.io` and it will prompt you for your github username and a personal access token (pat). the pat must have the correct permissions for writing packages (which is how github handles container registries). be careful not to use your regular password. generate a specific token for this. if you haven’t done this step, go do it before we continue. failing that will be, well, anticlimactic.

now, let’s assume you've got your lando setup, and you've run something like `lando build` successfully, which has created a docker image. the tricky part is finding the correct image id, and then tagging it so ghcr can understand where it goes. this where i tripped up the first time, believe me, it took me a while to realize the image i thought i was pushing wasn't the one lando actually built.

here’s a little bash script i use sometimes, i find it useful. it grabs the image name created by lando, tags it, and then pushes it to the ghcr registry:

```bash
#!/bin/bash

# assuming you are in the lando project root

# get the image name of the application container
image_name=$(lando info | grep -oP 'appserver: \K(.*)')

# this is where the github registry details are set.
# change `your_github_username` and `your_repo_name` accordingly
ghcr_image_tag="ghcr.io/your_github_username/your_repo_name:${LANDO_APP_NAME}-$(date +%Y%m%d%H%M%S)"

# tag the image with the ghcr details
docker tag "$image_name" "$ghcr_image_tag"

# push the image
docker push "$ghcr_image_tag"

echo "successfully pushed image $ghcr_image_tag"
```

i remember, the first time i did this, i forgot to set the `LANDO_APP_NAME` variable, leading to an ugly tag name with no context. those were the days… debugging random tag names was not my favorite pastime.

let's break down what's happening in this script:

*   `lando info | grep -oP 'appserver: \K(.*)'`: this line is extracting the docker image name that lando has built. it looks for the line starting with `appserver:` in the output of `lando info`, and then extracts everything after it, using grep with perl regex, `oP` flags are powerful.
*   `ghcr_image_tag="ghcr.io/your_github_username/your_repo_name:${LANDO_APP_NAME}-$(date +%Y%m%d%H%M%S)"`: this is where you craft the full ghcr image tag. you absolutely must replace `your_github_username` and `your_repo_name` with your actual github username and repository name.  the `$(date +%Y%m%d%H%M%S)` is simply adding a timestamp for keeping a history of pushed images, which is quite handy in a team environment, believe me.
*   `docker tag "$image_name" "$ghcr_image_tag"`: this tags your existing lando built image with the correct ghcr tag. remember docker tagging does not copy the image. it adds an alias.
*   `docker push "$ghcr_image_tag"`: finally, this pushes the tagged image to the github container registry.

now, another problem i had was dealing with a specific lando setup which had a multi-stage docker build, it wasn't as simple as grabbing the first image id. if this is your case then you will need to modify the script to use a different container name or a container from a stage within the build. i'll give you another little snippet. imagine you have a `builder` stage and an `app` stage in your dockerfile, you'd probably want the latter, not the former. in that case, you might need to use `docker inspect` to get the image id, let's say the dockerfile has the final stage named `app`, and you are interested on getting the image id from that:

```bash
#!/bin/bash

# assuming you are in the lando project root

# get the image id of the app stage
image_id=$(docker inspect $(lando info | grep -oP 'appserver: \K(.*)') --format='{{.ContainerConfig.Image}}')

# this is where the github registry details are set.
# change `your_github_username` and `your_repo_name` accordingly
ghcr_image_tag="ghcr.io/your_github_username/your_repo_name:${LANDO_APP_NAME}-$(date +%Y%m%d%H%M%S)"

# tag the image with the ghcr details
docker tag "$image_id" "$ghcr_image_tag"

# push the image
docker push "$ghcr_image_tag"

echo "successfully pushed image $ghcr_image_tag"

```

this changes the line where we grab the image to be pushed. instead of getting the `appserver`, we get the `containerconfig.image` property from the inspection of the lando built image.

the main trick with all this is getting the correct image id or hash. depending on how your lando recipe is configured, your mileage may vary. you might need to inspect and trace things to find out what's happening, and then adjust accordingly.

also, make sure you are using a correct version of docker and lando, there were problems in the past with earlier versions not dealing very well with this.

another thing, be mindful of your internet connection when pushing large images. i’ve had a push fail mid-way due to flaky wifi (yes, i know, the shame). it’s worth verifying the final pushed image once it’s up there.

one last piece of advice, if you are working in a team, using github actions for this sort of thing makes life easier (and way more automated). you could set up a workflow that builds and pushes the image when you merge code into the `main` branch, for instance. this is usually how we do it. there’s no need to have manual steps when you can automate them. here is an example of how it would look like in a github action:

```yaml
name: docker push

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: checkout code
        uses: actions/checkout@v4

      - name: setup lando
        uses: lando/github-action@v1

      - name: get image name
        id: get-image-name
        run: echo "image_name=$(lando info | grep -oP 'appserver: \K(.*)')" >> $GITHUB_OUTPUT

      - name: set ghcr tag
        id: set-ghcr-tag
        run: echo "ghcr_image_tag=ghcr.io/${{github.repository_owner}}/${{ github.event.repository.name }}:${LANDO_APP_NAME}-${GITHUB_SHA::7}" >> $GITHUB_OUTPUT
        env:
          LANDO_APP_NAME: myapp

      - name: docker login
        run: docker login ghcr.io -u ${{github.actor}} -p ${{secrets.GITHUB_TOKEN}}

      - name: tag and push
        run: |
           docker tag ${{steps.get-image-name.outputs.image_name}} ${{steps.set-ghcr-tag.outputs.ghcr_image_tag}}
           docker push ${{steps.set-ghcr-tag.outputs.ghcr_image_tag}}
```

this is a very simple workflow, using the lando github action which will make sure lando is set up before we start using it. then it gets the image name, creates a tag for ghcr, and logs in to github container registry and then pushes it.

for deeper dives into the specifics of docker, i strongly recommend "docker deep dive" by nigel poulton, it is a classic and it explains all the internal working of docker which can be really useful, and "programming docker" by jaffar ali siddiqui. on the lando side there isn’t a specific book that i know, but their documentation is pretty good if you need to find out specific details. for github actions, again, the documentation is the best place to start.

so, there you have it. pushing a lando-built docker image to github container registry isn’t that hard once you've tackled the basics. it's all about understanding how lando manages its containers, crafting the correct tags, and making sure you have the correct authentication set up.

i hope that helps and happy coding!
