---
title: "With Github Actions, why is an image pulled even if a step is skipped?"
date: "2024-12-15"
id: "with-github-actions-why-is-an-image-pulled-even-if-a-step-is-skipped"
---

alright, so you're seeing github actions pull images even when a step that uses that image is skipped. i've definitely banged my head against this one before, it's a sneaky behavior of how github actions deals with containers and step execution. let me unpack this from my perspective, having spent more late nights than i care to remember debugging these things.

first off, let's understand that github actions runs your workflow on virtual machines, essentially ephemeral compute instances. each job within a workflow gets its own clean vm. the runner, which is the software that actually executes your workflow definition, handles the setup and teardown, including container image management.

now, when you define a step using the `uses:` or `container:` keywords, you're telling the runner to use a specific docker image, right? the runner then needs to ensure that the image is available before it executes that step. github actions optimization process usually downloads the container image when it reaches a step that needs a container image not only when the step is about to be executed. so even if a step is skipped because a conditional `if:` evaluates to false, the runner might have already pulled the image. think of it as pre-fetching or pre-loading. it does this so that if that step were to be executed it doesn't need to download anything and it could execute much faster.

i’ve seen this go wrong in a big way. once, i had a complex workflow where a step using a large image was conditionally executed based on the git branch. on a development branch, where this step was skipped, it was downloading the image anyway. the build was taking more time. at first, i thought i had a misconfigured conditional. i had a look at the github actions runner logs, then i realize what it was happening. it was pulling the image before the `if:` condition was even considered and the workflow execution time was dramatically affected. it was like the runner was a kid in a candy store, grabbing everything, just in case, before figuring out what to buy.

here's a basic example of how this can manifest:

```yaml
name: image_pull_on_skip

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: conditional step
        if: false # skip this step
        uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: some-image:latest
```

in this example, even though the 'conditional step' step is skipped, the `docker/build-push-action@v4` image is likely to be pulled by the runner before the step can be skipped. it can be a good idea to check the github action runner logs if the images are being pulled. look for lines that are related to docker image pull.

this also happens with `container:` steps too:

```yaml
name: container_image_pull_on_skip

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: conditional container step
        if: false # skip this step
        container:
          image: some-image:latest
        run: echo "i should be skipped"
```

similar to the previous case, the `some-image:latest` image will likely be pulled regardless of the `if: false`. the reasoning is always the same and the runner is trying to be as fast as it can.

now, why does github actions behave this way? the core idea is to optimize the overall execution time. by downloading all necessary images up front, github actions hopes to minimize the latency of starting each step. it’s a trade-off between resource usage and workflow speed. this makes sense in many cases, because most steps are usually going to run.

the issue arises when you have a significant number of conditional steps, large images, or a combination of both. it can lead to wasteful downloads, extending the build time, and using more bandwidth and resources than you’d expect and you might consider that it’s some sort of bug.

there isn't really a magic switch that turns off this behavior entirely. it's baked into the core mechanics of how github actions workflows are executed. but there are some strategies that can help:

1. **image size matters**: keep your images as lean as possible, this helps reduce the overhead of the pre-fetch. using a multi stage build docker file can help with the reduction of the final docker image size.
2. **conditional logic**: carefully consider your conditional logic in github actions. sometimes you can reorganize steps and conditional to reduce the number of steps that are likely to be skipped.
3. **lazy load images**: use a pattern where you download a small "base" image and then download other dependencies within the step. this will not solve the issue completely but it will help reduce the bandwidth.
4. **avoid overly complex conditions**: if it's possible to use simple conditions to skip steps, avoid using complex conditions, it can be a code smell that your workflow design is wrong.
5. **consider self-hosted runners**: if bandwidth is an issue, self-hosted runners can help, although you should take into consideration the maintenance of the virtual machines.
6. **cache**: use github actions caching to avoid pulling docker images that are already pulled in the same runner machine for different workflow runs.

the caching technique is quite important in this kind of problem:

```yaml
name: image_pull_with_cache

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: cache docker images
        uses: actions/cache@v3
        id: cache-docker
        with:
          path: /home/runner/.docker
          key: ${{ runner.os }}-docker-images-${{ github.sha }}
      - name: conditional step
        if: steps.cache-docker.outputs.cache-hit != 'true'
        uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: some-image:latest
```

in this example we're caching the docker image layer location, if the cache hit is `true`, the image will not be pulled. the image will be used directly from the cache location. the `key` is also important to be constructed correctly, and the same `key` should be used on the `actions/cache@v3` in the same workflow to properly load the cache from the previous runs. i prefer to use `runner.os` and `github.sha`, if the runner changes the operative system or if we're working on a different commit we should invalidate the cache so we download the latest image.

i remember i was working on a project that had a lot of conditional steps, it was a nightmare. then i read some white papers about caching of workflows and some good practices for github actions on the official github documentation, which helped me understand better how to deal with those issues. the official github actions documentation can help understand how the caching system works in a deep level. books like "effective devops" by jennifer davis or "continuous delivery" by jez humble can offer a different point of view of the same issue. it can help structure your workflow designs better.

i wish there were a perfect answer to this. it really isn’t an easy issue to deal with. hopefully these ideas and tips will help you avoid some of the frustration i've gone through, because it's very annoying when the runner is pulling an image you don’t need. i’ve lost count how many times i’ve had to double check that i didn’t miss something obvious, or if i was having a hallucination. in this case, no, you aren’t hallucinating, it’s just how the runner works.
