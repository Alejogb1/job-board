---
title: "How can I build and push Docker images without specifying a registry in a DevOps pipeline?"
date: "2024-12-23"
id: "how-can-i-build-and-push-docker-images-without-specifying-a-registry-in-a-devops-pipeline"
---

Alright, let’s tackle this challenge. I've seen this situation pop up more than a few times across different projects, and it's usually when teams are trying to avoid hardcoding registry URLs directly into their pipelines. There are some very valid reasons for that – flexibility, easier environment management, and improved security being the primary ones. Essentially, the core issue you're facing is how to decouple the image build and push processes from specific registry locations within your devops workflow.

The first thing to understand is that docker isn't inherently tied to a specific registry during a build. The `docker build` command focuses solely on constructing the image layers from your dockerfile and context. The registry comes into play only when you *tag* and *push* the resulting image. This gives us the leverage we need – we just need to handle the tagging and pushing operations carefully within our pipelines.

Let’s start by focusing on a scenario where you’re using a simple gitops approach, leveraging something like GitHub Actions or GitLab CI. The primary concern revolves around defining your image name and tag in a way that isn’t fixed to a particular registry. My experience tells me that relying heavily on environment variables to determine the registry target usually provides the best balance between flexibility and control.

Here's a fundamental concept I usually apply: I use a base image name, along with dynamically generated tags that usually relate to the branch name and/or commit hash, and then prepend the registry location only during the push phase. This lets the build process be largely generic.

Here's an example showcasing how to construct the tag dynamically within a github actions workflow:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches:
      - main
      - develop

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          IMAGE_NAME="my-base-image"
          IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8} #extract branch, replace slashes and truncate commit hash
          docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

      - name: Login to Container Registry
        run: |
          echo "${{ secrets.CR_PASSWORD }}" | docker login ${{ secrets.CR_SERVER }} -u ${{ secrets.CR_USERNAME }} --password-stdin

      - name: Push Docker image
        run: |
           IMAGE_NAME="my-base-image"
           IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8} #extract branch, replace slashes and truncate commit hash
           REGISTRY_URL="${{ secrets.CR_SERVER }}" # using github secret to obtain registry url
           docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
           docker push ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
```

In this snippet, you'll see how the image tag is created from the branch name, commit sha, and the image name is fixed. The registry credentials and server url are obtained from github secrets, and these are only utilized during the `docker login` and `docker push` steps. The important concept is that we are dynamically assembling the full image path (registry url + image name + image tag) right before the push operation. This method makes your workflow reusable in different environments.

Another practical scenario involves situations with multiple potential registries, perhaps across different environments (dev, staging, prod). In these cases, a more sophisticated approach is required, possibly leveraging a configurable variable. You can pass this variable (containing the registry url or a reference to it) as a pipeline variable. This approach gives a lot of control over where your images are published.

Here’s an example using gitlab ci where we utilize a `CI_ENVIRONMENT_NAME` gitlab predefined variable along with an additional environment variable for a production-like registry:

```yaml
stages:
  - build
  - push

build-image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - export IMAGE_NAME="my-app"
    - export IMAGE_TAG=${CI_COMMIT_SHORT_SHA}
    - docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
  artifacts:
    paths:
      - image.tar # optional: save the image as an artifact if needed
    expire_in: 1 day

push-image:
  stage: push
  image: docker:latest
  services:
    - docker:dind
  script:
    - |
       if [[ "${CI_ENVIRONMENT_NAME}" == "production" ]]; then
           export REGISTRY_URL=$PRODUCTION_REGISTRY_URL  #defined as a gitlab variable in the project settings.
       elif [[ "${CI_ENVIRONMENT_NAME}" == "staging" ]]; then
            export REGISTRY_URL=$STAGING_REGISTRY_URL # or use gitlab environment variables
       else
           export REGISTRY_URL=$DEV_REGISTRY_URL # or use gitlab environment variables
       fi
    - export IMAGE_NAME="my-app"
    - export IMAGE_TAG=${CI_COMMIT_SHORT_SHA}
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
    - docker push ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
  dependencies:
      - build-image
```

In this gitlab example, the `push-image` stage makes a conditional choice of the `REGISTRY_URL` based on the gitlab variable `CI_ENVIRONMENT_NAME`, demonstrating how dynamic registry assignment can be achieved based on runtime settings. The environment-specific registry URLs can be stored as gitlab variables, providing another layer of flexibility and keeping those settings out of the direct yaml file.

A third, more involved situation, appears when dealing with multi-architecture images, which also requires a bit of work. You have to consider that image manifests need to be handled specifically, as you cannot tag them individually. You will need to generate the manifest list and push this, instead of pushing the individual images. This is crucial when supporting platforms like ARM64 and x86_64. The steps would include building the separate images, pushing these with a tag that identifies their architecture, and then creating the manifest list and pushing this one with the main tag:

```yaml
name: Multi-Architecture Build and Push

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform: [linux/amd64, linux/arm64]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image for ${{ matrix.platform }}
        run: |
          IMAGE_NAME="my-multi-arch-image"
          IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8}-${{ matrix.platform }}
          docker buildx build --platform ${{ matrix.platform }} -t ${IMAGE_NAME}:${IMAGE_TAG} --load .

      - name: Login to Container Registry
        run: |
          echo "${{ secrets.CR_PASSWORD }}" | docker login ${{ secrets.CR_SERVER }} -u ${{ secrets.CR_USERNAME }} --password-stdin

      - name: Push Docker image for ${{ matrix.platform }}
        run: |
            IMAGE_NAME="my-multi-arch-image"
            IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8}-${{ matrix.platform }}
            REGISTRY_URL="${{ secrets.CR_SERVER }}"
            docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
            docker push ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}

      - name: Create Manifest List
        if: matrix.platform == 'linux/amd64'
        run: |
             IMAGE_NAME="my-multi-arch-image"
             IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8}
             REGISTRY_URL="${{ secrets.CR_SERVER }}"
             docker manifest create ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG} \
              ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}-linux/amd64 \
              ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}-linux/arm64

      - name: Push Manifest List
        if: matrix.platform == 'linux/amd64'
        run: |
            IMAGE_NAME="my-multi-arch-image"
            IMAGE_TAG=$(echo "${GITHUB_REF#refs/heads/}" | sed 's/\//-/g')-${GITHUB_SHA::8}
            REGISTRY_URL="${{ secrets.CR_SERVER }}"
            docker manifest push ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}
```

This example demonstrates how to push multi-architecture images utilizing `docker buildx build` and then combines the images into a manifest list, demonstrating a more complex but extremely useful scenario for handling varied architectures.

For further study on the underlying technologies, I highly recommend diving into the official Docker documentation (especially around `docker build`, `docker tag`, and `docker manifest`), and looking into cloud provider documentation regarding the use of container registries, including security aspects. Also "Effective DevOps" by Jennifer Davis and Ryn Daniels is very useful to understand the different scenarios involved in setting up pipelines for image building and deployment. The "Docker Deep Dive" book by Nigel Poulton provides very detailed information about docker itself, which is also invaluable.

In summary, the key to building and pushing docker images without a fixed registry is to decouple the build phase from the push phase, dynamically constructing the fully-qualified image name only right before the push and leveraging environment variables or a config management solution. These techniques ensure the portability and flexibility you need in complex environments. I've found these methods provide a reliable and robust way to manage image deployments across my projects.
