---
title: "How can I build a CI/CD pipeline for Ruby projects with multiple gems or microservices?"
date: "2024-12-23"
id: "how-can-i-build-a-cicd-pipeline-for-ruby-projects-with-multiple-gems-or-microservices"
---

,  Building a robust ci/cd pipeline for ruby projects, particularly when you're dealing with multiple gems or microservices, is a challenge I've faced firsthand in a few different iterations over the years. It’s less about finding the *one* perfect solution and more about assembling the correct building blocks for your specific scenario. Here's how I approach it, combining practical experience with some best practices.

First, forget the notion of treating all pipelines identically. A single monolithic pipeline trying to handle multiple microservices is, in my experience, a recipe for inefficiency and eventual pain. Instead, think about breaking down your workflow into component parts, each with its dedicated pipeline. The key is managing the interdependencies gracefully.

The foundation of any good ci/cd pipeline, for Ruby or any other technology, revolves around a version control system. I'm a big fan of git, and I assume we’re all on the same page there. From a commit to the repository, it should trigger an automated process, which includes:

1.  **Automated Testing:** This is non-negotiable. Each code commit should launch a series of tests, unit tests, integration tests, and if appropriate, end-to-end tests. The goal here is to catch errors early in the lifecycle.

2.  **Build Artifact Creation:** If you're dealing with multiple gems, each should have its own build process, potentially creating gem packages. If it's a microservice, this would generally result in a deployable image, likely a docker image.

3.  **Release and Deployment:** Once the build is complete and the tests have passed, the pipeline needs to push those artifacts into the next stage, which might be a staging environment and finally, production.

Now, how to orchestrate all of this? I've found that using tools specifically designed for ci/cd are critical for success. For Ruby, popular options include, but are not limited to, github actions, gitlab ci, and circleci. I'll use gitlab ci for our examples, as its yaml configuration is very explicit and easy to follow, and it's something I've used quite extensively.

Let's start with an example of a gem project pipeline:

```yaml
stages:
  - test
  - build
  - publish

test:
  image: ruby:3.2 # or whatever version you need
  stage: test
  script:
    - bundle install
    - bundle exec rspec

build:
  image: ruby:3.2
  stage: build
  script:
    - gem build *.gemspec
  artifacts:
    paths:
      - "*.gem"

publish:
  image: ruby:3.2
  stage: publish
  only:
    - main # Or your main branch name
  script:
    - gem push *.gem # assuming you have credentials configured for the gem server
```

In this example, we've got three stages: test, build, and publish. The *test* stage runs the rspec test suite. The *build* stage creates a gem package, and the *publish* stage pushes the gem to a gem repository upon successful builds on the 'main' branch. The artifact step is critical to pass along build outputs, preventing recomputation.

Next, let’s look at a slightly more complex example that illustrates a microservice pipeline:

```yaml
stages:
  - test
  - build
  - dockerize
  - deploy

test:
  image: ruby:3.2
  stage: test
  services:
      - docker:dind
  script:
    - bundle install
    - bundle exec rspec
    - docker network create test_net || true

build:
  image: ruby:3.2
  stage: build
  script:
     - bundle install
     - RAILS_ENV=production bundle exec rails assets:precompile # if it's a rails app

dockerize:
  image: docker:20.10.12
  stage: dockerize
  services:
     - docker:dind
  variables:
    DOCKER_IMAGE: your-repo/your-app
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker push $DOCKER_IMAGE:$CI_COMMIT_SHA
  artifacts:
    paths:
       - Dockerfile

deploy:
  image: alpine:latest
  stage: deploy
  only:
    - main # Or your main branch name
  script:
   - apk add curl
   - curl -X POST -H "Content-Type: application/json" -d "{\"image_tag\": \"$CI_COMMIT_SHA\"}" "$DEPLOYMENT_ENDPOINT"
```

Here, after the *test* and *build* stages, we introduce a *dockerize* stage. It builds a docker image using a dockerfile (which we assume exists in your repository), tagging it with the current commit sha and pushes it to a docker registry. The deploy stage then invokes a remote deploy endpoint via a http request. This is a fairly simplified deployment approach and would typically involve other tools like kubernetes, but serves to showcase the main idea. Environment variables are used extensively for the deployment and access to the image. `docker:dind` is employed for docker-in-docker capability.

Finally, let's add an example of a pipeline that utilizes another gem within the same organization:

```yaml
stages:
  - test
  - build
  - dockerize
  - deploy

test:
  image: ruby:3.2
  stage: test
  services:
      - docker:dind
  script:
    - bundle install
    - bundle exec rspec
    - docker network create test_net || true

build:
  image: ruby:3.2
  stage: build
  before_script:
   - bundle config set --local path vendor/bundle # ensure your internal gems are located within the bundle
   - gem install your_internal_gem --version "1.0.0" # specify version number for internal gem, ensure it exists within your registry
  script:
     - bundle install
     - RAILS_ENV=production bundle exec rails assets:precompile

dockerize:
  image: docker:20.10.12
  stage: dockerize
  services:
     - docker:dind
  variables:
    DOCKER_IMAGE: your-repo/your-app
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker push $DOCKER_IMAGE:$CI_COMMIT_SHA
  artifacts:
    paths:
       - Dockerfile

deploy:
  image: alpine:latest
  stage: deploy
  only:
    - main # Or your main branch name
  script:
   - apk add curl
   - curl -X POST -H "Content-Type: application/json" -d "{\"image_tag\": \"$CI_COMMIT_SHA\"}" "$DEPLOYMENT_ENDPOINT"
```

In this case we utilize a specific version of an internal gem that was previously built and pushed to your registry. It is specified in the before script to be installed prior to running any other tests or build commands. This allows teams to share functionality among their microservices without a strong coupling among codebases.

A critical aspect, which the examples hint at but don't exhaustively cover, is *dependency management*. When microservices depend on each other, or gems are used across projects, you absolutely must establish a robust versioning strategy and ensure that your deployments are only pulling the correct versions of each dependency. For that, I strongly recommend reading up on Semantic Versioning (SemVer) as a starting point for how to properly version these gems and internal dependencies. Beyond that, reading through "Continuous Delivery" by Jez Humble and David Farley, would be immensely beneficial. Finally, "The Phoenix Project" by Gene Kim, Kevin Behr, and George Spafford, despite not being strictly a technical book, will provide you with perspective on how a healthy culture is paramount to a successful CI/CD implementation.

Furthermore, don’t underestimate the value of monitoring and logging in your pipeline. Once code is deployed, it's crucial to track performance and identify any issues, which are going to happen. Tools like prometheus and grafana are a solid starting point for metrics collection, while splunk or the elk stack offer superb logging capabilities. Also, make it a habit to regularly review your ci/cd configuration and tweak it for optimal efficiency.

Finally, and this is often overlooked, communicate your changes. Document your pipeline steps and inform your team about changes to how the application or gems are deployed. A solid, well-documented process is as critical as well-functioning pipeline automation.

In conclusion, building an effective ci/cd pipeline is not a one-time task but rather a journey of continuous improvement. It involves careful consideration of dependencies, effective use of automation tools, and continuous attention to the monitoring and logging mechanisms. Remember, a well-oiled pipeline is one of the best investments you can make for your software development process.
