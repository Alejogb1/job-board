---
title: "Why Ruby on Rails on Docker - Bundler seems to install different versions of gems?"
date: "2024-12-15"
id: "why-ruby-on-rails-on-docker---bundler-seems-to-install-different-versions-of-gems"
---

let's tackle this, i've definitely been down this rabbit hole before, and it's frustrating when you expect consistent gem versions and get something totally unexpected. the short answer, the reason bundler sometimes seems to install different gem versions when using ruby on rails within docker is a combination of factors, primarily related to how docker layers are built and how bundler resolves dependencies. let's break it down a bit.

first, understand that docker builds images layer by layer. each command in your `dockerfile` effectively creates a new layer. when you modify your `gemfile` or `gemfile.lock`, and rebuild, docker might only rebuild layers that have been changed. this is a good thing, it's part of the power of docker, it makes builds faster when you're only tweaking small parts. however, it can lead to confusion if not handled precisely.

now, here's where the bundler magic, or lack of magic, kicks in. when bundler runs `bundle install`, it examines your `gemfile` for the gems you want, then uses the `gemfile.lock` to fix the specific versions. this lockfile is crucial. it's meant to ensure everyone in your team gets exactly the same gem versions. however, if you're not careful, different builds can end up with slightly different `gemfile.lock` states, and this happens when docker caches incorrectly.

let's say, for example, you have a dockerfile that does the following, very simplified:

```dockerfile
from ruby:3.2

workdir /app

copy gemfile gemfile.lock ./

run bundle install
```

now, here is the issue this could trigger: suppose you've done the initial `bundle install` outside of docker, generated a `gemfile.lock` then added the file to the docker context. the initial docker build will use this lockfile to install gems inside the container. great so far.

now, you decide to add a new gem to the `gemfile` , do a `bundle install` outside docker, again it creates a new `gemfile.lock` with the new changes and copy the two files into the docker context. here is the issue. docker will check its layers, and only rebuild the layer from the copy command, and then the run command, but it will keep the cached layer from the `from ruby:3.2` instruction and if you had a gem already installed inside of the base ruby image it will not be overwritten. so if the installed gem has conflicts with the new gem that you install inside of the docker container. boom, you're seeing different gem versions. so this might not look like a docker problem at all, but it's a layer caching issue and it could trigger unpredictable behavior.

another common scenario is when your `gemfile.lock` gets changed outside of docker, and you copy an older version of the lockfile to the docker context, when building the image, the old dependencies are installed inside of the docker. and then later you copy the new changed lockfile to the docker, then docker might find the difference and rebuild the layer which installs gems. however, that does not guarantee that all installed gems have the latest versions because gems could be installed previously in layers, and docker tries to be very conservative with its cache mechanism. so this can be tricky.

to mitigate this, i've learned a couple of tricks over the years. the first is, always, and i mean *always* copy your `gemfile` and `gemfile.lock` in the same copy statement in the dockerfile:

```dockerfile
from ruby:3.2

workdir /app

copy gemfile gemfile.lock ./

run bundle install
```

this ensures that if *either* file changes, docker will invalidate the cache for this layer, and it will be forced to rebuild, meaning that `bundle install` will run again. this is a fundamental fix, but it does not solve the problem of old gems installed on the base image.

secondly, consider cleaning your gems and your cache before install the new ones. this will ensure that there are no hidden gems that could cause issues in your system. this can be done before the `bundle install` call.

here is a slightly improved dockerfile example with gem clean up:

```dockerfile
from ruby:3.2

workdir /app

copy gemfile gemfile.lock ./

run gem uninstall -ax && bundle install --clean
```

this `gem uninstall -ax` removes all gems installed in your ruby base image, then `bundle install --clean` forces bundler to double check and install only the gems that you have in your `gemfile`. this approach works pretty well and makes sure that your docker builds are consistent. it has never failed me, except when i didn't double check my environment variables.

and a third strategy i've used, for when i was doing a more complex set of builds, and it also ensures immutability is to use a multi-stage build. this means you first install gems in one container, then copy the gems needed for the other container. this approach prevents you from installing gems in other layers that you don't want to.

here is a complete dockerfile example for multi-stage builds:

```dockerfile
# builder image
FROM ruby:3.2 as builder

WORKDIR /app

COPY gemfile gemfile.lock ./
RUN gem uninstall -ax && bundle install --clean --jobs 4

# final image
FROM ruby:3.2

WORKDIR /app
COPY --from=builder /app/vendor/bundle vendor/bundle
copy . .
run bundle exec rake assets:precompile
# rest of configurations
```

here the first image installs the gems, and the second image only copies the gem files from the builder and does not trigger a `bundle install` again, this makes sure that the install step is only done once and that the production image does not contain unnecessary gems.

it is worth mentioning that gem dependencies might come from sources that are not listed on your lockfile. i know, i know... it doesn't make sense. however, if you add a gem that depends on other gems you didn't install, bundler might try to install the sub-gems from different sources. in order to avoid this, you have to lock every dependency on your lockfile. in order to do this you need to make sure you have the `--with-git` argument when you install your gems, in order to ensure that the git dependencies are also listed on the lockfile, even when installing from a gem source, not a git source.

so it is very important that your lockfile contains the git references of the gems if you have any, or if you have any gems with dependencies that are not on your `gemfile`. for example `bundle install --with-git`. but this is not mandatory it depends on your project configuration.

i remember one time, a few years back, when we were transitioning to docker at my previous job, i spent two days because of this exact issue. it turned out that one of the developers was running `bundle install` without `--clean` on the development machine, which led to slightly different dependency versions in the lock file, and because of docker caching our containers were having a real mess of different gem versions. it was quite annoying to trace the source of the bug that was producing inconsistencies in production. it's the kind of stuff that makes you want to use some other tech, but then you remember it's always something silly like this that breaks things.

in terms of resources to understand all of this better, i'd really recommend looking into the bundler documentation itself, specifically about the `gemfile.lock`, how dependencies are resolved and especially on how the `--clean` option works, they have very good examples. for more details about how docker caching works, check the official documentation and search for "docker layer caching", also, some practical articles on multi-stage builds can help you a lot when debugging your dockerfiles.

the key to all this is being mindful of docker's layering system, making sure your `gemfile.lock` is consistent across your development and build environments, and always cleaning up before installing gems. it's a common problem, and once you get the hang of it, it becomes much easier to manage your ruby on rails projects with docker.
