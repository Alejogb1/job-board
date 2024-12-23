---
title: "Why are gem native extensions failing to build in Docker images?"
date: "2024-12-16"
id: "why-are-gem-native-extensions-failing-to-build-in-docker-images"
---

, let's tackle this one. I've seen my share of gem native extension build failures within Docker, and it can definitely feel like chasing ghosts if you don't understand the underlying causes. It’s a particularly annoying issue because the build often works perfectly on your local machine, but falls apart in the container. It's almost never a random occurrence. There are usually specific and logical reasons why this happens, primarily stemming from discrepancies in the environment, dependencies, and build processes.

When I first encountered this problem a few years back, while setting up a continuous integration pipeline for a rather complex rails application, I nearly pulled my hair out. The application relied heavily on image processing libraries, many of which had native extensions compiled from C. Locally, development was smooth sailing. But in the dockerized CI/CD environment, these gems kept failing during the `bundle install` phase. After much troubleshooting, it became clear that the Docker environment wasn’t as close to my development setup as I’d initially thought.

The core issue often boils down to these key areas:

**1. Missing Build Tools and Dependencies:**

Native extensions require specific system-level tools to compile, like `make`, `gcc`, or potentially `clang`. These tools, along with the required libraries for the extension, might not be present or correctly configured within your base Docker image. A common scenario is using a minimal image to save space, which often sacrifices these crucial build tools. For example, the `alpine` images, while efficient, require you to manually install most build dependencies. This is frequently seen with image processing gems like `rmagick` or cryptography-related gems relying on `openssl`.

**2. Incompatible Architecture or Operating System:**

Another common culprit is building on an architecture different from the target deployment environment. If you develop on an x86_64 machine but deploy on an ARM64 server (or vice-versa), pre-built native extensions will likely be incompatible. The same problem occurs with minor version differences of underlying libraries (for example, different version of `libc`). If you compile the extensions *within* the Docker build process on the correct target system, this mismatch is avoided. However, pre-built binaries from gems are tied to the environment they were compiled in. This is why it is crucial to build native extensions *inside* the final docker image, and not use pre-built binaries.

**3. Incorrect or Incomplete System Libraries:**

Some native extensions depend on specific system libraries that aren't provided by the base image or aren't in the correct version required. These are often low-level libraries that interact directly with the operating system. A classic example is a gem needing `libvips` for image manipulation or a specific version of `libpng`. Even subtle version mismatches can cause build failures or runtime errors.

**4. Issues with Bundler Configuration:**

Sometimes the problem isn't the environment but how Bundler is instructed to install the gems. If you are relying on bundler to install pre-built binaries, or are attempting to lock to a platform with the `bundle lock --add-platform <os-arch>` flag, you must ensure that all gems have viable pre-built binaries available for the specified platform. If the pre-built binary doesn't exist, Bundler will not be able to download it and will likely fail.

Now, let's illustrate these points with some code snippets.

**Example 1: Addressing Missing Build Tools (Dockerfile)**

This example shows how you might adjust a Dockerfile to add build tools for a gem requiring compilation, assuming a debian-based image:

```dockerfile
FROM ruby:3.2-slim

# install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpq-dev \ # example of db-related native extension dependency
    zlib1g-dev # common lib for compressions

# Copy gemfile
COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .
# more of the app setup here...
```

In this `Dockerfile`, we're adding `build-essential`, `cmake`, `libpq-dev`, and `zlib1g-dev` as dependencies. `build-essential` provides `gcc`, `make` and other necessary tools for compiling native extensions. `libpq-dev` is included since many applications use PostgreSQL which has a native gem. The `zlib1g-dev` is a very common dependency for many gems, including some that process images or text files. By including these, we avoid the build failures caused by missing tools.

**Example 2: Platform-Specific Gems (Gemfile)**

To handle the issue of platform-specific gems and avoid relying on pre-built binaries, one can utilize the `platforms` functionality of the Gemfile. Here's how to use this to lock a gem to the `x86_64-linux` platform for building native extensions:

```ruby
source 'https://rubygems.org'

gem 'nokogiri', '~> 1.16'

# Gem that has a native extension and may cause issues if built for the wrong architecture
gem 'pg', '~> 1.5'

# If you encounter issues with pg in your docker build, lock the build process to the desired architecture
# this would typically be the architecture of the target deployment server
# note: you should not do this for *all* gems, only those that have proven problematic
platforms :x86_64_linux do
 gem 'pg', '~> 1.5'
end
```

This example shows how to configure `nokogiri` and `pg` for all platforms, but then ensures `pg` has its native extensions built under `x86_64-linux` conditions. This forces Bundler to build the `pg` gem if the platform during build is different to this. The build of native gems should be done on the *target* environment and not on the development host.

**Example 3: Using a Multi-Stage Build for Smaller Images**

Finally, here’s a multi-stage dockerfile showing how to reduce the final image size, while still building native extensions:

```dockerfile
# Stage 1: Builder
FROM ruby:3.2-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpq-dev \
    zlib1g-dev
WORKDIR /app
COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .

# Stage 2: Final image (cleaner runtime)
FROM ruby:3.2-slim

WORKDIR /app
COPY --from=builder /app /app
RUN apt-get update && apt-get install -y --no-install-recommends \
  libpq5
COPY . .
# ... rest of the setup...

CMD ["rails", "server", "-b", "0.0.0.0"]
```

This multi-stage build first creates a builder stage (`builder`), where the gems including native extensions are built and installed, followed by a final image that only copies the built gems and necessary runtime dependencies. This allows you to maintain a smaller final image while ensuring build tools are present when needed. I have included an additional runtime dependency (`libpq5`) to illustrate that the dependencies used at *build* time may differ from those at *run* time. It’s best to investigate which dependencies are required for your application for both build and run phases.

**Resources for Deeper Understanding**

For a comprehensive understanding of Docker and its intricacies, I highly recommend “Docker Deep Dive” by Nigel Poulton. It provides a detailed and practical approach to docker containerization.

For further details on building and handling native extensions, the RubyGems documentation itself is an excellent source. Pay specific attention to the sections discussing native extension compilation, and the Gemfile specification. Also, the documentation for your specific gems that utilize native extensions is crucial for understanding the requirements of that particular gem.

In conclusion, gem native extension build failures within Docker containers are rarely random. They usually stem from mismatches in the build environment compared to the target deployment environment. Addressing these differences through tools, dependencies and a proper understanding of your Gemfile and Bundler’s configuration will help to solve this issue. Understanding your build environment is half the battle.
