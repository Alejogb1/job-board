---
title: "Why am I getting a 'cannot load such file -- nokogiri' error when running Docker and docker-compose?"
date: "2024-12-23"
id: "why-am-i-getting-a-cannot-load-such-file----nokogiri-error-when-running-docker-and-docker-compose"
---

Alright, let's tackle this nokogiri loading issue within the context of Docker and docker-compose. It’s a frustrating error, I’ve certainly encountered it myself back when I was managing a complex Ruby on Rails application that heavily relied on web scraping. The crux of the problem, in my experience, often stems from inconsistencies between the build environment and the runtime environment, especially regarding native extensions like those used by nokogiri. It’s rarely a simple case of a missing gem in the Gemfile.

When you see that "cannot load such file -- nokogiri" error within a Docker container, it signals that the compiled binary version of the nokogiri gem isn’t present, or, critically, it's incompatible with the container's operating system and architecture. You might have included it in your Gemfile and even `bundle install`ed successfully, but the `bundle install` process often compiles native extensions against the system it’s being run on. This becomes an issue if the host machine's environment—where you likely built the initial Docker image—is substantially different from the container’s final operating system or underlying libraries.

Imagine this: you're developing on macOS, which has a particular set of system libraries and compilers. `bundle install` fetches nokogiri and compiles its C extensions using your macOS tooling, producing a binary specific to macOS. You then build your Docker image without properly considering this. The Docker image, however, is likely based on a Linux distribution (like Debian or Alpine), which is a totally different ecosystem. When the Docker container tries to load the macOS-compiled nokogiri binary, it fails, hence the `cannot load such file` message.

This error isn’t isolated to nokogiri; it’s common with any gem that incorporates native extensions. Libraries such as `pg` (for postgresql) or `ffi` face the same issues. A good analogy, though I usually avoid them, is attempting to use a key carved for one lock on another, different lock. They’re just incompatible.

The resolution usually involves ensuring that `bundle install` occurs *within* the Docker build process, inside an environment that mirrors the container’s target architecture. Moreover, it requires that all the necessary build dependencies are available in your Docker image.

Here are a few specific tactics, with code examples to highlight each one:

**1. Multi-Stage Builds and Bundling in a Builder Container**

This is my preferred method for most projects. The core idea is to isolate the compilation process in a separate 'builder' image, which then copies the compiled gems into the final 'runtime' image. This keeps the final image smaller and more secure.

```dockerfile
# --- Builder Stage ---
FROM ruby:3.2-bullseye AS builder

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .
RUN bundle exec rake assets:precompile # Optional, if you use asset pipeline
# --- Final Stage ---
FROM ruby:3.2-slim-bullseye

WORKDIR /app

COPY --from=builder /app/vendor/bundle ./vendor/bundle
COPY --from=builder /app/public ./public
COPY --from=builder /app/config ./config
COPY --from=builder /app/lib ./lib
COPY --from=builder /app/app ./app
COPY --from=builder /app/bin ./bin

# Your entry point script, e.g., "bin/rails server"
CMD ["./bin/rails", "server", "-b", "0.0.0.0"]
```

In this example:

*   The `builder` image is based on a full Ruby image, providing everything necessary for compilation.
*   The `bundle install` step occurs within this builder image, ensuring native extensions are compiled for the target architecture.
*   The final image is based on a slim Ruby image, keeping it small and lean. It only copies the compiled gems and your application code and assets from the `builder` stage.

**2. Explicitly Installing Build Dependencies within the Dockerfile**

If multi-stage builds are not an option for some reason (though I heavily recommend them), you need to ensure your Dockerfile has all the necessary build dependencies installed. This involves pre-installing the compiler and development libraries needed by `nokogiri` and other gems with native extensions.

```dockerfile
FROM ruby:3.2-bullseye

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .
# Your entry point script
CMD ["./bin/rails", "server", "-b", "0.0.0.0"]
```

In this approach:

*   We explicitly install `build-essential`, `libxml2-dev`, `libxslt1-dev`, and `zlib1g-dev` – key development packages needed by nokogiri. These names may vary slightly based on the base image of your Dockerfile, and I suggest you double-check the gem’s documentation on what to install or look up relevant community resources for your distribution.

**3. Leveraging `--platform` Flag (when targetting a different architecture)**

A more niche situation that I’ve encountered is when deploying to a different architecture, for example, targeting arm64 from an x86 machine. Docker supports the `--platform` flag during building. This informs Docker to target a specific architecture during the build process.
You might see something like this (and this would require QEMU for emulation):
```bash
docker build --platform linux/arm64 -t my-arm64-app .
```
However, for this scenario, you might need to also include the appropriate platform-specific gems. This approach can become complicated and I'd only recommend this if you have the necessary infrastructure and knowledge, as cross-compilation often introduces further points of failure.

**Key Resources**

To further your understanding of this and similar situations, I would recommend these specific resources:

*   **The Official Docker Documentation:** I mean, this is crucial for all things Docker. Pay special attention to sections on multi-stage builds and Dockerfile best practices.
*   **“Programming Ruby” by David Thomas, Chad Fowler, and Andy Hunt:** This book, though comprehensive, contains useful chapters on the underlying mechanisms of gem dependency management in Ruby. It is not an introductory book, but rather for experienced developers who wish to explore the underlying mechanisms.
*   **The Nokogiri Gem's README on GitHub:** Specifically, check sections related to installation and build dependencies. Often, the most up-to-date advice and specific library dependencies are listed here. You may find recommendations on the specific packages you need for different operating systems.
* **The gem documentation in RubyGems itself.** Often includes specific details that are not widely published elsewhere, and can be vital if you encounter a tricky situation.

In summary, the "cannot load such file -- nokogiri" error in a Docker context almost always points to a mismatch in compilation environments. By meticulously controlling the build process and ensuring the correct dependencies are present, this issue is very much resolvable. My personal inclination remains towards multi-stage builds; they promote a clean, repeatable and portable workflow. Always remember, with Docker, your build environment *is* your deployment environment.
