---
title: "How to reference an environment variable from devcontainer when vscode is connected to a remote machine via ssh?"
date: "2024-12-14"
id: "how-to-reference-an-environment-variable-from-devcontainer-when-vscode-is-connected-to-a-remote-machine-via-ssh"
---

alright, so referencing environment variables within a devcontainer when your vscode is connected remotely via ssh, yeah, i've been down that rabbit hole. it's not always straightforward, and i've definitely spent more than a few late nights debugging this exact scenario. the core problem, as i see it, is that environment variables aren't automatically propagated everywhere by magic. there's a chain of processes and contexts involved, and we need to ensure those variables make their way through each step.

let’s start with my first encounter with this issue, a few years ago. i was working on a python microservice project, and we had a ton of sensitive api keys stored as environment variables on our development server. we used devcontainers for consistency, since i am always pushing for that, it's a lifesaver. initially, things seemed fine, we had `.env` files and everything looked good, but when i tried to connect via ssh to a remote machine and fire up a container, the application crashed because, obviously, all the keys were missing. initially we thought it was docker, docker-compose or ssh errors, but it turned out that the environment variables needed more explicit handling.

first, let's understand the basics. when you use `devcontainer.json`, vscode essentially builds and starts a docker container. vscode needs to communicate with the container to make your life easier. the key here is that vscode executes the `devcontainer.json` directives on your local machine, then it connects using the ssh remote extension to the remote machine and inside there the docker container executes. the env variables in your host operating system (where vscode is running) are not automatically forwarded to the container. this is a security feature, and a good thing in my opinion.

the approach i’ve found to be most reliable revolves around passing those variables through the ssh connection and into the docker container build and runtime environment.

here's how it generally works:

1.  **ssh configuration:** first, you need to explicitly tell ssh to forward the necessary environment variables. you can do this by modifying your `~/.ssh/config` file (or equivalent). for example, if you want to forward a variable named `api_key`, you add this to your configuration for the remote server (this is not the devcontainer config):

```
host your_remote_host
    hostname your_remote_server_ip
    user your_remote_user
    forwardagent yes
    sendenv api_key
    sendenv another_key
```

here `your_remote_host` is a nickname for your remote server, and the others are self explanatory. `sendenv` specifies the names of environment variables you want to forward. this tells ssh to actually send the env variables over the wire. if you want to send all env variables you can use sendenv *.

2.  **devcontainer setup:** next, we need to ensure these forwarded variables are picked up within the devcontainer. this usually involves two main places: the `devcontainer.json` file, and your dockerfile if you have one. in the `devcontainer.json`, you can define the `remoteEnv` section, this section allows variables to be made available during container startup:

    ```json
    {
      "name": "my-devcontainer",
      "build": {
        "dockerfile": "Dockerfile"
      },
      "remoteEnv": {
           "api_key": "${localEnv:api_key}",
           "another_key": "${localEnv:another_key}"
      }
    }
    ```

    the `"${localEnv:variable_name}"` syntax is crucial. it tells the devcontainer to grab the variable that has been forwarded via ssh, with the specific name that comes after the colon. if you use `sendenv *` you can potentially forward any variable you need just by specifying `"${localEnv:variable_name}"` inside your `devcontainer.json`.

3. **dockerfile configuration (if needed):** sometimes, you might need to bake these environment variables into your image itself (for build process reasons, or because you want to persist them in a new image). in this case you might need to use the `ARG` instruction and then pass the environment variable as a build argument.

    ```dockerfile
    FROM ubuntu:latest

    ARG api_key
    ENV API_KEY=$api_key

    ARG another_key
    ENV ANOTHER_KEY=$another_key
    ```

    note how `ARG` is used for defining variables passed during build time and `ENV` to actually set the environment variables in the container's runtime.
    to actually pass the values during the image creation you might need to add a build section in your `devcontainer.json`. i don’t usually use this, but i’ve seen cases where this was useful.

    ```json
        {
        "name": "my-devcontainer",
            "build": {
            "dockerfile": "Dockerfile",
            "args": {
                 "api_key":"${localEnv:api_key}",
                 "another_key":"${localEnv:another_key}"
                }
            },
          "remoteEnv": {
           "api_key": "${localEnv:api_key}",
           "another_key": "${localEnv:another_key}"
          }
        }
    ```
    this makes the environment variables available in both the build and runtime phases of the docker container creation and execution. if you only need to have the variables available during runtime the `remoteEnv` configuration is the one you will be using 99% of the time.
4.  **checking**: finally, to make sure that everything is working correctly, once the devcontainer is running, open a terminal within the container and run `printenv` or `env`. you should see the variables like `api_key` and `another_key` correctly defined, if not, then you know that something is not properly configured and can go back to troubleshooting.

over the years, i have also seen some folks trying to work around this by directly exposing environment variables into the docker container using docker commands, like `-e` argument, or in docker-compose files, but i'd advise against this approach. it's less portable and complicates things unnecessarily, especially if you have many variables. in my experience, it's better to stick with explicit ssh forwarding and using the `remoteEnv` section of `devcontainer.json` as demonstrated.

one interesting case i faced was when the variables where not being correctly exported because the ssh agent was not correctly configured, and the variables needed for it to work were not being properly forwarded. in that case i had to add `forwardagent yes` to my ssh configuration, as specified in the previous code snippet, in the host machine, so the remote machine can also access my keys to forward the env variables through ssh. sometimes these things can be really tricky. there was also the time i spent two hours just looking for a typo in the `sendenv` variables… you know, just another day as a developer (i swear, one day i'm going to find a good spell checker for code) but you get the point, these things can be annoying.

the thing is, most of the time you might need to configure a `.env` file in your project directory, this usually works fine for local development. but for remote development, you will have to explicitly pass variables through the ssh tunnel. my experience has been that you need a configuration similar to the examples above.

as for resources, i'd suggest looking at the official vscode documentation about devcontainers, it's actually pretty good. the official docker documentation is also fundamental for understanding the build process and how environment variables are handled at build and runtime time. also, if you are interested in security aspects related to forwarding of ssh keys and environment variables i would recommend *secure shell: the definitive guide* by richard barrett and rj silverman. it is an old book, but it has really good explanations that will give you a deep understanding of the underlying processes, although not really related to the topic at hand.

in summary, the key to passing env variables from your local machine to a devcontainer running remotely over ssh is to use `sendenv` in your ssh configuration to forward the variables, and then to reference those forwarded variables within your `devcontainer.json` using the `${localEnv:variable_name}` syntax, and in some cases pass `ARG` variables using the `args` key. you may need to do this on a per variable basis depending on your use case. also keep in mind that some variables, like ssh keys, require special attention to be correctly forwarded through ssh. it seems cumbersome at first, but once you have a good understanding it's quite simple and efficient.
