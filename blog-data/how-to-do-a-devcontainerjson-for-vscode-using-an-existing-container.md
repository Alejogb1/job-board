---
title: "How to do a devcontainer.json for VScode using an existing Container?"
date: "2024-12-15"
id: "how-to-do-a-devcontainerjson-for-vscode-using-an-existing-container"
---

alright, so you're looking to hook up vscode with a devcontainer using an already running container, right? i've been down that road, more times than i care to recall, and trust me, it can be a little fiddly the first couple of tries. i’m assuming you're already comfortable with docker and have a container kicking around, maybe one you've been using for local development or some kind of personal project.

let's break it down. the goal here is to get vscode to connect to that existing container for its development environment rather than having it build a brand new one every time. you're bypassing that initial build process, which can be quite time-consuming, especially when dealing with more complex setups. the key piece is that `devcontainer.json` file. this little json guy tells vscode how to interact with your container.

here's the basic approach, followed by some example configuration snippets, and then i'll throw in some tips i've learned over the years.

the core of making it work is to avoid using the `build` key in your `devcontainer.json`, instead we specify the `container` property. this property points vscode to the id or name of your pre-existing container. we also use the `workspaceFolder` key, to specify the location of your source code in the container file system. think of it like this: you're telling vscode to attach to a running instance instead of spinning up a fresh one from a dockerfile.

let's see a basic example:

```json
{
    "name": "existing-container-dev",
    "container": "my-existing-container",
    "workspaceFolder": "/home/dev/myproject",
    "customizations": {
		"vscode": {
			"extensions": [
				"ms-python.python",
                "esbenp.prettier-vscode"
			]
		}
    },
    "remoteUser": "dev"
}
```

in this setup, `my-existing-container` should be the name of your container, and `/home/dev/myproject` should be where your code lives inside it. the `customizations` and `remoteUser` fields are optional but provide you with a place to add vscode extensions to make sure you’re development tools are all on the same page. you can add extensions that have a server component so that it can be accessed from inside the container. also, `remoteUser` ensures that any vscode operations happen under the right user in the container.

one issue i encountered when i started using this approach was getting the folder mapping setup properly, it resulted in vscode complaining that the working folder did not match between host and container. this can be especially annoying if you're doing anything with file watching, or working on a compiled language where vscode is constantly trying to sync up changes.

for instance, if your container has a mounted volume and is being ran by the command `docker run -v /local/path:/container/path my-image` then your devcontainer.json needs to mirror the `/container/path` and not the `/local/path`. vscode accesses the container filesystem directly, and does not know where your host folders are located, but it does know where they are on your container.

another important aspect is the container's user configuration. by default, the `remoteUser` value is usually the user that started the container, but this might not always align with vscode's expectations. if you find yourself with permission issues, this is one place to begin your inspection. the `remoteUser` value can be configured to `root` user if you wish.

moving on to another configuration example, this time we're using a container id instead of a name, this could be usefull for those who like to make a container from an image by the command line instead of by using docker-compose.

```json
{
    "name": "existing-container-dev",
    "container": "d4567b679a87",
    "workspaceFolder": "/app",
    "customizations": {
        "vscode": {
            "settings": {
                "python.pythonPath": "/usr/local/bin/python",
                "terminal.integrated.defaultProfile.linux": "bash"
            },
            "extensions": [
                "ms-python.python",
                "ms-vscode.cmake-tools"
            ]
        }
    },
    "remoteUser": "myuser"
}
```

here the `container` value is set to a container id. you can get a container's id by running `docker ps -a` and then copying the corresponding value. also note how we are using the `settings` value to specify which version of `python` will be used and the terminal that will be started when opening the container, which can be very handy in those environments that uses a different terminal by default. this `settings` field also helps in ensuring a smooth transition and in avoiding surprises if your container have some other unusual settings.

now, let's tackle a slightly advanced scenario. say your container doesn’t have some utility that vscode needs. what do you do? you can use the `postCreateCommand` key to execute commands after vscode connects to the container, for example, installing some needed tools.

```json
{
    "name": "existing-container-dev",
    "container": "my-preexisting-container",
	"workspaceFolder": "/src",
    "customizations": {
		"vscode": {
			"extensions": [
				"ms-python.python"
			]
		}
    },
    "postCreateCommand": "pip3 install debugpy && apt-get update && apt-get install -y git",
    "remoteUser": "vscodeuser"
}
```

in this example we're doing a `pip3 install debugpy` and an `apt update` followed by the installation of `git`. the `&&` allows us to execute this commands in sequence. this is a powerful feature for adding configuration and tools that you might need on demand, so you do not need to rebuild your image every time you need a tool inside the container.

one thing i want to mention here is that these commands are executed every time you start vscode, so if your command takes to long to complete, or is not meant to be ran each time, then you should use docker to bake it in directly into your image. for example, you can create a dockerfile and install these dependencies by using something like `run pip3 install debugpy` and then use that to run your container instead of an already existing container with the command line. remember that every situation is different, and you need to consider the trade-offs of each one.

a couple of things that i learned through errors and some debugging was the following: be meticulous with your folder mappings; always double-check the `workspaceFolder` location because having that misconfigured can cause some headaches. also remember to install the vscode remote development extension pack, without them, you won’t be able to use the devcontainer functionality.

one more thing, if you start the devcontainer feature on vscode with an already existing container and your container is already running you must attach it instead of starting a new one, otherwise you will have 2 containers running which can cause problems such as port conflicts.

i’ve seen many developers tripped up by small permission issues or subtle container differences. it's usually better to start simple and build up to more complex setups gradually rather than dive into a complex configuration on the first go.

in the end, creating a devcontainer for vscode with an existing container is not that difficult, it just needs you to get the basics down and pay attention to details, i'm pretty sure that with this setup you will see yourself saving some precious development time. now, what is the favorite type of container? the one you ship! (don't worry i'm not a comedian, i just wanted to get that one out of the way)

now, if you want to explore this area even further, i’d suggest looking at these:
*   **docker documentation:** the official documentation is always a good place to understand how docker works and get a better understanding of how containers work and the interactions between them.
*   **vscode remote development documentation:** Microsoft has extensive documentation on vscode remote development and the workings of the `devcontainer.json` file, that's something that you should read, trust me.
*   **advanced docker for developers:** this paper goes through the many docker functionalities like multi-stage builds, custom networks, and volume management, also it has a section about how to debug running containers which i think will be helpful.
*   **container security:** reading about security is very important, so i recommend you to read about secure container practices, because containerizing your development has a lot of security implications to consider.

i think that will do it for now. if you have any other questions or problems, feel free to ask, there is a whole community waiting to help you.
