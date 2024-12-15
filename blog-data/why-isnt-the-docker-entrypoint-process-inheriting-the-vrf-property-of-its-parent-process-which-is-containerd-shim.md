---
title: "Why isn't the Docker ENTRYPOINT process inheriting the vrf property of its parent process which is containerd-shim?"
date: "2024-12-15"
id: "why-isnt-the-docker-entrypoint-process-inheriting-the-vrf-property-of-its-parent-process-which-is-containerd-shim"
---

alright, so, this is a classic headache, and i've spent way too many late nights staring at logs trying to figure this exact thing out. it's one of those docker quirks that isn't immediately obvious, and it really hits you when you're trying to do something a little more involved than just running a simple webserver. let me break it down from my experience.

you’re seeing that the *vrf* property, which is set at the *containerd-shim* level, isn't being passed down to the *entrypoint* process in your docker container. this boils down to how process isolation and namespaces work in linux, and how docker and containerd manage container lifecycles.

basically, *containerd-shim* is the direct child of *containerd*, acting as the immediate parent of your container process. it’s responsible for setting up the necessary namespaces and control groups for your container to be isolated from the host and other containers, this includes things like the network, user, and mount namespaces and other security settings. *containerd-shim* sets up everything before the actual process inside the container starts, including any environment variables and other settings needed for the container to function. and indeed *vrf* seems to be one of them.

the *entrypoint* process you specify in your dockerfile, it's not a direct descendant of *containerd-shim*. the docker runtime uses *execve* internally (or similar functionality) to start your process within those already created namespaces and cgroups. *execve*, replaces the current process with the new one, instead of creating a new process like a fork. what this means for your problem is that things like *vrf*, are not automatically inherited unless you take specific actions to pass them down, as *execve* does not pass most attributes and file descriptors (other than the 0,1,2 file descriptor streams) down from parent to child by default, it is a totally brand new process even though it is within the same namespaces. This is by design, ensuring process isolation and security within a container runtime.

here's where i personally tripped up the first time. i assumed the *entrypoint* would magically inherit all the environment and security properties of its immediate parent process, *containerd-shim*. i tried a bunch of things, thinking there was a configuration that i was missing but no dice. i then started looking into the linux kernel itself to understand how this happens.

let’s look at it from a code perspective, using a simple example of a dockerfile:

```dockerfile
from alpine:latest
run apk add --no-cache bash
entrypoint ["/bin/bash", "-c", "echo vrf: $VRF; while true; do sleep 1; done"]
```

now let's start the docker container with the *vrf* environment variable.

```bash
VRF=myvrf docker run --rm -it my-image
```

if you then try to inspect the environment within the container using a command such as `env` or simply observe the output of the *entrypoint* command itself you will not see that *vrf* is there.

you'll find that the *entrypoint* process doesn’t pick up the `VRF` environment variable if you only define it on `docker run`. instead, we can inject the `VRF` variable by passing it directly to docker runtime environment:

```dockerfile
from alpine:latest
run apk add --no-cache bash
env VRF="default-vrf"
entrypoint ["/bin/bash", "-c", "echo vrf: $VRF; while true; do sleep 1; done"]
```

now rebuilding the container and running it:

```bash
VRF=myvrf docker run --rm -it my-image
```
now the echo output will say `vrf: default-vrf`. even if you are passing that variable on docker run command it will not pick it up. that is because docker will only inject the variable during build stage or as runtime environment variable not as a system environment variable passed by the operating system that was set during `docker run` on shell environment.

that said, the fact that you are using a variable named *vrf* and *containerd-shim* makes me think that you are dealing with network vrf's which are a special case, since usually people are not aware of this kind of issue. so, let's assume you are working on network namespaces, so, in that specific case *vrf* is not an environment variable but is a setting that goes with the network namespace. when the docker runtime starts the container, it does not pass the *vrf* setting to the process running in the container's network namespace. the process in container will not inherit the *vrf* attribute of the parent network namespace by default, even if it is within the same container networking context.

here is the second example, a more elaborate solution using a shell wrapper:

```dockerfile
from alpine:latest
run apk add --no-cache iproute2 bash
copy entrypoint.sh /
entrypoint ["/entrypoint.sh"]
```

and here is the `/entrypoint.sh` file:

```bash
#!/bin/bash

# Check for an overridden VRF, if not default to 'main'
VRF="${VRF:-main}"

# Create vrf if it does not exist.
ip vrf show "$VRF" || ip vrf add "$VRF"

# Assign the primary interface to the VRF
ip link set dev eth0 master "$VRF"

# bring the vrf up
ip link set "$VRF" up

# now run the entrypoint command, or the container payload
echo "vrf: $VRF"
exec "$@"
```

now, let's build this image and run it:

```bash
docker build -t my-vrf-image .
docker run --rm -it --net host my-vrf-image bash
```

*note:* to make the network namespace work correctly, you must run the container with host networking by using the flag `--net host`.

in this example, the *entrypoint.sh* script sets up the *vrf* based on the environment variable `VRF`, if it is not specified it defaults to `main`. it ensures that the vrf interface exists and moves the container's interface to that vrf. if you did not add an *entrypoint.sh* you would be running your `bash` process in the original container network namespace which does not have the required vrf settings.

this is a bit hacky and not always the best approach as it tightly couples your container to network namespace details, but i've seen this approach used a lot when you need to do specific networking setups within docker containers.

it is a process that involves setting up *vrf* at the container level and ensuring that processes inside the container can use it. if you pass the environment variable `VRF` during runtime such as: `VRF=myvrf docker run --rm -it --net host my-vrf-image bash` the script will pick it up and create the correct interface setup for you. it may be that containerd-shim has this type of vrf attribute set but your container will not be aware of it because it is not passed down during execution using *execve* command. the solution is to create the *vrf* inside the container context during startup.

now, this leads to a key takeaway here: you often need to explicitly manage environment settings and network configuration, if required, within your *entrypoint* script, or use the docker provided environment variables when possible, as well as using docker network specific features such as subnets and network profiles. relying on automatic inheritance from *containerd-shim* just doesn't work the way we hope.

finally, here’s one more approach, but this time it's not based on the bash wrapper but rather it uses *docker network connect*. it might be suitable if you do not want to alter the *entrypoint*. in this case you should define an external docker network with the required *vrf* configurations, and then attach the container to that network. here is a small example:

first create the network:

```bash
docker network create -d bridge --opt com.docker.network.bridge.name=br-myvrf --opt vrf=myvrf myvrf-network
```

and then run the container using it:

```bash
docker run --rm -it --network myvrf-network alpine:latest sh
```

this will now run the container using the *myvrf-network* which has the *vrf* setting we want. you can even make that network persistent so the container would reuse it after restarts.

here's a snippet that also shows how to remove this network:
```bash
docker network rm myvrf-network
```

again, you may not see the *vrf* attribute directly in any of the process's environment variables, but the process will be running inside the context of that network and it will function as if the *vrf* is enabled since it is part of the network profile itself. this is more of a docker-native way to handle network configurations instead of relying on shell scripts to do the job for you. this has the advantage that you do not need to alter the *entrypoint* at all and it is more portable.

to really get a solid grasp on this, you should look into the linux kernel documentation, particularly sections on namespaces, cgroups, and the *execve* syscall. the *containerd* and *docker* documentation are also crucial. i've personally found "linux system programming" by robert love to be a good resource to look into such concepts. it is dense, but it's worth the effort if you want to dive deep into how process isolation works in linux. also, the man pages for *execve* and *namespaces*(7) are pure gold, and that's it, you can even cite me on it.

in summary, the *vrf* attribute (or any similar inherited property that are set at *containerd-shim* level) does not automatically trickle down to the *entrypoint* process because of how *execve* (or equivalent) works under the hood, and how container runtimes are implemented. you need to manage these aspects manually during container startup. i hope this helps, i know i had the same headache a while ago, it got me going.
