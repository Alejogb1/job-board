---
title: "Why can't TestContainer find the Docker daemon under Lima?"
date: "2024-12-23"
id: "why-cant-testcontainer-find-the-docker-daemon-under-lima"
---

Okay, let's tackle this. It's a situation I've certainly encountered enough times to have a few battle scars. The problem of Testcontainers seemingly failing to locate the Docker daemon when running under Lima is less about a straightforward bug and more about the intricacies of how these tools interact with virtualized environments. Fundamentally, Testcontainers relies on being able to communicate directly with the Docker daemon, and Lima, acting as a lightweight virtualization solution, creates a layer of abstraction that can disrupt this direct communication path.

The core issue stems from the way Lima sets up its virtual machine (VM) and exposes the Docker daemon. Lima typically operates by creating a separate Docker environment *inside* its VM. This is crucial: the Docker daemon accessible inside the Lima VM is not the same as the Docker daemon running on your host operating system, which is usually where Testcontainers expects to find it. Consequently, your tests executing outside of the Lima VM cannot directly connect to the Docker daemon running within. It's a network boundary problem primarily.

In my past project, we had a similar setup for CI/CD using GitHub Actions with macos runners, which defaulted to Lima for Docker. The initial failures were perplexing until we mapped out the communication flow. Testcontainers, by default, tries to connect to the Docker daemon using environment variables such as `DOCKER_HOST` or through a socket file like `/var/run/docker.sock`. When using Lima, these are often pointing to the host's daemon, which, again, is not the one actually running within the Lima VM.

There are a few strategies I found effective in resolving this. The primary one revolves around ensuring Testcontainers connects to the *correct* Docker daemon – the one *inside* the Lima VM. We can accomplish this by either reconfiguring `DOCKER_HOST` to point inside the lima VM, or by using a socat bridge.

Here's the first code snippet, illustrating how to set the `DOCKER_HOST` environment variable dynamically, which I used successfully on my previous project.

```java
import org.testcontainers.DockerClientFactory;
import java.util.Optional;

public class LimaDockerSetup {

    public static void configureForLima() {
        // Check if we are likely inside a Lima environment
        String limaInstance = System.getenv("LIMA_INSTANCE");
        if (limaInstance != null && !limaInstance.isEmpty()) {
          // Get the lima docker host socket path
          Optional<String> dockerHostFromEnv = Optional.ofNullable(System.getenv("DOCKER_HOST"));
          if(dockerHostFromEnv.isPresent()){
              System.out.println("DOCKER_HOST is set to "+ dockerHostFromEnv.get() + ". Skipping configuration");
              return;
          }
          // Extract the lima instance name
          String limaHome = System.getenv("HOME");
          if(limaHome == null || limaHome.isEmpty()){
              System.out.println("Home directory not found. Skipping lima configuration");
              return;
          }
          // Build docker socket path
           String dockerHost =  String.format("unix://%s/.lima/%s/docker.sock", limaHome, limaInstance);
           System.setProperty("testcontainers.docker.host", dockerHost);
           System.out.println("testcontainers.docker.host set to " + dockerHost);

           //Explicitly set it in case `DOCKER_HOST` variable is checked
           System.setProperty("DOCKER_HOST", dockerHost);
           System.out.println("DOCKER_HOST system variable set to " + dockerHost);

           //Force initialization of docker client
           DockerClientFactory.instance().client();
        }
        else{
            System.out.println("Not in a Lima environment. Skipping configuration");
        }
    }

   public static void main(String[] args) {
        configureForLima();
    }
}
```

This java snippet checks for the existence of a `LIMA_INSTANCE` environment variable, which is usually present in lima environments. If found, it attempts to dynamically build the path to the docker socket inside lima's VM and sets the `testcontainers.docker.host` and `DOCKER_HOST` system properties to use this socket. This will then be read by testcontainers to find the docker daemon. Note that if the system is already configured using the `DOCKER_HOST` environment variable, this snippet does not attempt to overwrite the already configured value. This method provides a more portable solution, avoiding hardcoded paths that might differ slightly on various setups.

Another method I have used involves setting up a `socat` bridge. This is especially helpful when you need to use other tools that cannot be configured as easily as Testcontainers. It's a bit more involved, but offers greater flexibility. Below is a bash snippet demonstrating how to establish a `socat` bridge. It basically forwards a port on the host machine to the docker socket within the lima VM.

```bash
#!/bin/bash

# Get the lima instance name
LIMA_INSTANCE=$(printenv LIMA_INSTANCE)

if [[ -z "$LIMA_INSTANCE" ]]; then
  echo "Not running in a Lima instance, skipping socat bridge setup."
  exit 0
fi

# Get the lima home directory
LIMA_HOME=$(printenv HOME)

# Build the path to the docker socket inside lima
LIMA_DOCKER_SOCKET="$LIMA_HOME/.lima/$LIMA_INSTANCE/docker.sock"

# Find a free port
FREE_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Start socat to create a TCP bridge on localhost:FREE_PORT
echo "Starting socat bridge on localhost:$FREE_PORT -> $LIMA_DOCKER_SOCKET"

socat TCP4-LISTEN:$FREE_PORT,fork UNIX-CONNECT:$LIMA_DOCKER_SOCKET &

# Store the bridge port
export LIMA_BRIDGE_PORT=$FREE_PORT

echo "Socat bridge set up on localhost:$LIMA_BRIDGE_PORT"
echo "Use the DOCKER_HOST environment variable to use the bridge"

# Give the socat time to set up
sleep 1

# Example usage (uncomment if needed)
#export DOCKER_HOST=tcp://localhost:$LIMA_BRIDGE_PORT
#docker ps

```

This script first checks if we’re in a Lima environment. Then, it finds a free port on the host machine and starts a `socat` process that forwards traffic from that port to the Docker socket inside the Lima VM. Now, any application wanting to talk to the Docker daemon can do so using `tcp://localhost:$LIMA_BRIDGE_PORT`. The script exports the port so that you can set `DOCKER_HOST` to this value in your tests. The `sleep 1` command gives socat a small window to start the forwarding before exiting the script. This is crucial as if the script exists before socat successfully starts, the forwarding will fail. Note the script should be run on your host system prior to running your tests.

Finally, as a third approach, you could modify the testcontainers configuration using its `DockerHost` class. This is a more programmatic way to set things up if you want to define custom socket communication, and it's typically used for unusual edge-cases which require more fine-grained control over Docker communication.

```java
import org.testcontainers.DockerClientFactory;
import org.testcontainers.dockerclient.DockerClientConfigUtils;
import org.testcontainers.dockerclient.DockerClientProviderStrategy;
import org.testcontainers.dockerclient.DockerHostResolver;

import java.util.Optional;

public class LimaCustomDockerHostSetup {

    public static void configureCustomDockerHost(){
      // Check if we are likely inside a Lima environment
      String limaInstance = System.getenv("LIMA_INSTANCE");
      if (limaInstance != null && !limaInstance.isEmpty()) {
        String limaHome = System.getenv("HOME");
        if (limaHome == null || limaHome.isEmpty()) {
          System.out.println("Home directory not found. Skipping lima configuration");
          return;
        }

        String dockerSocketPath = String.format("unix://%s/.lima/%s/docker.sock", limaHome, limaInstance);

        DockerClientProviderStrategy strategy = DockerClientFactory.instance().getProviderStrategy();

        DockerHostResolver resolver = new DockerHostResolver(){
          @Override
          public Optional<DockerClientConfigUtils.DockerHost> resolveDockerHost() {
             return Optional.of(new DockerClientConfigUtils.DockerHost(dockerSocketPath, null, false, null));
          }
        };

        strategy.setDockerHostResolver(resolver);

        System.out.println("Custom docker host resolver registered for lima");

        //Force docker client initialization
        DockerClientFactory.instance().client();

      }else{
        System.out.println("Not in a Lima environment, skipping configuration");
      }
    }


  public static void main(String[] args) {
    configureCustomDockerHost();
  }
}
```

This java code intercepts the docker client resolution logic using a custom `DockerHostResolver`. It constructs the socket path similarly to the first code snippet and creates a `DockerClientConfigUtils.DockerHost` object, which Testcontainers uses to connect to the Docker daemon. By setting this resolver on the current strategy, we effectively override how Testcontainers attempts to connect to the docker daemon. Again, the code includes a check for the `LIMA_INSTANCE` variable and only sets up the custom resolver in a lima environment.

These are some of the methods I've had success with. As for further reading, I’d recommend “Docker Deep Dive” by Nigel Poulton for a comprehensive understanding of Docker internals, including networking. Also, the official Docker documentation is excellent, especially the sections on the Docker daemon and its socket communication. Understanding the Docker architecture and networking is key to troubleshooting these kinds of issues. The Testcontainers documentation also has a good section on custom docker hosts that's worth checking out.
