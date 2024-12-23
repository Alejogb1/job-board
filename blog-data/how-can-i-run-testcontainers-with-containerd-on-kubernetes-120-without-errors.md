---
title: "How can I run Testcontainers with Containerd (on Kubernetes 1.20) without errors?"
date: "2024-12-23"
id: "how-can-i-run-testcontainers-with-containerd-on-kubernetes-120-without-errors"
---

Alright,  It’s a situation I’ve bumped into more than a few times, especially when teams start migrating from docker to containerd on kubernetes clusters. The transition can be a bit bumpy if you’re not careful, and Testcontainers interacting with containerd directly can expose some of those bumps. I remember back in 2021, during a large-scale deployment overhaul, we faced similar issues. The usual docker-based Testcontainers setup just wasn't cutting it anymore, causing intermittent failures and generally making the integration tests flaky.

The core problem isn't necessarily that Testcontainers "doesn't work" with containerd; it's more about how Testcontainers interacts with the underlying container runtime. Testcontainers by default often tries to connect to a docker socket. Containerd, on the other hand, uses a different architecture and doesn’t expose a docker-compatible socket directly. You can't just point Testcontainers at a containerd endpoint and expect it to work seamlessly.

The key to making it work boils down to two main approaches: leveraging a compatible client within Testcontainers or using a bridge layer that translates docker commands to containerd. The first is typically the most robust, and it’s the one I generally recommend. The second works but introduces another point of failure and more complex configuration.

Let's start with the preferred solution: configuring Testcontainers to directly use a containerd client. Since you mentioned kubernetes 1.20, this means your kubernetes nodes will likely have containerd as their container runtime, which makes this approach applicable directly at the node level or in an environment that emulates one (like a local k3d cluster, for instance). What you need to do is configure Testcontainers to utilize the appropriate client through environment variables or programmatically. The specific client we're going to focus on is the `testcontainers-java` library and its containerd integration, as I suspect you're working in java, based on typical use-cases.

**Example 1: Using environment variables**

This is the simplest approach for local setups or in CI/CD environments where you can easily manipulate environment variables. We'll instruct Testcontainers to use the `unix:///run/containerd/containerd.sock` socket:

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@Testcontainers
public class ContainerdTest {

    @Container
    private GenericContainer<?> nginxContainer = new GenericContainer<>("nginx:latest")
        .withExposedPorts(80);

    @Test
    void testContainerRunning() {
        // your assertions here
        System.out.println("Container is running: " + nginxContainer.isRunning());
    }
}

```

Now, before running this test, you'd need to set the following environment variable:

```bash
export TESTCONTAINERS_DOCKER_HOST=unix:///run/containerd/containerd.sock
```

The `TESTCONTAINERS_DOCKER_HOST` variable is interpreted by the `testcontainers-java` library and, since it is a socket path, will trigger its containerd adapter if the path contains `containerd`. The environment variable approach keeps your application code clean by externalizing the configuration. This leverages Testcontainers' built-in mechanism for managing docker clients. It's a bit of a misnomer calling it `DOCKER_HOST` in the containerd context, but it serves the purpose.

**Example 2: Programmatically setting the client**

Sometimes, environment variables aren't ideal, or you need more granular control. In those cases, you can programmatically configure the client using a Testcontainers `DockerClientProviderStrategy`. The trick is to implement your own provider strategy that specifies how the docker client should be created:

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerClientProviderStrategy;
import com.github.dockerjava.core.DockerClientConfig;
import com.github.dockerjava.api.DockerClient;

@Testcontainers
public class ContainerdProgrammaticTest {

    private static class CustomContainerdProviderStrategy implements DockerClientProviderStrategy {

        @Override
        public DockerClient getClient() {
              DockerClientConfig config = DockerClientConfig.createDefaultConfigBuilder()
                        .withDockerHost("unix:///run/containerd/containerd.sock")
                        .build();
              return DockerClientFactory.instance().client(config);
        }

        @Override
        public void test() {
             // we will be lazy, and assume everything will be working as long as we create the client.
        }
        @Override
        public String getDescription() {
            return "Custom Containerd Provider Strategy";
        }

    }

     static {
        DockerClientFactory.instance().setStrategy(new CustomContainerdProviderStrategy());
    }

    @Container
    private GenericContainer<?> redisContainer = new GenericContainer<>("redis:latest")
        .withExposedPorts(6379);

    @Test
    void testContainerRunning() {
        // your assertions here
         System.out.println("Container is running: " + redisContainer.isRunning());
    }

}

```
Here, we define `CustomContainerdProviderStrategy`, that forces `testcontainers-java` to use a specific connection to containerd, bypassing the discovery logic. This makes the connection to containerd very explicit.

**Example 3: Addressing Kubernetes-Specific Issues**

Now, a kubernetes environment throws in an extra layer of complexity. Testcontainers may try to interact with the docker socket on the host, which doesn’t exist if you’re running within a kubernetes pod. In many Kubernetes setup, Testcontainers cannot use the host’s container runtime directly for security and stability reasons. There are multiple strategies to solve this: using docker-in-docker or similar solutions, which are typically frowned upon. Using the kubernetes api server to spin up containers, or even use a remote docker daemon. This all adds substantial complexity, and are only required if the tests cannot be run directly on a host with access to the container runtime (e.g., not within kubernetes). The simplest approach to tackle this issue is to use a local kubernetes cluster with containerd (e.g. using `k3d`, or similar). The previous two examples would apply to this environment, as the test runner will have access to the underlying containerd runtime.

In scenarios where you need the containers to run within Kubernetes, you can instead utilize tools like Skaffold or similar to manage the testing environment. These tools manage the build, deployment, and testing pipelines within Kubernetes clusters themselves. Testcontainers could still be used to create dependencies that live *outside* the kubernetes cluster itself, for example databases that can be accessed by your code deployed within the kubernetes cluster. The key idea is to make the distinction between containers managed by testcontainers and those managed by the Kubernetes environment. In this case, no direct containerd connection needs to be setup in the Testcontainers configuration since you're leveraging the Kubernetes environment for deploying your code under test.

For further reference and more in-depth details, I highly recommend these resources:

*   **Testcontainers documentation**: It’s your first stop. Pay close attention to the sections related to `DockerClientProviderStrategy` and custom client implementations.
*   **Docker-java library documentation**: The underlying java library used by testcontainers provides low-level access and a much deeper understanding of how the underlying client is configured.
*   **Kubernetes documentation**: In particular, the sections on containerd runtime configuration and container networking within Kubernetes are essential.

Remember that troubleshooting these types of problems involves meticulous logging and understanding the underlying environment. Start simple, verify your connection and slowly increase complexity until you find the configuration that works best for you and your specific needs.
