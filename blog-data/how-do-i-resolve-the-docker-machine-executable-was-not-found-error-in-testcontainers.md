---
title: "How do I resolve the 'docker-machine executable was not found' error in Testcontainers?"
date: "2024-12-23"
id: "how-do-i-resolve-the-docker-machine-executable-was-not-found-error-in-testcontainers"
---

Ah, this familiar beast. I've encountered the "docker-machine executable was not found" error more times than I care to recall, particularly when setting up CI pipelines or complex test environments utilizing Testcontainers. Let's unpack what's going on and how to consistently resolve it. It's a frustrating error, but usually points to a pretty straightforward root cause: Testcontainers, in certain configurations, relies on `docker-machine` to manage Docker environments, and if it can't locate this executable, it'll throw that specific error. It's less common these days as Docker Desktop and other native solutions have matured, but it's definitely not extinct.

In my experience, this issue typically arises in two primary scenarios: older systems or development setups where the environment hasn’t fully transitioned to the modern, native Docker tooling, or when Testcontainers is explicitly configured to use the `docker-machine` driver, perhaps unintentionally. The default setting, where it uses the docker daemon directly, avoids this, however there may be reasons or specific conditions where one may want to rely on `docker-machine`.

The core problem isn’t that your Docker installation is faulty (usually), but rather that Testcontainers can’t locate the `docker-machine` executable. It's important to remember that `docker-machine` is a separate tool, originally created to manage Docker environments on systems that did not directly support the Docker daemon itself, such as older MacOS versions and other platforms. It creates and manages virtualized Docker hosts. Therefore, to fix it, we need to ensure the `docker-machine` executable is actually installed and that its location is either in the system's path or explicitly specified to Testcontainers.

Let's dive into how we might address this, including some code snippets to show the actual implementation.

First and foremost, determine whether you actually need `docker-machine` at all. If you're working on a system where Docker Desktop (or an equivalent native solution) is running and properly configured, you almost certainly don't. Testcontainers can usually interface directly with the daemon. In this case, the fix is to simply ensure that no `docker-machine` specific settings are in place.

However, let's assume you *do* need to use `docker-machine`, either due to a legacy requirement or some specific infrastructure constraint. In that case, let's begin by confirming that the `docker-machine` executable is installed. The procedure varies depending on your operating system; if you are working on Mac or Linux, typically, you can get this via `brew install docker-machine` if using homebrew on MacOS, or from your package manager on Linux (it might be available in your distro as `docker-machine` package). Ensure it’s on your system's path using your OS specific path variable methods. Check `which docker-machine` or `where docker-machine` to see if it's accessible from your command line.

Once confirmed that the executable is installed, the first code snippet below shows how to configure Testcontainers to explicitly use the `docker-machine` driver, and specify the location of your docker machine binary, if it’s not in your system path. Note that you’ll replace `/usr/local/bin/docker-machine` with the path where your `docker-machine` executable resides. Also, you will need to make sure your `docker-machine` environment is properly set up to work before even launching Testcontainers.

```java
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

import java.io.File;
import java.nio.file.Paths;
import com.github.dockerjava.core.DefaultDockerClientConfig;
import com.github.dockerjava.core.DockerClientConfig;
import com.github.dockerjava.api.DockerClient;

public class DockerMachineExample {

    public static void main(String[] args) {
        DockerClientConfig config = DefaultDockerClientConfig.createDefaultConfigBuilder()
                .withDockerHost("tcp://192.168.99.100:2376") // Replace with your docker machine IP and port
                .withDockerCertPath(Paths.get(System.getProperty("user.home"), ".docker", "machine", "certs").toString()) // Replace if your cert path differs
                .build();

        DockerClient client = DockerClientFactory.instance().client(config);
        GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("nginx:latest"))
           .withExposedPorts(80);
        container.start();
        // do stuff with the container here
        container.stop();

    }
}
```

This example illustrates the manual configuration of the `DockerClient` to connect to a `docker-machine` instance. You'll want to replace the example docker host IP address and the cert path with your own values, typically these are found from running `docker-machine env <machine_name>` in your terminal. Using the client directly circumvents the driver detection logic of test containers and lets you directly manage docker connections. This setup avoids the necessity of Testcontainers discovering the executable.

Now, if you're finding that you don't actually need `docker-machine`, and you're running on a system where the Docker daemon is accessible directly (like Docker Desktop), we should instead ensure that we're not accidentally forcing Testcontainers to use the docker-machine driver.

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

public class DockerDesktopExample {

    public static void main(String[] args) {
        GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("nginx:latest"))
                .withExposedPorts(80);
        container.start();
         // do stuff with the container here
        container.stop();
    }
}
```

The above example demonstrates the simplest scenario, no particular driver configuration is specified, meaning Testcontainers will default to using the docker daemon directly. This generally works perfectly with Docker Desktop (or any native Docker solution). It's clean and doesn't require any manual `docker-machine` handling. In many cases, this is the correct and preferred approach.

Lastly, there are scenarios where you might be running test containers in a CI/CD environment where the environment variable `TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE` is set. If you have the environment variable `TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE` set to `unix:///var/run/docker.sock`, and you don't have a local docker daemon running, then test containers will error out, however not with the docker-machine missing error. In that case, you need to either remove that env variable or ensure the docker daemon is running locally. This variable was only introduced in Testcontainers 1.19.0 and later.

Here's an example of how you can configure the `DockerClientConfig` with a docker socket override.
```java

import com.github.dockerjava.core.DefaultDockerClientConfig;
import com.github.dockerjava.core.DockerClientConfig;
import com.github.dockerjava.api.DockerClient;
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;


public class DockerSocketOverrideExample {
    public static void main(String[] args) {
        DockerClientConfig config = DefaultDockerClientConfig.createDefaultConfigBuilder()
                .withDockerSocketOverride("unix:///var/run/docker.sock")
                .build();

        DockerClient client = DockerClientFactory.instance().client(config);

        GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("nginx:latest"))
                .withExposedPorts(80);

        container.start();
        // do stuff with the container here
        container.stop();
    }
}
```
This example demonstrates the case where you might need to set the socket override.

To sum it up, resolve "docker-machine executable was not found" by either ensuring `docker-machine` is installed and its path is configured properly (if required), or by ensuring your setup is using the docker daemon directly where applicable. It typically comes down to either configuring the driver to look for it correctly, or avoiding the driver altogether. For reference, I would highly recommend reading the official Testcontainers documentation. The "Docker environment setup" section is the one you'll need the most here, and also the general guide section on driver configurations for testcontainers. Also, the source code of the `DockerClientFactory` in the Testcontainers java library might provide further insights on its connection mechanism. Also, going through the docker-java client library is helpful to understand how the underlying client connection settings are made. This level of depth might be overkill, but sometimes, especially with tricky build issues, going deeper into the underlying mechanism becomes necessary. Finally, don't overcomplicate the setup if you don't need to use `docker-machine`. Often the most effective solution is to remove any dependencies on `docker-machine`.
