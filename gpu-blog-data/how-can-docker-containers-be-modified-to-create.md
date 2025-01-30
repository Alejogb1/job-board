---
title: "How can Docker containers be modified to create new images?"
date: "2025-01-30"
id: "how-can-docker-containers-be-modified-to-create"
---
Modifying a running Docker container to generate a new image represents a practical workflow for iterative development and specific configuration adjustments. Directly altering a container, rather than rebuilding an image from scratch, can save time and computational resources, especially when troubleshooting or fine-tuning an application's environment. However, it is crucial to understand that these changes are not persistent across container restarts unless explicitly committed to a new image.

The core mechanism for converting modifications within a container into a new image is the `docker commit` command. When a container is launched from an image, a read-write layer is added on top of the immutable image layers. Operations performed inside the container modify this read-write layer. `docker commit` takes a snapshot of this layer and saves it as a new image, preserving those alterations. The process is akin to saving a file after making changes. The new image shares the original image's base layers, promoting efficiency in storage and deployment. It is also important to know that this approach should be used judiciously, particularly with large or complex containers. This method of directly altering containers to create new images should be seen as a convenient, iterative process, rather than the standard for producing final, production-ready images.

The `docker commit` command requires at least one argument: the container ID or name of the source container. You may also add several options, including a target image name and an optional tag, which effectively creates a label of that image. Consider the following scenario: I had a simple Python application running inside a container which required a particular package I hadn't included in the initial Dockerfile.

```dockerfile
# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY app.py .

# Run the application
CMD ["python", "app.py"]
```

This is a standard Dockerfile that sets up a Python environment, copies the file `app.py` and runs the application when a container is started. Let’s say the `app.py` file looks like this:

```python
import os
print("Hello, World!")
```

The first stage involves building the initial image and running the container:

```bash
docker build -t my_python_app .
docker run -d --name my_container my_python_app
```

This generates the image with the name `my_python_app` and starts the container named `my_container` in detached mode. Suppose I later determine that the application requires the `requests` library. I can use `docker exec` to install the library inside the running container:

```bash
docker exec -it my_container pip install requests
```

This command starts an interactive terminal session inside the running container where I use pip to install the requests library. The installed package is saved to the writable layer of the container. The next step uses `docker commit` to save the changes as a new image:

```bash
docker commit my_container my_python_app_updated:v1
```

This creates a new image named `my_python_app_updated` with a tag `v1`. The newly created image now contains the `requests` library. Running a container from `my_python_app_updated:v1` will immediately have the `requests` library available without running another `pip install` command.

Another common use case occurs when environment variables need modification or new ones introduced. For instance, an application might rely on a configuration file that gets copied into the image. After running the container I discovered a change is necessary. Let's start with a simplified Dockerfile and a sample config file:

```dockerfile
# Base image
FROM ubuntu:latest

# Create directory for configuration
RUN mkdir /config

# Copy configuration file
COPY config.txt /config/config.txt

# Command to start a simple process
CMD ["cat", "/config/config.txt"]
```

The `config.txt` file consists of a single line:

```
MODE=development
```

We build the image and run it as before:

```bash
docker build -t config_app .
docker run --name config_container config_app
```

This will print the content of config file to standard output. Now I decide that I need to change the configuration to production mode directly inside the running container. I can use `echo` and output redirection to modify the config file:

```bash
docker exec -it config_container bash
echo "MODE=production" > /config/config.txt
exit
```

After using `docker exec` to get shell access, I updated the config file and then exit the shell. To save this change, I will commit it as a new image:

```bash
docker commit config_container config_app_updated:v1
```

The new image `config_app_updated:v1` now contains the modified config file. Running a container from this new image will reflect this change. Finally, sometimes I have debug applications by adding utilities or altering binaries. For example, let’s say I am running a complex Go application inside a container:

```dockerfile
FROM golang:1.20-alpine as builder

WORKDIR /app

COPY main.go .

RUN go build -o my_app .

FROM alpine:latest

COPY --from=builder /app/my_app /app/my_app

CMD ["/app/my_app"]
```

Here is a simple Go main.go file for this example:
```go
package main
import "fmt"
func main() {
	fmt.Println("Running my go app")
}
```
This is a multi stage build that compiles the go program and copies the binary to a smaller alpine base image. We build and run the application as before:

```bash
docker build -t go_app .
docker run --name go_container go_app
```

Suppose I need to install a debugger like `gdb` inside the container to debug some unusual behavior that I've seen. I can execute this command inside the running container:

```bash
docker exec -it go_container apk add gdb
```

To save the container state, I commit the changes to a new image:

```bash
docker commit go_container go_app_debug:v1
```

The new image now has `gdb` installed. It’s important to remember that the new image will include `gdb`, even though it was installed after the fact inside the container. This approach is useful when the specific setup of an environment is difficult to predict or it needs to be rapidly adjusted.

While `docker commit` offers a method to create new images, this approach is not always the best method for creating long-lived production images. The primary limitation is its lack of repeatability. Since the build process is manual within the container, it's difficult to reconstruct the exact steps taken to modify the container to generate the new image. This contradicts the principles of immutable infrastructure and continuous integration. For production environments, the preferred approach is to modify the Dockerfile and rebuild the image. This ensures that the build process is repeatable, traceable, and manageable through version control.

For those aiming to delve further into managing Docker images, there are several excellent resources. The official Docker documentation provides detailed information on image creation, management, and optimization. Online courses on containerization platforms frequently include modules on image layering and best practices. Books such as “Docker in Action” and "The Docker Book" offer practical insights into working with Docker containers and images. Finally, many tech blogs by well-known engineers and consultants offer practical advice on handling images in real-world scenarios, such as continuous integration pipelines. Studying these resources will allow developers to choose the correct approach when making images, whether it’s iterative commits or fully repeatable Dockerfiles.
