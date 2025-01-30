---
title: "What do Docker's `ps` output differences between `123-125/tcp` and `0.0.0.0:123-125->123-125/tcp` ports signify?"
date: "2025-01-30"
id: "what-do-dockers-ps-output-differences-between-123-125tcp"
---
The distinction in Docker `ps` output between `123-125/tcp` and `0.0.0.0:123-125->123-125/tcp` directly indicates how ports are mapped between the host machine and the container. The former signifies that the container exposes a range of ports internally, whereas the latter indicates a direct mapping where a host’s port range is explicitly linked to that internal range. Having spent several years managing containerized microservices, I’ve encountered this frequently and have found understanding the nuances to be critical for proper network configuration and troubleshooting.

Let’s first examine `123-125/tcp`. This output means that the Docker container is listening on ports 123, 124, and 125 within its internal network namespace. This specification does not, by itself, imply any direct accessibility from outside the container’s network. These ports are exposed for other containers on the same Docker network, or by applications running within the container, to communicate. Crucially, at this stage, they are not accessible via the host machine’s network interface. If we want external access, this must be explicitly established through port mapping. This configuration is often encountered when containers are designed to talk to each other within a cluster or when services are accessed through a reverse proxy operating on the same Docker network.

The output `0.0.0.0:123-125->123-125/tcp`, in contrast, signifies that a port mapping has been established. Specifically, the host machine’s network interface, represented by `0.0.0.0`, has been configured to forward incoming traffic on ports 123, 124, and 125 to the corresponding ports inside the container. The `->` arrow visually demonstrates the flow of traffic. This is a crucial distinction. Without port mapping, a container's exposed ports are only visible to other containers on the same network. This explicit mapping opens them up to the external world, allowing for communication from the host itself and other external networks that can reach the host. The use of `0.0.0.0` as the host address effectively means “listen on all available network interfaces of the host.” If, instead, a specific IP address were listed, that would indicate that the ports are only accessible from that particular interface of the host.

To illustrate this, consider a basic web server container.

```dockerfile
# Dockerfile for web server
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

```python
# app.py (example Python web app using Flask)
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```

Here, the `EXPOSE 8000` line in the Dockerfile designates that the container will internally listen on port 8000. The Flask application explicitly binds to port 8000 and address 0.0.0.0 inside the container. If I now build and run this image without explicit port mapping:

```bash
docker build -t my-web-app .
docker run -d --name my-container my-web-app
```

Running `docker ps` might then show output similar to: `8000/tcp`. The application within the container is listening on port 8000, but there is no external access granted. To make this service available from the host’s network interface, I must use the `-p` flag during the run command:

```bash
docker run -d -p 8080:8000 --name my-container my-web-app
```

The `docker ps` output will then show something akin to `0.0.0.0:8080->8000/tcp`. This tells us that traffic sent to port 8080 on the host will be forwarded to port 8000 inside the container. In this example I am mapping port 8080 of my host to port 8000 inside the container. If I want to make the host port the same as the container port, the run command could look like this:

```bash
docker run -d -p 8000:8000 --name my-container my-web-app
```

And `docker ps` will then show: `0.0.0.0:8000->8000/tcp`. Now I can access my service running in the container, from my host, by pointing a web browser to `localhost:8000`.

Now consider a more complex scenario using a range of ports. Suppose a custom service requires a range of ports for communication.

```dockerfile
FROM alpine:latest
RUN apk add --no-cache tcpdump
EXPOSE 10000-10002
CMD ["sh", "-c", "while true; do tcpdump -i any port 10000 or port 10001 or port 10002; sleep 1; done"]
```

Building and running this image as:

```bash
docker build -t range-test .
docker run -d --name range-container range-test
```

The output of `docker ps` would be: `10000-10002/tcp`, reflecting that the container internally uses this range. Now let's map the range to the same ports on the host:

```bash
docker run -d -p 10000-10002:10000-10002 --name range-container range-test
```

This results in the following `docker ps` output: `0.0.0.0:10000-10002->10000-10002/tcp`. The host can now send and receive data using ports 10000, 10001, and 10002, and that data will be forwarded to the corresponding ports inside the container. The `tcpdump` command, which is running in the container, would then pick up any traffic destined for those ports.

Finally, if a specific IP on the host is being used, consider this:

```bash
docker run -d -p 192.168.1.10:9000:9000 --name specific-ip-container my-web-app
```

In this case, `docker ps` would output something like `192.168.1.10:9000->9000/tcp`. This indicates that the port 9000 on the container is accessible through the host’s IP address `192.168.1.10` on port 9000, rather than on all available network interfaces as represented by `0.0.0.0`.

In summary, the presence of the `0.0.0.0:` prefix along with the `->` indicator in the `docker ps` output reveals explicit host-to-container port mapping. Without it, the ports are exposed within the container, but remain isolated from the host's network.  Understanding these subtle but crucial differences is fundamental when working with Docker, especially when dealing with more complex setups such as multi-container applications.

For further exploration of this topic, I suggest consulting the official Docker documentation, particularly the sections on network configuration and port mapping.  Additionally, resources on container networking best practices often provide valuable insights.  Lastly, any comprehensive guide on Docker fundamentals should include clear explanations regarding exposed vs. mapped ports.
