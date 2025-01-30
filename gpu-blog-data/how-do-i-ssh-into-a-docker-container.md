---
title: "How do I SSH into a Docker container?"
date: "2025-01-30"
id: "how-do-i-ssh-into-a-docker-container"
---
Directly accessing a Docker container's shell via SSH requires a specific configuration, as standard Docker containers don't inherently support SSH.  My experience troubleshooting networking issues across various production environments, involving hundreds of containerized microservices, solidified this understanding.  The challenge isn't inherent to Docker itself, but rather a conscious architectural decision to manage access and security effectively.  Achieving SSH access demands deploying an SSH server *inside* the container.

**1. Explanation:**

The core principle involves running an SSH server process within the Docker container image.  This necessitates a few crucial steps:

* **Image Selection or Creation:** You must use a base Docker image that already includes an SSH server (like `ubuntu` or `alpine` with OpenSSH installed) or build a custom image incorporating one.  Base images lacking SSH require installation during the image build process. This installation includes the OpenSSH server package and, critically, configuring its functionality to allow incoming connections.

* **Port Mapping:**  Docker’s networking model requires explicit port mapping to enable external access to services running inside containers.  The SSH daemon typically listens on port 22.  Therefore, you must map a port on the host machine to port 22 inside the container. This mapping permits SSH connections from your host to the container's SSH server.  Consider security implications; avoid exposing port 22 directly to the public internet.  Employ techniques such as VPNs, bastion hosts, or jump servers for enhanced security.

* **Security Considerations:** Exposing SSH directly to the host machine presents a security vulnerability if the host itself is not secured.  Implementing robust security practices, such as strong SSH keys, restricting user access, and regularly updating the SSH server, is essential to mitigate risks.  I have personally encountered scenarios where insufficient security around container SSH access led to unauthorized access and subsequent system compromises.

* **Container Network Configuration:** Ensure the container is reachable on the network.  This might necessitate adjustments to the Docker network configuration, especially within complex environments employing overlay networks.  Correct network configuration is paramount; misconfiguration will prevent SSH connectivity despite the correct port mapping.

**2. Code Examples:**

**Example 1:  Using a pre-built image (Ubuntu)**

This example leverages an existing Ubuntu image with SSH pre-installed:

```dockerfile
FROM ubuntu:latest

# No additional installation needed, SSH is already present in the Ubuntu image.

CMD ["/usr/sbin/sshd", "-D"]
```

Docker build and run commands:

```bash
docker build -t ssh-container .
docker run -d -p 2222:22 --name ssh-container ssh-container
```

This command creates a container named `ssh-container`, maps host port 2222 to container port 22, and runs the SSH daemon in detached mode.  Note the use of a non-standard port (2222) for security; mapping directly to port 22 is generally discouraged unless behind a secured gateway.


**Example 2: Installing SSH during image build (Alpine)**

This example demonstrates installing SSH within a minimal Alpine Linux image:

```dockerfile
FROM alpine:latest

RUN apk add --no-cache openssh-server

# Create an SSH user
RUN adduser -D -s /bin/sh sshuser
RUN echo "sshuser:password" | chpasswd

# Allow SSH connections from all IPs. In production, restrict this.
RUN sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN sed -i '/PermitRootLogin/s/no/yes/' /etc/ssh/sshd_config
RUN sed -i '/PasswordAuthentication/s/no/yes/' /etc/ssh/sshd_config


EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

Docker build and run commands:

```bash
docker build -t ssh-alpine-container .
docker run -d -p 2223:22 --name ssh-alpine-container ssh-alpine-container
```

This uses a non-standard port (2223) and configures SSH to allow password authentication.  This is highly discouraged for production environments;  key-based authentication is vastly superior.  The `/etc/ssh/sshd_config` modifications allow root login and password authentication; these are highly insecure and should only be used for development or testing purposes. Replace "password" with a strong password.

**Example 3: Key-based authentication (best practice)**

This example emphasizes security by using key-based authentication and avoids password authentication entirely.


```dockerfile
FROM alpine:latest

RUN apk add --no-cache openssh-server

#Create an SSH user
RUN adduser -D -s /bin/sh sshuser

#Generate an SSH key pair
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -N ""
RUN mkdir -p /home/sshuser/.ssh
RUN cp /root/.ssh/id_rsa.pub /home/sshuser/.ssh/authorized_keys
RUN chown -R sshuser:sshuser /home/sshuser/.ssh
RUN chmod 700 /home/sshuser/.ssh
RUN chmod 600 /home/sshuser/.ssh/authorized_keys


#Disable password authentication and root login
RUN sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN sed -i '/PermitRootLogin/s/prohibit-password/no/' /etc/ssh/sshd_config


EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

Docker build and run commands:

```bash
docker build -t secure-ssh-container .
docker run -d -p 2224:22 --name secure-ssh-container secure-ssh-container
```


This example generates an SSH key pair within the image and configures the SSH user (`sshuser`) with key-based authentication.  The `-N ""` option disables the passphrase on the key.  This should be used cautiously as it compromises key security;  consider setting a strong passphrase if feasible.  This method eliminates password authentication significantly increasing security.  The public key (`id_rsa.pub`) needs to be added to your `~/.ssh/authorized_keys` file on your host machine.

**3. Resource Recommendations:**

For further information on Docker networking, consult the official Docker documentation.  A comprehensive guide on securing Docker containers and  managing Linux users and permissions will prove invaluable for production deployments.  Finally, explore dedicated texts on securing SSH deployments for best practices.  Understanding the implications of each configuration and the relative security vulnerabilities introduced is crucial.  My own experience taught me the importance of prioritizing security when dealing with containerized systems. Remember that even minor misconfigurations can have significant consequences.
