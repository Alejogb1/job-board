---
title: "Is it possible to run on Windows Server with Containers Docker-in-Docker?"
date: "2024-12-14"
id: "is-it-possible-to-run-on-windows-server-with-containers-docker-in-docker"
---

yes, it's totally possible to run docker-in-docker (dind) on windows server with containers, but it's not exactly a walk in the park, and i've definitely had some adventures getting it to behave properly.

let me lay it out for you. the core concept of dind, as you probably already grasp, involves launching a docker daemon *inside* a docker container. this is often needed for situations like ci/cd pipelines, where you build and deploy docker images from within a dockerized environment. it's recursive, a docker inside a docker, quite a inception, if i can use such a concept from a movie.

now, windows server throws a few extra curves into this process compared to linux. the biggest one revolves around the container isolation mode. you generally have two main options here: process isolation and hyper-v isolation.

process isolation shares the host kernel, which is simpler and faster, but it *doesn’t* play nice with dind. since the dind container needs its own kernel namespace to run its docker daemon, process isolation just won't cut it. that's usually the first thing i check when someone is having issues, process isolation is a real dind killer.

hyper-v isolation, on the other hand, provides a lightweight virtual machine for each container, which gives us the kernel-level separation needed for dind to work. this means we can run the docker daemon inside a hyper-v container without causing havoc on the host or other containers. it’s like having a little sandbox for your docker daemon.

i remember one particularly frustrating afternoon i had with this. i was setting up a ci/cd pipeline on a windows server build agent, and i was banging my head against the wall trying to get docker commands to work inside my agent container, which was in process isolation. after what felt like an eternity of troubleshooting, i finally realized my mistake: i wasn't using hyper-v isolation. a quick change in my docker-compose file and it all started working magically, or at least it felt that way. i learned my lesson to always double-check the container isolation level. i even put a reminder in my personal documentation, and that, i believe, is a must for all of us.

here's a simplified example of a `docker-compose.yml` file that should get you started with a dind setup on windows server, paying special attention to hyper-v isolation:

```yaml
version: "3.9"
services:
  dind:
    image: docker:24.0.7-dind  # use the specific version of docker to avoid unexpected behavior
    privileged: true  # this is required for dind to function correctly
    isolation: hyperv
    environment:
      DOCKER_TLS_CERTDIR: "/certs"
    volumes:
      - dind-data:/var/lib/docker  # persist docker data
      - certs:/certs
    ports:
      - 2376:2376

volumes:
  dind-data:
  certs:
```

a few things to point out here:

*   **`image: docker:24.0.7-dind`**: this is the official docker dind image, you should pick the specific tag as it will provide a more stable base to work.
*   **`privileged: true`**: this grants the dind container the necessary privileges to run the docker daemon. you’ll need it for dind to work, but bear in mind the security implications, specially if working with production environment.
*   **`isolation: hyperv`**: this is crucial for windows server. ensure that the container is running with hyper-v isolation. it’s the key to the whole thing.
*   **`volumes`**: these are used to persist the docker data and also the certificates.

after that `docker-compose up -d`, you should have a dind service running, and you can point your docker client at it:

```bash
docker -H tcp://localhost:2376 version
```

but keep in mind that you might need to set the docker host correctly, or configure your docker client to use certificates if you are using tls. and that takes us to the certificate part of this setup. usually when working with remote docker daemons, you’d enable tls for secure communication between the docker client and the daemon.

in the previous `docker-compose.yml` example, you'll see the `certs` volume, which is used to persist the certificates, so the following snippet shows how you’d generate the certificates:

```bash
openssl genrsa -out ca-key.pem 2048
openssl req -x509 -new -nodes -key ca-key.pem -sha256 -days 3650 -out ca.pem -subj "/CN=dind-ca"
openssl genrsa -out server-key.pem 2048
openssl req -new -key server-key.pem -out server.csr -subj "/CN=localhost"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -days 3650 -sha256
openssl genrsa -out client-key.pem 2048
openssl req -new -key client-key.pem -out client.csr -subj "/CN=client"
openssl x509 -req -in client.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out client-cert.pem -days 3650 -sha256
```

after generating those certs, you need to copy them into a folder and map them to the `/certs` directory inside the container, and then make sure that you are using tls to connect to the daemon inside the dind container, in the following way:

```bash
docker -H tcp://localhost:2376 --tlsverify --tlscacert ./certs/ca.pem --tlscert ./certs/client-cert.pem --tlskey ./certs/client-key.pem version
```

it’s quite some work to get it all going, i know, but once set it will save you a ton of time, when you have some automations to perform with docker inside docker.

regarding resources, i’d suggest looking into the docker documentation of course, but also consider reading “docker deep dive” by nigel poulton if you’re really trying to understand the docker architecture. another great resource would be the “kubernetes in action” book from marko luksa, since it explains very well many of the low-level concepts involved in container isolation and container runtimes. although not directly related to dind, understanding these concepts will help you greatly. and as usual, the windows server documentation for container features is always something to check, you can also find a ton of good articles there.

another potential hurdle is storage. the storage driver you are using for your host might not be compatible inside of the dind container. so you need to make sure that the driver being used inside the container is compatible with the host, usually its something that comes with the docker dind images, but it should also be checked, at least once.

one thing i've seen a few times is people forgetting to expose ports. if you are planning on running a service inside the dind container, you'll need to remember that the ports are exposed from the dind container to the host, and then to the real world. it's like a double hop, so you'll have to expose them at two points if you have a use case that requires this kind of exposure.

also, performance can be a bit of an issue depending on the load of the nested environment, so don’t expect the same performance as if you were running the docker daemon on a bare-metal machine. but it's usually acceptable for most use cases, specially in development or testing environment.

so, in a nutshell, dind on windows server with containers is achievable, but it needs some special handling. remember to use hyper-v isolation, pay attention to certificate management for secure communication, and be aware of the potential pitfalls in terms of storage and networking, and all should be fine. oh, and don’t forget to update docker to the latest version. i once spent a whole week troubleshooting, only to discover that i was using a very outdated docker version. that's life in the fast lane, i guess.
