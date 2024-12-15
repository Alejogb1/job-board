---
title: "How to Simulate a computer network with docker containers & socket programming?"
date: "2024-12-15"
id: "how-to-simulate-a-computer-network-with-docker-containers--socket-programming"
---

alright, so you want to spin up a simulated network using docker and socket programming, that's a cool project, and i've been there. i actually did something similar back when i was fiddling around with distributed systems for my uni thesis, it was a real headache at first.

essentially, what you're doing is creating a bunch of isolated containers that can talk to each other as if they were on a real network. docker helps us with the isolation part, creating these mini-virtual machines, and socket programming gives us the tools to make them communicate. i'm not gonna lie, it does take a bit to wrap your head around it the first time, but it's very doable.

let's break it down. first, you need your docker setup. you'll be making a dockerfile for each type of node in your network. if you have clients and servers, you might have at least two different dockerfiles. you'll also need a docker-compose file to bring these containers up, and connect them on a shared virtual network.

here's a basic example of a dockerfile for a simple server:

```dockerfile
# server Dockerfile
from python:3.9-slim-buster

# setting up working directory
WORKDIR /app

# copy the server file to this location inside the container
COPY server.py .

# install any needed python packages
RUN pip install --no-cache-dir

# run the server when the container starts
CMD ["python", "server.py"]
```

and here is a matching client one:

```dockerfile
# client Dockerfile
from python:3.9-slim-buster

# setting up working directory
WORKDIR /app

# copy the client file to this location inside the container
COPY client.py .

# install any needed python packages
RUN pip install --no-cache-dir

# run the client when the container starts
CMD ["python", "client.py"]
```

notice that these files are super basic. they just grab a python image, set up a working directory, copy your code, and run your code. the real magic is in the python files `server.py` and `client.py`, where you do the socket programming.

here's a rudimentary example of `server.py`:

```python
# server.py
import socket

# the host should be 0.0.0.0 to listen to all incoming connections
host = '0.0.0.0'
port = 12345  # the port number must be the same on both ends

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

print(f"server is listening on {host}:{port}")
# waits to accept a connection
connection, address = server_socket.accept()
print(f"connection from: {address}")

while True:
    # receive data
    data = connection.recv(1024).decode()
    if not data:
        break # when no more data is sent, stop

    # prints the data received
    print(f"received: {data}")
    response = "message received"
    connection.sendall(response.encode())  # send a response

# when finished close the connection
connection.close()
```
and for the `client.py` file:

```python
# client.py
import socket

host = 'server' # we'll use the docker container name here
port = 12345

# creates the socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

message = "hello from client!"
client_socket.sendall(message.encode()) # send data

response = client_socket.recv(1024).decode()
print(f"response from server: {response}")

client_socket.close()
```

notice how the client is connecting to 'server'. this is the crucial part where docker networking helps. the container named `server` is reachable on the docker network by that name. now, how to glue it together? with docker-compose.

here's a docker-compose.yml that sets up a simple server-client network:

```yaml
version: '3.8'
services:
  server:
    build: ./server # this folder contains your server Dockerfile
    ports:
      - "12345:12345" # exposes the server port to the outside but not needed for the client
  client:
    build: ./client # this folder contains your client Dockerfile
    depends_on:
      - server  # tells docker that the client depends on the server
```

you'll save these files in an accessible location for you like a folder called `sim-network`. the directory structure would be something like this:

```
sim-network/
├── docker-compose.yml
├── server/
│   ├── Dockerfile
│   └── server.py
└── client/
    ├── Dockerfile
    └── client.py
```
to use it you will have to navigate to the sim-network directory and run `docker compose up --build`. docker-compose will now build your docker images using the dockerfiles provided and run these based on the configuration found in the docker-compose file.

a couple of things here. first, notice the `depends_on` in docker-compose. this makes sure the server container starts before the client. second, the client uses 'server' as the hostname which docker uses to resolve to the container named server when networking between containers on the same bridge network. this is called the docker network. third, if you have more clients you can just add more of them. finally, the port mapping in the server section of the docker-compose file, is useful if you want to connect to the server from outside the docker network but is not needed for the client to work.

now, the python code, the socket programming part. you're using `socket.socket` to create a socket, specifying ipv4 (`socket.af_inet`) and tcp (`socket.sock_stream`). on the server, `bind` binds the socket to an address and port. `listen` makes the server socket listen for connections. `accept` accepts the connection and gives you a new socket for communicating with a client. the server uses the loop to receive data from the client using `recv`, decodes it, prints it and replies sending `sendall`. the client on the other hand does `connect` and sends a message, receives a response and prints it and finally closes the socket. you should know that the client needs to connect to a server that's running and listening or it will throw an exception. the ports should match on both sides. this is a basic connection, but you can build very complex protocols on top of this, including different message structures and error handling.

one detail that messed me up a while back, was the blocking nature of socket calls. if your client or server gets stuck in a `recv` or `accept` call, it stops processing until data is received or a connection is made. so that is something that should be taken into account, also, if you want to send multiple messages, you should implement some kind of message boundary, or you will need to use a loop. but that's a bit advanced for a basic simulated network.

i remember when i did this, i was trying to send some json over the socket and i was running in circles because of the encoding problems, and then the whole blocking calls. i ended up spending more time figuring out these basic network principles than actually doing the simulations for my thesis. and then i found a quote on a book about networking that resonated a lot "the network is reliable, but the network is not always reliable". funny when you think about it.

if you want to get deeper in this i recommend the book "computer networking: a top-down approach" by kurose and ross. it will give you a very thorough introduction to networking principles. also, the linux socket programming api can be found in the man pages. this is where i learned all the details of socket programming. there is also a great chapter on sockets on the "unix network programming" by stevens and rago. i would strongly recommend checking these, i went though them when i was building my distributed simulation and it really saved me a lot of headache, i think it can be useful for you as well. it's quite different from all the hand holding that modern high level libraries do.

in the end this all boils down to a solid grasp of networking basics, docker configuration, and socket programming. once you get the hang of it, you can simulate pretty intricate network scenarios. it will take a couple of attempts and errors and reading documentation, but it will get there, i know from experience. don't be scared to try things and don't forget to check the documentation and have fun with it.
