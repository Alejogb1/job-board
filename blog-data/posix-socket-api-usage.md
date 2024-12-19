---
title: "posix socket api usage?"
date: "2024-12-13"
id: "posix-socket-api-usage"
---

Alright so you're asking about the POSIX socket API usage right Been there done that got the t-shirt and probably a few lingering network errors to prove it I've spent way too many late nights wrestling with those damn things believe me

Let me break it down from a real trenches perspective I mean we're talking about the bedrock of network communication here sockets man sockets

First off what are sockets exactly Well it's your program's handle to the network think of it like a file descriptor but instead of a file on disk it's a connection to another machine or even another process on the same machine The POSIX API gives you the tools to create connect listen send and receive data through these sockets It's powerful but it's also raw you're dealing with the nitty gritty here no magic just pure bytes and addresses

So what's the usual dance well you gotta start with `socket()` right This syscall is your entry point you gotta specify a domain address family which is typically `AF_INET` for IPv4 or `AF_INET6` for IPv6 Then you need the type which is either `SOCK_STREAM` for reliable TCP connections or `SOCK_DGRAM` for UDP datagrams And finally a protocol which is often just 0 since the system infers it from the other two parameters

Here's a little code snippet to get you going

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int sockfd;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    printf("Socket created successfully: %d\n", sockfd);
    // Don't forget to close it later
    close(sockfd)
    return 0;
}
```

That creates a socket but it's just a handle in space it isn't connected to anything To actually do something you need a server and a client right?

On the server side first you need to bind the socket to a specific address and port using `bind()` You fill out a `sockaddr_in` struct with your IP address typically `INADDR_ANY` to bind to all available interfaces and the port number you want to listen on

Then you call `listen()` This tells the system you're ready to accept incoming connections and sets the maximum backlog of pending connections

Finally you use `accept()` This is a blocking call that waits until a client tries to connect then it creates a new socket for that connection returning a new file descriptor You use that for all data communication with that client

Here is an example of server initialization

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    int sockfd, newsockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8888); // Port number

    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    if(listen(sockfd, 5) == -1){
       perror("listen failed");
       close(sockfd);
       exit(EXIT_FAILURE);
    }
    
    printf("Server listening on port 8888\n");

    newsockfd = accept(sockfd, (struct sockaddr *)&client_addr, &client_len);
    if (newsockfd == -1) {
        perror("accept failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    printf("Client accepted. New socket fd %d\n", newsockfd);
    // Now communicate through newsockfd 

    close(newsockfd);
    close(sockfd);
    return 0;
}
```

The client side is simpler You create a socket like before then you use `connect()` passing in the server's address and port If the server is listening on the specified port and address the call succeeds then you can start sending and receiving

Now you have a connected socket and can start blasting out or receiving bytes The functions `send()` and `recv()` are your friends here They are low level and take raw byte buffers be aware you will have to serialize data if you want to send complex objects

And don't forget the good old `close()` to release the socket descriptor when done

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char buffer[1024];

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8888); // Port number
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) <= 0) {
        perror("inet_pton failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
       perror("connect failed");
       close(sockfd);
       exit(EXIT_FAILURE);
    }

    printf("Connected to server\n");

    strcpy(buffer, "Hello from client");
    if (send(sockfd, buffer, strlen(buffer), 0) == -1) {
        perror("send failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    close(sockfd)

    return 0;
}
```

Now let's talk about the gotchas there are plenty believe me

One common one is the blocking nature of some of these calls `accept()` and `recv()` especially will just halt your program until something happens if you don't handle them properly This is where non-blocking sockets using `fcntl()` to set the `O_NONBLOCK` flag come into play and where `select` or `poll` can help you manage multiple sockets concurrently

Also endianness matters if your client and server are running on machines with different endianness you need to convert integers and other multi-byte data using `htons()` `htonl()` on the sending side and `ntohs()` `ntohl()` on the receiving side

Another pitfall is error handling The POSIX socket API functions can fail for a ton of reasons always check the return values of the calls especially if they return -1 they usually set `errno` which tells you what went wrong and you need to check the man pages and debug those using a debugger like gdb for example

And what about UDP UDP is like firing a message into the void there are no persistent connections and it isn't reliable each packet is independent You use `sendto()` to send and `recvfrom()` to receive specifying the destination and source addresses directly UDP is simpler but you have to handle packet loss and out-of-order delivery yourself

My biggest nightmare was debugging a complex network application where some packets where being dropped under load turned out to be insufficient buffer size for the kernel to handle the data I have learned it the hard way

In terms of resources for learning I'd recommend "UNIX Network Programming Volume 1 The Sockets Networking API" by W Richard Stevens that is a bible for socket programming also explore the man pages of functions that can be called directly from the shell like: `man socket` `man bind` `man connect` etc they provide all the details needed

Oh and quick joke why did the network engineer break up with the computer because he said she was not compatible with his IP address alright back to work

This is the gist of it It's a lot to digest initially but with time and practice it'll become second nature Network programming with sockets is a fascinating and powerful skill to have It's a very good skill in your tech toolbox And again check man pages and learn from them they are gold when you are dealing with systems programming and don't forget gdb it is your best friend when debugging
