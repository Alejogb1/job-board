---
title: "How can I use the C socket API in C++ on z/OS?"
date: "2025-01-26"
id: "how-can-i-use-the-c-socket-api-in-c-on-zos"
---

The z/OS operating system, while often perceived as a legacy mainframe environment, fully supports standard POSIX socket APIs, albeit with nuances stemming from its underlying architecture and specific network configuration requirements. Having spent considerable time developing high-performance communication layers on z/OS, I've found the key to successful socket programming lies in understanding how the platform's networking subsystems interact with the standard BSD socket interface. These subsystems often necessitate specific configuration beyond what one might expect in Linux or Windows environments.

The foundational principle remains consistent: we use the standard `socket()`, `bind()`, `listen()`, `accept()`, `connect()`, `send()`, and `recv()` functions, among others. However, considerations such as EBCDIC encoding, z/OS-specific address families, and the use of USS (Unix System Services) file descriptors demand specific attention. The crucial difference stems from z/OS's TCP/IP implementation, which allows for various network interfaces and configuration options that are not present in typical workstation operating systems. Failure to account for these aspects will often result in unexpected connection errors or data corruption.

A typical socket communication sequence would involve establishing a socket, associating it with a specific port on a specified interface, connecting to a remote server, or accepting a connection request, and then sending and receiving data. The address family utilized is crucial; `AF_INET` would typically be used for IPv4 connections, and `AF_INET6` for IPv6. However, for z/OS, even with `AF_INET`, there might be specific interface configurations one has to account for, using the `sin_addr` field of the `sockaddr_in` structure. One might, for example, need to bind to a specific z/OS IP interface if multiple network devices are present. If you're using dynamic VIPA (virtual IP address), you might not bind to a specific interface. It becomes crucial to understand the z/OS TCPIP profile settings.

Let’s illustrate the socket creation, address binding and listening steps with the following C++ code example:

```c++
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return 1;
    }

    // Set socket options to reuse address/port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        return 1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // Bind to all available interfaces
    address.sin_port = htons(8080); // Port to listen on

    // Bind the socket to the specified address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return 1;
    }

    // Listen for incoming connections
    if (listen(server_fd, 3) < 0) {
        perror("listen failed");
        return 1;
    }

    std::cout << "Server listening on port 8080..." << std::endl;

    // Example accepting a single connection
     if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) {
         perror("accept");
         return 1;
     }
    
    std::cout << "Connection accepted" << std::endl;
    close(new_socket); // Close the accepted socket connection
    close(server_fd); // Close the listening socket
    
    return 0;
}
```

In this example, I've shown a basic server setup. We first create a socket using `socket(AF_INET, SOCK_STREAM, 0)`. This creates a TCP socket for IPv4 addressing. `setsockopt()` is used to enable reuse of the address and port, often useful during testing. We then populate the `sockaddr_in` struct specifying `INADDR_ANY` to listen on all available interfaces and set the port to 8080. Crucially, we use `htons()` to convert the port number to network byte order. `bind()` associates the socket with the provided address and port, and `listen()` sets up the socket for accepting incoming connections with a backlog of 3. Finally, `accept()` blocks until a client attempts to connect. I then close the accepted connection and then the listening socket to cleanly exit. In practice, this would be wrapped in a loop. One should handle potential errors more robustly, but this showcases the key steps.

A practical example where you need to make a connection is shown below. This represents how you would create a client.

```c++
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket creation error");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080); // The server port

    // Convert IPv4 address from text to binary form
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        perror("invalid address/ address not supported");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connection failed");
        return -1;
    }

    const char *message = "Hello from client";
    send(sock, message, strlen(message), 0 );
    std::cout << "Message sent" << std::endl;

    int valread = recv(sock, buffer, 1024, 0);
    if(valread > 0) {
        std::cout << "Server response:" << buffer << std::endl;
    } else {
         perror("receiving failed");
    }
    
    close(sock); // Close the socket
    return 0;
}
```

In this client code, the same `socket` function is used, but `connect()` is used to actively connect to the server. I've used `inet_pton()` to convert a string IP to the numeric address required. The server IP in this example is "127.0.0.1", the loopback address. If connecting to a server on a remote z/OS LPAR, you would need the actual server IP address. Once the connection is established, the message is sent using the `send` call, and `recv` is used to receive the message from the server. Error handling is included to gracefully fail if something goes wrong. I close the socket once communication is complete. Note: This example expects a server listening on localhost, port 8080, such as from the first code block.

Data transmitted over a socket may require conversion if the server and client systems have different character encodings. z/OS primarily uses EBCDIC, whereas most other platforms default to ASCII/UTF-8. A failure to account for this will result in garbled data. To handle EBCDIC to ASCII/UTF-8 conversions on z/OS, you might utilize the `iconv` library, which is available within USS, or, alternatively, if not a large amount of data, perform conversion manually character-by-character after receiving, and before sending, respectively, with lookup tables. It is vital that this data encoding is well understood between the two endpoints. If the data is binary, then these conversions are not required.

Finally, a more advanced example would involve using non-blocking sockets and `select()` or `poll()` to manage multiple concurrent connections. A single threaded server could handle several clients simultaneously by using I/O multiplexing rather than a thread per connection. The code below demonstrates this pattern using `select`.

```c++
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <sys/select.h>
#include <vector>

int main() {
    int master_socket, addrlen, new_socket, client_socket[30], max_clients = 30;
    struct sockaddr_in address;
    int activity, valread, sd, max_sd;
    char buffer[1025];
    fd_set readfds;
    
    for (int i = 0; i < max_clients; ++i) {
        client_socket[i] = 0;
    }
    
    if ((master_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        return 1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);
    
    if (bind(master_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        return 1;
    }

    if (listen(master_socket, 3) < 0) {
        perror("listen failed");
        return 1;
    }
    
    std::cout << "Server listening..." << std::endl;

    addrlen = sizeof(address);

     while (true) {
        FD_ZERO(&readfds);
        FD_SET(master_socket, &readfds);
        max_sd = master_socket;

        for (int i = 0; i < max_clients; i++) {
            sd = client_socket[i];
            if (sd > 0)
                FD_SET(sd, &readfds);
            if (sd > max_sd)
                max_sd = sd;
        }
        
        activity = select(max_sd + 1, &readfds, nullptr, nullptr, nullptr);
        if ((activity < 0) && (errno != EINTR)) {
             perror("select error");
        }
         
        if (FD_ISSET(master_socket, &readfds)) {
            if ((new_socket = accept(master_socket, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) {
                perror("accept error");
                return 1;
            }

            std::cout << "New connection: socket fd: " << new_socket << ", ip: " << inet_ntoa(address.sin_addr) << " port: " << ntohs(address.sin_port) << std::endl;

            for(int i = 0; i < max_clients; ++i) {
              if(client_socket[i] == 0) {
                client_socket[i] = new_socket;
                break;
              }
            }
         }
       
        for (int i = 0; i < max_clients; i++) {
            sd = client_socket[i];
           if(FD_ISSET(sd, &readfds)){
             valread = recv(sd, buffer, 1024, 0);
              if(valread == 0) {
                   std::cout << "Client disconnected: " << sd << std::endl;
                  close(sd);
                  client_socket[i] = 0;
              } else {
                buffer[valread] = '\0';
                  std::cout << "Received message from client " << sd << ": " << buffer << std::endl;
                 send(sd, buffer, strlen(buffer), 0);
               }
           }
        }
     }
     close(master_socket);
    return 0;
}
```

This example demonstrates a non-blocking server. `select` is used to monitor multiple sockets for readability. The server listens on `master_socket` for incoming connections, and then it adds newly accepted connections to `client_socket` array. Each socket added is then checked if there is a readable message by calling `recv` in a loop of all of the clients. If a client has disconnected, then the socket is closed, and removed from the list. The data is then echoed back to the client. This illustrates a fundamental approach for concurrently handling several clients using a single process.

For a deeper dive, consider reviewing IBM's documentation on z/OS UNIX System Services, particularly its sections on socket programming. "TCP/IP Guide" from IBM and the "z/OS Communications Server IP Configuration Guide" provide context on TCP/IP configuration and networking concepts specific to z/OS. Additionally, books dedicated to advanced Unix socket programming provide invaluable insights. Examining open-source networking projects with z/OS support would also prove valuable. These combined resources will provide a comprehensive view of socket development on the z/OS platform.

In summary, while the POSIX socket API is standard across platforms, z/OS presents unique challenges requiring careful attention to detail. Understanding platform-specific networking and character encoding requirements is essential for successful socket development in this environment.
