---
title: "How can a single-threaded event loop avoid busy waiting?"
date: "2025-01-30"
id: "how-can-a-single-threaded-event-loop-avoid-busy"
---
The core challenge in designing efficient single-threaded event loops lies in preventing the CPU from constantly polling for new events, a process known as busy waiting, which consumes resources unnecessarily. Instead of continuously checking if an event has occurred, effective event loop implementations rely on mechanisms that allow the thread to suspend itself until an event is ready, thus conserving processing power. This is achieved through operating system-level facilities that notify the thread when data becomes available or a timeout expires.

A rudimentary approach to an event loop might look like a while loop that repeatedly checks a queue for new events and processes them if found; however, this naive implementation suffers from the aforementioned busy waiting problem. The loop executes regardless of whether new events exist, wasting CPU cycles and power. To mitigate this, event loop implementations employ techniques like blocking system calls, asynchronous I/O, and explicit event registration.

Specifically, the primary method for avoiding busy waiting involves a combination of select/poll/epoll (or equivalent mechanisms on different operating systems) and non-blocking I/O operations. These mechanisms allow an application to request that the operating system notify it when a file descriptor (e.g., a network socket or a pipe) becomes ready for reading, writing, or has encountered an error. Instead of directly performing an I/O operation, the application asks the OS to inform it when the operation can proceed without blocking.

The process is as follows: I maintain a registry of file descriptors or "watchers" along with associated event types (e.g., read, write) that the event loop monitors. When an operation involving a watched file descriptor is initiated, the code switches to a non-blocking I/O operation. The system call does not halt the current thread if the file descriptor isn't ready, but rather immediately returns a specific error code indicating that no data is ready (for reading) or that it is not possible to write at this moment (for writing).

The event loop then uses a system call like `select`, `poll`, or `epoll` to tell the operating system to wait until at least one of the watched file descriptors becomes ready. Crucially, this wait operation blocks the thread. The thread does not perform continuous polls of the watched file descriptors but is put into a suspended state by the operating system until the condition is met. Once at least one watched file descriptor triggers an event, the operating system wakes the thread, and the event loop can then proceed with processing the newly available events, executing their associated callbacks.

Below are examples illustrating this concept. These are simplified to demonstrate the core mechanisms; practical implementations would include more robust error handling, resource management, and event dispatching logic.

**Example 1: Using `select` with Sockets**

```python
import socket
import select

def simple_server_select():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8000))
    server_socket.listen(5)
    server_socket.setblocking(False) # Sets the socket to non-blocking

    inputs = [server_socket]  # List to monitor for readiness
    
    print("Server listening on port 8000")
    
    while inputs:
        readable, _, _ = select.select(inputs, [], [])
        
        for sock in readable:
            if sock is server_socket:
                conn, addr = server_socket.accept() # Accept new connections
                conn.setblocking(False)
                inputs.append(conn) # Add new socket to the monitor
                print("Accepted connection from", addr)
            else:
                try:
                     data = sock.recv(1024)
                     if data:
                         print("Received:", data.decode())
                         sock.send(data) # Echo back data
                     else:
                         print("Closing connection")
                         inputs.remove(sock)
                         sock.close()
                except ConnectionResetError:
                      print("Client disconnected abruptly.")
                      inputs.remove(sock)
                      sock.close()


if __name__ == "__main__":
    simple_server_select()
```

In this Python example, the server socket and client sockets, once accepted, are added to the `inputs` list. The `select.select(inputs, [], [])` call blocks until one or more sockets become readable, avoiding a busy-waiting loop. The server socket, when readable, indicates that a new client connection is waiting to be accepted, whilst a client socket, when readable, has data available. The sockets are also configured to be non-blocking, so if a read fails it does not block the thread.

**Example 2: Polling for Timed Events**

```python
import time
import select

def simple_timer_loop():
    timeouts = {time.time() + 2: "Timeout 2s", time.time() + 5: "Timeout 5s"}
    
    while timeouts:
        next_timeout = min(timeouts)
        timeout = next_timeout - time.time()
        if timeout <= 0:
            print(timeouts[next_timeout])
            del timeouts[next_timeout]
        else:
            readable, _, _ = select.select([], [], [], timeout) # Waits for timeout

if __name__ == "__main__":
   simple_timer_loop()
```

Here, the `select.select([], [], [], timeout)`  is used for timing purposes. Since we are not interested in file descriptor readiness events, the first two lists of the `select` call are empty. Instead, we are interested in the fourth argument, which specifies the maximum amount of time to wait. The call will block for up to that duration or return earlier if a signal is received. This prevents the program from continuously checking time, and uses the blocking capability to wait for a precise timeout.

**Example 3: A Simple `epoll` based example (Linux specific)**

```python
import socket
import select
import errno

def simple_server_epoll():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8001))
    server_socket.listen(5)
    server_socket.setblocking(False)

    epoll = select.epoll()
    epoll.register(server_socket.fileno(), select.EPOLLIN) # register the socket

    connections = {}
    print("Server listening on port 8001")

    try:
        while True:
            events = epoll.poll(1) # Wait for events on the watched file descriptors
            for fileno, event in events:
                if fileno == server_socket.fileno():
                    conn, addr = server_socket.accept()
                    conn.setblocking(False)
                    epoll.register(conn.fileno(), select.EPOLLIN) # register new client socket
                    connections[conn.fileno()] = conn
                    print("Accepted connection from", addr)
                elif event & select.EPOLLIN:
                      conn = connections[fileno]
                      try:
                           data = conn.recv(1024)
                           if data:
                               print("Received:", data.decode())
                               conn.send(data)
                           else:
                               print("Closing connection")
                               epoll.unregister(fileno)
                               conn.close()
                               del connections[fileno]
                      except ConnectionResetError:
                            print("Client disconnected abruptly.")
                            epoll.unregister(fileno)
                            conn.close()
                            del connections[fileno]

    finally:
        epoll.unregister(server_socket.fileno())
        epoll.close()

if __name__ == "__main__":
    simple_server_epoll()

```

This example demonstrates the usage of `epoll`, which is a more scalable and efficient alternative to `select` for managing large numbers of file descriptors, especially in Linux environments.  `epoll` allows registering specific file descriptors to be watched for specific events and it will return only the descriptors for which events occurred. The `epoll.poll()` call is blocking, similar to `select.select()` in the prior examples, preventing busy waiting. The `try finally` block ensures that resources such as the epoll instance are released when they are no longer needed.

For further study, I recommend examining operating system manuals pertaining to system calls like `select`, `poll`, and `epoll` (or their OS-specific counterparts).  Furthermore, texts discussing asynchronous programming paradigms and event-driven architectures provide additional insight into the theoretical underpinnings of these concepts. Books on network programming can also be quite helpful, as sockets and other network resources are common examples for event loops. Looking into implementations of popular event loop libraries in various programming languages will provide valuable practical knowledge. For Python, the asyncio library and frameworks like Twisted are prime examples of established event loop implementations. For Node.js, the Libuv library serves a similar purpose. Examining code directly used in these widely-used event loops is extremely valuable for deepening the understanding of the underlying mechanisms.
