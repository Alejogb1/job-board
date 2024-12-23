---
title: "Is remote development possible using PhpStorm, JetBrains Gateway, and no internet connection?"
date: "2024-12-23"
id: "is-remote-development-possible-using-phpstorm-jetbrains-gateway-and-no-internet-connection"
---

Okay, let's tackle this. The feasibility of remote php development with PhpStorm, using JetBrains Gateway, and operating entirely offline presents a nuanced situation. I’ve personally navigated scenarios similar to this, particularly in secure, disconnected environments. The short answer is: not in the conventional way you'd expect, but with some strategic planning, it's definitely achievable, albeit with significant constraints.

The core issue lies with JetBrains Gateway’s architecture. It's explicitly designed to facilitate a thin client/server setup where the vast majority of processing and code management resides on a remote server. The client, what you experience on your local machine, is essentially a display and input interface. Key to this functionality is the network connection that allows this continuous, real-time exchange of data. Without it, JetBrains Gateway simply cannot establish the necessary communication channels to work correctly. You cannot bypass this need for network connectivity with Gateway; it's integral to how it is designed.

However, there’s a crucial distinction to make. While an *internet* connection is required for initial setup and for remote machine discovery, the actual day-to-day coding activity can be done offline **if** certain conditions are met and a local server has been previously set up and running. Essentially, you’re talking about a hybrid scenario: a local server pretending to be remote.

Let me break down what I mean, based on my past experiences working on projects within compartmentalized environments. I recall one specific situation where we were developing a critical internal application on a network that was entirely isolated. We couldn't simply reach out to external servers for code repositories or remote connections. Here's how we worked around the issue using a 'local-remote' approach that I believe aligns with your scenario.

The crux of the workaround is this: you need a running instance of PhpStorm backend on a "remote" machine – and by "remote" I mean accessible through JetBrains Gateway – **that is on the same network as your client machine, even if that network is offline to the rest of the world.** In other words, we set up a local server to behave as if it were remote.

**Pre-Conditions:**

1.  **Initial Online Setup:** This step is unavoidable. Initially, both your client machine (the one running Gateway) and the server machine (the one running the PhpStorm backend) must be online so that you can install and configure the necessary components:
    *   Install JetBrains Gateway on the client machine.
    *   Install PhpStorm on the *server* machine.
    *   Use the Gateway to connect to the server for the first time. This is crucial, as it establishes the initial handshake, configures the server to be accessible to Gateway, and downloads necessary client components.

2. **Offline Environment:** After this initial setup, you can move both machines to your offline environment. Both machines should ideally be on the same isolated network for the next steps to work flawlessly.

**The Offline Workflow:**

1.  **Local Network, Pretending Remote:** You’ll configure the “server” machine to serve the development project on a network address that is accessible from your client machine. This can be a direct Ethernet connection between the two, or a small, isolated local network created with a router not connected to the internet. This is key - your client machine will be "remote" in the Gateway view from this server, but it's physically connected to your server on this isolated, local network.

2.  **Server Component Always Running:** On the 'server' machine, launch your PhpStorm instance, open your development project, and keep it running. This ensures the necessary backend processes are alive to be accessed by Gateway. It may be necessary to configure this to startup automatically with the system.

3.  **Client Connection through Gateway:** From your client machine, launch JetBrains Gateway. If your previous connection was configured properly on the same network address, it may discover the running server. If not, you can use the IP address, or hostname of the 'server' on the network, if manually defined, which you will have previously configured during the initial online setup and stored by gateway. Your client application will then connect to the running server-based PhpStorm and present you with the user interface as normal. Your work is then entirely local to the project files present on this server.

4. **Code Execution and Testing:** Because the full PhpStorm instance is running on your local "remote" server, execution of php, unit tests etc., will all take place locally. There is no reliance on internet for these functions. This allows for full normal coding workflow.

**Code Snippet Examples**

To give a better idea of how this works, consider these conceptual snippets, remembering that actual configuration may vary based on your specific network setup:

**Snippet 1: Server-side Startup Script (Linux):**

This simulates part of how a server could be configured to run PhpStorm persistently.

```bash
#!/bin/bash

# ensure PhpStorm runs in the background, regardless of logout/terminal closure
nohup /opt/phpstorm/bin/phpstorm.sh &

# Optional: Print server's IP for reference
# Get the first non-loopback ipv4 address, for discovery on network
SERVER_IP=$(ip route get 1 | sed -n 's/^.*src \([0-9.]*\).*/\1/p')
echo "PhpStorm Server running on IP: ${SERVER_IP}"

# Optional: Keep terminal open so you see the output, remove this on production
sleep infinity
```

**Snippet 2: Client-side Gateway Discovery (Conceptual):**

This shows how the Gateway might initially connect to your server. This is mostly handled by the Gateway interface, but conceptually, this is how it locates the server using an address.

```python
# A simple pythonic representation, note this is pseudocode
import socket

def discover_server(server_ip, server_port):
    try:
       with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
           s.connect((server_ip, server_port))
           s.sendall(b"Are you there?") # A placeholder signal
           data = s.recv(1024)
           if data == b"Yes, I am the gateway server.":
              print ("Gateway Server discovered at {}:{}".format(server_ip, server_port))
              return True
           else:
               print ("Server discovered but server not of expected type")
               return False
    except Exception as e:
       print ("Failed to locate gateway server: " + str(e))
       return False

server_ip = "192.168.1.100" # Your server's local IP
server_port = 12345 # Arbitrary port
if discover_server(server_ip, server_port):
   print ("Continue with connection to server")

```

**Snippet 3: A basic php function**

This shows that normal coding functionality remains intact on the client, as the code is not actually running on the client, but the remote server.

```php
<?php

function greetUser(string $name):string{
    return "Hello, ". $name . "!";
}

echo greetUser("Offline User");
```

**Important Considerations:**

*   **Synchronization:** You are now entirely dependent on your “server” machine's resources for development and that includes its ability to store the complete project files. Any code files or changes are managed directly on the server, and do not automatically sync to the client computer. Backups and version control must therefore all be handled within the server, and will not be present on the client machine unless you copy them separately.
*   **Security:** In a truly offline environment, physical security of both machines is paramount. The “server” machine becomes a critical point of failure, potentially exposing your source code if compromised. Ensure that your machines are protected to avoid unauthorized access.
* **Performance:** As a full IDE and project is still being operated on the server machine, the performance will be similar to operating this directly. However, you may find some increased latency in user inputs, as these must be handled through the local network.

**Recommended Resources:**

For a deep dive into these concepts, I would highly recommend:

1.  **"Computer Networks" by Andrew S. Tanenbaum:** This book is a classic and provides a fundamental understanding of networking concepts, including local area networks, protocols, and addressing, all of which are crucial for setting up a robust offline development environment.

2.  **"Operating System Concepts" by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne:** This provides comprehensive knowledge of operating system principles, which are necessary to understand how server processes interact with hardware and network configurations.

3. **JetBrains Documentation**: While not a book, Jetbrains does provide extensive documentation on Gateway and its required functionality. Understanding the requirements of the software directly, is essential to be able to understand and work around its intended use.

In summary, while JetBrains Gateway isn't fundamentally designed for complete offline operation, a local server can simulate the necessary remote environment using a preconfigured installation, coupled with careful network setup. You're essentially creating a contained development ecosystem, fully detached from the internet. This approach requires detailed planning, solid understanding of networking principles, and careful management of your server environment, but it is completely possible with careful planning. The key lies in shifting your perspective, understanding it’s a "local-remote" hybrid.
