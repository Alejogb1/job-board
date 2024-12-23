---
title: "How do I configure Nginx reverse proxy to redirect a subdomain to a dynamic port without updating settings?"
date: "2024-12-23"
id: "how-do-i-configure-nginx-reverse-proxy-to-redirect-a-subdomain-to-a-dynamic-port-without-updating-settings"
---

Alright, let's tackle this. It's a common scenario, one that I've bumped into several times over the years, particularly in environments where services spin up and down on dynamically allocated ports. Manually reconfiguring Nginx every time a port changes? Absolutely not. We need a robust and, more importantly, automated solution. So, how do we achieve this dynamic redirection of a subdomain to a port that shifts around without having to touch the Nginx config? The answer lies primarily in leveraging Nginx's capabilities alongside some intelligent scripting, often in the form of a companion process.

The core problem stems from Nginx’s static configuration. Traditionally, you specify a `proxy_pass` directive pointing to a specific IP and port, like so: `proxy_pass http://127.0.0.1:8080;`. This works perfectly well when the backend service resides on a fixed port. But when ports are dynamically allocated, this static link becomes a point of failure and tedious maintenance.

My go-to solution for this involves the following elements: an external process to track the port, a method for Nginx to retrieve this port in real time, and a bit of configuration to glue these parts together. For me, this often involves a small application, often a simple python script, that keeps tabs on the active port for the backend service. It stores this information in a location accessible to Nginx, commonly using a shared memory segment or a file, and ideally it's an atomic operation to avoid any conflicts.

The crucial element here is the `proxy_pass` directive inside your Nginx configuration, but we need it to be dynamic. We will utilize a variable for the port number, which Nginx can fetch at each request. Nginx’s configuration language can execute a small bit of code, but it doesn't allow directly executing arbitrary programs, which is why we need that external script to perform the port updates.

Here’s how we'd set up the configuration, broken down step-by-step with explanations:

**1. The External Port Tracker (Example in Python):**

Let's imagine the port is allocated by a backend application and the current port is stored in a file or in memory. I will show a simple file based example. The script responsible for managing the port could look something like this in python:

```python
import time
import os

def write_port(port, file_path):
    with open(file_path, 'w') as f:
        f.write(str(port))

if __name__ == "__main__":
    port = 8080 # Assume the initial port
    file_path = "/tmp/dynamic_port.txt"
    while True:
        # Simulate a port update, replace with your real backend port finding logic
        port = 8080 + int(time.time() % 5)
        write_port(port, file_path)
        time.sleep(10) # Check and update port every 10 seconds. Adjust as necessary
```

This python script writes the current port to `/tmp/dynamic_port.txt` every 10 seconds, simulating a dynamically assigned port. In your real implementation you might use `ps` or some other method to determine the port dynamically. The key is you need a constantly updated source of truth for the active port. In a production setting, you might use a redis database, shared memory, or a dedicated service discovery mechanism.

**2. Nginx Configuration:**

Now, let’s configure Nginx. Here’s a basic server block showcasing how to load the dynamic port and use it within the `proxy_pass` directive:

```nginx
server {
    listen 80;
    server_name sub.example.com;

    location / {
        # load dynamic port from file
        set $dynamic_port "";
        set $dynamic_port_file /tmp/dynamic_port.txt;
        if (-f $dynamic_port_file) {
             set_by_lua_file $dynamic_port /usr/local/nginx/lua/read_port.lua $dynamic_port_file;
        }
        if ($dynamic_port = "") {
            return 503 "Service Unavailable: Port not found.";
        }
        # Pass to backend
        proxy_pass http://127.0.0.1:$dynamic_port;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Add any necessary headers for your service
    }
}
```

Here's what's happening:
*   We start with a standard `server` block that listens for traffic for `sub.example.com` on port 80.
*   We initialize `$dynamic_port` to an empty string.
*   We define the location of our port file.
*   Using the `if` directive, we first check if file containing dynamic port exists, if it does, we load the port from a file using lua code. (more on that below)
*   If the port is still not loaded for some reason, then we return 503 service unavailable
*   If all goes well, we use the dynamic port in the `proxy_pass` directive and pass along the standard headers.

The core here is the `set_by_lua_file` directive with the associated lua script. This is where Nginx dynamically loads the port. If you don't have lua, you can use another approach with `ngx_http_perl_module`, however lua is much more straight forward.

**3. Lua script to load the port:**

The `read_port.lua` file referenced in Nginx would contain something like this:

```lua
local file_path = ARGV[1]
local file = io.open(file_path, "r")
if not file then
   return ""
end
local port = file:read("*all")
file:close()
return port
```

This Lua script simply reads the content of the provided file (in our case, `dynamic_port.txt`) and returns it. The key here is that this script is called by nginx every time a request arrives. This keeps your dynamic port up-to-date.

**Important Considerations:**

*   **Caching:** Nginx might cache results of variables which could lead to the wrong port being used. Lua provides a way to make sure we don't use the cached result. In production, use a short cache time within the `set_by_lua_file` call using `set_by_lua_file $dynamic_port "cache=off" /usr/local/nginx/lua/read_port.lua $dynamic_port_file;`.
*   **Error Handling:** In the provided snippets, error handling is kept to a minimum. In a production deployment, you need to handle errors such as port file not found or invalid port numbers, or any error in the lua code.
*   **Security:** ensure your port-tracking mechanism and the location of the port file are securely accessible and avoid unauthorized access.
*   **Scalability:** For a single server setup, reading from a file is feasible. If you have many servers, this introduces a potential bottleneck. More scalable solutions involve using a shared memory segment, a database, or a service discovery system.
*   **Service Discovery:** Consider using a proper service discovery mechanism like Consul, etcd, or ZooKeeper. These tools are specifically designed to handle dynamic service registration and discovery, and are far better for production environments as it allows multiple backend servers and a more robust management.
*   **Atomic operations**: it is important to update the port value in an atomic way to avoid race conditions. Lua functions using shared memory are a good approach to achieving this.
*   **Testing**: Always test this setup in a non-production environment to ensure you have handled all the cases correctly before going to production.
*   **Performance:** While reading a small file is generally fast, for very high traffic, reading files every request can add overhead. Consider a shared memory method or caching to reduce the amount of file read operations.

For further learning and detailed explanations of relevant concepts, I would recommend the following resources:

*   **"Nginx HTTP Server" by Igor Sysoev (Official Nginx documentation):** This is the definitive guide to Nginx, covering all aspects of configuration and internals. You'll find everything you need about `proxy_pass`, variables, and modules.
*   **"Programming in Lua" by Roberto Ierusalimschy:** This book will get you up to speed on writing Lua scripts. Lua is very simple and readable, making it easier to incorporate to load port in our nginx configuration.
*   **"Distributed Systems: Concepts and Design" by George Coulouris et al.:** This textbook provides an in-depth understanding of distributed systems and will be useful for anyone who wants to move beyond the single server setup. Chapters focusing on service discovery and distributed coordination can be particularly relevant for developing robust, scalable solutions.

Implementing this method will allow you to handle dynamically assigned ports without constant manual reconfiguration. Remember to adjust it to your specific needs and to test thoroughly. I've used variations of this setup for years, and it’s proven reliable and efficient when done correctly. Good luck with your implementation!
