---
title: "How do I connect a Laravel Sail instance to an SSH tunnel?"
date: "2024-12-23"
id: "how-do-i-connect-a-laravel-sail-instance-to-an-ssh-tunnel"
---

Alright, let's talk about connecting a Laravel Sail instance through an ssh tunnel. This is something I've tackled more times than I care to count, particularly when dealing with databases hosted on a private network or legacy systems that refuse to play nice with direct connections. It's a common scenario in development environments, and getting it set up properly can save you a significant headache. So, let’s get into the specifics, aiming for a clear, technically grounded approach.

First, let’s clarify the core issue. Laravel Sail, using docker, isolates your application's environment. This is great for consistency but creates a barrier when you need to interact with resources outside of the docker network. An SSH tunnel essentially establishes a secure pathway between your local machine and a remote server, forwarding ports. This is what will allow your dockerized application, even one utilizing Laravel sail, to access services as if it were on the same network.

My experience with this wasn't theoretical; it stemmed from a particular project involving a client's legacy database system. We weren't permitted to expose the database directly, requiring us to establish a tunnel through their jump server. Let's break down the steps and I'll share some practical code examples.

The foundational principle revolves around port forwarding. You'll establish a tunnel using `ssh` with specific port forwarding directives and then configure your Laravel application to use this forwarded port when communicating with the remote service.

**Step 1: Establishing the SSH Tunnel**

The primary tool here is the `ssh` command. The syntax we use frequently looks like this:

```bash
ssh -N -L local_port:remote_host:remote_port user@remote_server
```

Let's dissect this command:

*   `-N`: This tells `ssh` not to execute a remote command. We only want the port forwarding functionality.
*   `-L local_port:remote_host:remote_port`: This specifies the port forwarding. `local_port` is the port on your local machine you will use to access the service. `remote_host` is the address of the server hosting the service as seen from the jump server, and `remote_port` is the service's port number on that machine.
*   `user@remote_server`: This is your login information for the server hosting the tunnel. This is typically a jump server or an intermediary machine accessible from your local network, and also in direct communication with the machine with the resource you're targeting (in this example, the database).

For example, imagine your database server is named `db.internal.example.com` running on port `5432`, the database server can be reached through a gateway server `gateway.example.com`, and you want to access it locally on port `6543`. Your command would look like this:

```bash
ssh -N -L 6543:db.internal.example.com:5432 user@gateway.example.com
```

This command sets up a persistent tunnel, forwarding traffic to `localhost:6543` to `db.internal.example.com:5432` through the specified gateway server. Keep this terminal window open; closing it will terminate the tunnel.

**Step 2: Configuring Laravel**

Now that the tunnel is open, you need to configure your Laravel application, specifically its database connection settings, to use the forwarded port. You’ll modify your `.env` file and potentially some database configuration files. Here's an example using postgresql database.

```dotenv
DB_CONNECTION=pgsql
DB_HOST=127.0.0.1
DB_PORT=6543
DB_DATABASE=your_database_name
DB_USERNAME=your_database_user
DB_PASSWORD=your_database_password
```

Notably, you’re setting `DB_HOST` to `127.0.0.1` (or `localhost`) since the port forwarding makes the remote service seem like it's on your machine. The crucial part is setting `DB_PORT` to `6543`, which corresponds with the `local_port` you defined in your SSH command.

**Step 3: Putting it together in Sail**

Often, the tunnel will need to be up before you start your Sail containers, but sometimes we want the tunnel to exist alongside sail, as part of the development environment, particularly if multiple services outside of your local environment are going to be necessary for development. In this case you can create a bash script that fires up the ssh tunnel and your sail environment. This bash script would look like the following, and would live in your project's root directory:

```bash
#!/bin/bash

# Start the SSH tunnel in the background
ssh -N -L 6543:db.internal.example.com:5432 user@gateway.example.com &

# Keep the process running and store its PID
TUNNEL_PID=$!

# Pause to allow the tunnel to start up
sleep 2

# Start the Laravel Sail environment
./vendor/bin/sail up -d

# Function to handle cleanup of the tunnel
cleanup() {
  echo "Cleaning up... Shutting down ssh tunnel."
  kill "$TUNNEL_PID"  # Use the stored PID to stop the tunnel process
}

# Register the cleanup function to be called on exit or interrupt
trap cleanup EXIT SIGINT SIGTERM

# Keep the main script running (useful to keep the docker environment in place)
while true; do
    sleep 3600; # Keep script alive
done;

```

This script does the following:
1. starts the tunnel process in the background
2. stores the process id so the tunnel can be killed later
3. pauses to allow the tunnel to spin up
4. fires up the sail environment as a detached process, the `-d` flag ensures that the script is unblocked so it can continue execution
5. sets up an error handler to kill the tunnel when we kill the script
6. the loop keeps the script alive and allows sail to remain up, until the user interrupts the script.

You can make the script executable using `chmod +x startup.sh`. Then run the script using `./startup.sh`, to get a sail environment up that's connected to an ssh tunnel. This is useful, and was frequently used, when we were interacting with legacy internal services, and was useful for a more consistent development experience.

**Troubleshooting**

*   **Connectivity issues:** If your application can't connect, double-check your `.env` settings, ensure the tunnel is active, and confirm that the ports are correctly configured on both ends. Use `lsof -i :6543` on your local machine to see if the tunnel is listening on the correct port.
*   **Firewall interference:** Verify that your local firewall is not blocking the local port you are forwarding through the tunnel. On linux, use `iptables -L` or `ufw status` (if `ufw` is installed) to check for active rules.
*   **Timeout issues:** Sometimes, the connection will take a while to establish on first connection. This can sometimes appear as a connection failure, but trying again may resolve the connection. This is especially common if the remote service is waking up from a period of inactivity or is under heavy load.

**Further Reading**

For a deeper understanding of SSH, I highly recommend checking out “*SSH, The Secure Shell: The Definitive Guide*” by Daniel J. Barrett and Richard Silverman. It provides comprehensive insights into SSH functionality and configuration, covering advanced tunneling techniques as well. Additionally, exploring the documentation of `ssh` with the command `man ssh` in your terminal will provide detailed information on each flag and its functionality. For specific details on docker and its networking, I recommend diving into Docker's official documentation, found at [docs.docker.com](https://docs.docker.com/). This documentation provides extensive information on docker networking, which is crucial in understanding how Sail works under the hood.

Remember that, setting up an SSH tunnel isn't overly complicated, but requires precision. With these steps and examples, you should be able to connect your Laravel Sail application to those tricky internal services, and I'm confident you'll find it a useful skill. As always, thorough testing and careful attention to detail will be key to a smooth experience.
