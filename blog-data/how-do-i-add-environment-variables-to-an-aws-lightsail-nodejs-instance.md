---
title: "How do I add environment variables to an AWS Lightsail Node.js instance?"
date: "2024-12-23"
id: "how-do-i-add-environment-variables-to-an-aws-lightsail-nodejs-instance"
---

,  I've certainly spent a fair amount of time navigating the intricacies of AWS Lightsail, and setting up environment variables for Node.js instances has been a common task. It’s one of those things that seems straightforward initially, but can quickly become a head-scratcher if not approached systematically. In my experience, particularly with a project that involved a complex microservices architecture deployed across Lightsail, proper environment variable management proved crucial for maintaining consistency and flexibility across different deployment stages (dev, staging, prod, and so on).

Fundamentally, the challenge stems from the fact that Node.js applications don’t magically know what variables you want to use. You have to explicitly provide these variables to the runtime environment where your application executes. There are several avenues available to achieve this with Lightsail, but I’ve found a couple that are consistently reliable and relatively easy to maintain.

First, let's clarify why this is important. Environment variables are a powerful mechanism for configuring applications without modifying their code. This separation of configuration from application logic promotes cleaner, more maintainable, and more secure applications. You might use environment variables to store database connection strings, api keys, different port numbers, and any other setting that might vary depending on the specific environment your application is running in.

The most straightforward method, and the one I often rely on for simpler configurations, is to directly set environment variables within the shell environment of your Lightsail instance. When a Node.js process starts, it inherits the environment of its parent process – typically the shell (bash, zsh, etc.) or systemd in this case. Here's how you'd typically approach it.

Connect to your Lightsail instance via SSH, and then, you can set variables in your `.bashrc` or `.zshrc` file. These files are executed every time you open a new shell, ensuring that your variables are set. For example:

```bash
# In ~/.bashrc or ~/.zshrc
export DATABASE_URL="mongodb://your_mongodb_uri"
export API_KEY="your_secret_api_key"
export PORT=3000
```

After adding these lines, you'll need to apply the changes. You can do this either by sourcing the file using `source ~/.bashrc` (or `source ~/.zshrc`) in the current session, or simply close and re-open your SSH session so these settings take effect in the new session.

Within your Node.js application, you can access these variables using `process.env`, like this:

```javascript
// server.js
const express = require('express');
const app = express();

const databaseUrl = process.env.DATABASE_URL;
const apiKey = process.env.API_KEY;
const port = process.env.PORT || 3000; // Default to 3000 if PORT isn't defined

app.get('/', (req, res) => {
  res.send(`Database URL: ${databaseUrl}, API Key (masked): ${apiKey.substring(0, 5)}...`);
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
```
This snippet illustrates how your Node.js application retrieves the variables set in your bash profile. Note the defensive default for the `PORT` variable, which is a good practice to implement.

However, relying solely on `.bashrc` or similar for system-wide application settings can be problematic. It conflates user-specific environment settings with application configuration, and makes it difficult to manage changes without SSH access. For this reason, I prefer using systemd to manage my Node.js applications, and this approach lends itself more cleanly to proper environment variable segregation. Systemd units are configuration files that describe how a service should be run, including settings for things like start commands, restart policies, and, crucially for our purpose, environment variables.

If you are running your node application as a systemd service, here is how you would modify your systemd unit file, usually found at `/etc/systemd/system/<your_service_name>.service`.

Suppose your service file initially looks something like this:

```ini
[Unit]
Description=My Node.js App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/my-app
ExecStart=/usr/bin/node server.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
To set environment variables, you'd modify the `[Service]` section:

```ini
[Unit]
Description=My Node.js App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/my-app
Environment="DATABASE_URL=mongodb://your_mongodb_uri"
Environment="API_KEY=your_secret_api_key"
Environment="PORT=3000"
ExecStart=/usr/bin/node server.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

After modifying the service file, you'll need to reload the systemd daemon and restart your service for the changes to take effect:

```bash
sudo systemctl daemon-reload
sudo systemctl restart <your_service_name>
```

By setting variables in the systemd unit, you ensure that they're only available to that particular service, thus preventing accidental leakage into other processes running on the same server. Moreover, it provides a central place for managing the environment for your specific application. This is important when scaling, using CI/CD pipelines, or automating infrastructure management using tools like ansible.

There are other advanced techniques, such as utilizing dotenv files (a popular library), which involve reading environment variables from a dedicated `.env` file. This pattern is more suitable for development and local environments. While using `.env` files is beneficial for keeping settings out of version control, their use in production comes with some considerations. Be sure to secure that file from unauthorized access, and always evaluate your security posture.

In my experience, using a combination of `.bashrc` or `.zshrc` for user specific settings, and systemd for application-specific configuration is a practical method for ensuring both flexibility and organization in a Lightsail environment. When setting environment variables via systemd, you gain a specific advantage, as you can explicitly set the user under which the application is run. However, the simplicity of `export` command in a shell configuration should not be overlooked for some use cases, especially if a process is being run directly from the shell.

For deeper reading, I’d highly recommend diving into the systemd documentation directly; it can be found on the freedesktop.org website or via your distribution's man pages (`man systemd.unit`, for example). For a better understanding of process environments, "Advanced Programming in the Unix Environment" by W. Richard Stevens and Stephen A. Rago is still an indispensable classic. Additionally, the Node.js documentation provides valuable information regarding the `process` global object, which has a dedicated section on environment variables. Understanding the fundamental principles of process management and user environments is important when configuring systems like Lightsail. Lastly, ensure to adopt a defense-in-depth approach to security. Avoid storing secrets directly in your code.

The information above represents my experience. There are numerous methods to get the job done, and each situation may call for a specific approach. The most important things are understanding how these environments work and implementing what suits your team's specific needs and workflows.
