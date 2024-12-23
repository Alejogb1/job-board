---
title: "How can Docker Compose be used with AWS .ebextensions?"
date: "2024-12-23"
id: "how-can-docker-compose-be-used-with-aws-ebextensions"
---

Okay, let’s tackle this. I've spent quite a bit of time wrestling… well, *dealing* with the intricacies of integrating Docker Compose within AWS Elastic Beanstalk, particularly when .ebextensions come into play. It's a powerful combination, but it requires a clear understanding of how these two systems interact. It isn't always straightforward, and I've certainly learned some lessons along the way.

The core challenge lies in the inherent architecture of Elastic Beanstalk. Ebextensions are essentially configuration files that Elastic Beanstalk interprets *before* the application deployment itself takes place. Docker Compose, on the other hand, is typically used to orchestrate containerized applications. So, the trick is getting Elastic Beanstalk to recognize and use your Docker Compose setup, rather than trying to force it into a conventional application deployment. The .ebextensions give us the hooks we need to make this happen.

Here's the strategy: we're essentially using .ebextensions to provision the environment and then, importantly, to trigger docker compose commands to bring our application containers up after the base environment is ready. We aren't forcing Elastic Beanstalk to ‘understand’ our docker compose file; instead, we are just using it to execute the necessary docker commands.

The crucial step, and one I've seen trip up a few people, is correctly configuring the `container_commands` section of our .ebextensions configuration file. These commands are run in order after the application has been unpacked onto the instance, but before the application is started. This makes it ideal for running docker commands. And this means that Elastic Beanstalk expects the location of your application code is what it is deploying in its environment. Therefore, one crucial thing is that the docker-compose.yml file, along with any supporting dockerfiles must be present in the root directory of the zip file uploaded to Elastic Beanstalk.

Let’s take a look at a basic example. Let's say we've got a simple application described by a `docker-compose.yml` file that defines two services: a web frontend and a backend api, all inside a folder called `app`.

First, our `docker-compose.yml` file (in the app directory):

```yaml
version: "3.8"
services:
  web:
    build:
      context: ./web
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - api
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
     - "3000:3000"
```

Now, let's create the basic Dockerfiles for each service. In `app/web/Dockerfile`:

```dockerfile
FROM nginx:latest
COPY ./html /usr/share/nginx/html/
```

And in `app/api/Dockerfile` (assuming a very simple node.js api):

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

And finally, let’s say our `app/web/html/index.html` is just:

```html
<h1>Hello from the web service</h1>
```

And our `app/api/index.js` is:

```javascript
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello from the api service');
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
```
and its `package.json` is simply:

```json
{
  "name": "simpleapi",
  "version": "1.0.0",
  "description": "simple api test",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
```

Now, with these files in our `app` folder, we must then place our `.ebextensions` configuration file in a folder named `.ebextensions` at the root of the same directory as the `app` folder (which becomes our root folder when we zip it up). We will call the file `docker-compose.config`

Here's how we’d configure our `.ebextensions/docker-compose.config` to make this work:

```yaml
packages:
  yum:
    docker: []

container_commands:
  01_create_compose_folder:
    command: "mkdir /opt/compose"
  02_copy_docker_compose:
    command: "cp /var/app/current/app/docker-compose.yml /opt/compose/"
  03_change_directory:
    command: "cd /opt/compose/"
  04_docker_compose_up:
    command: "docker-compose up -d"
```
(Note: the order of these commands matters!)

In this example, I'm installing docker using the `packages` block (though many eb environments come with docker installed already, but it is included here for clarity), then creating a folder for docker-compose, copying over the `docker-compose.yml` file from the application source (which is unzipped to `/var/app/current/` by Elastic Beanstalk), then navigating to that folder and finally running the `docker-compose up -d` command.

A more complex scenario involves managing environment variables within Docker Compose. This is where things get interesting because, generally, the variables we use in Elastic Beanstalk aren't directly passed into our docker containers. We can't just rely on docker compose's environment functionality directly here, because that would mean exposing sensitive information in the `docker-compose.yml` file.

Instead, we can use the `.ebextensions` to read the environment variables configured in the Elastic Beanstalk console and then pass them to our containers through the docker compose command.

Let’s say we've got an environment variable named `API_KEY` in Elastic Beanstalk. Here's how we modify our .ebextensions file to leverage that for a container, using `sed`:

```yaml
packages:
  yum:
    docker: []

container_commands:
  01_create_compose_folder:
    command: "mkdir /opt/compose"
  02_copy_docker_compose:
    command: "cp /var/app/current/app/docker-compose.yml /opt/compose/"
  03_copy_env:
      command: "cp /var/app/current/app/.env /opt/compose/"
  04_inject_vars:
    command: 'echo "API_KEY=$API_KEY" >> /opt/compose/.env'
  05_change_directory:
    command: "cd /opt/compose/"
  06_docker_compose_up:
    command: "docker-compose --env-file .env up -d"
```

In this revised config, we first create an empty `.env` file in our compose folder. Then, crucially, we use `sed` to insert the environment variable `API_KEY` from the Elastic Beanstalk environment into the .env file, and then we pass that file into the `docker compose up` command. This ensures that secrets are not hardcoded into any files. The .env file can also be used in the dockerfiles (although I personally find that less transparent), using the ARG directive and ENV directives of the Dockerfile.

Finally, let's consider a scenario where you need to do some pre-configuration of docker before running your compose file, perhaps to adjust resources.  Here's an example where we are increasing the docker memory allocation before running docker compose:

```yaml
packages:
  yum:
    docker: []
files:
  "/etc/docker/daemon.json":
    mode: "000644"
    owner: root
    group: root
    content: |
      {
        "default-ulimits": {
             "nofile": {
                "Name": "nofile",
                "Hard": 65535,
                "Soft": 65535
            }
        },
        "memory": "10g"
      }

container_commands:
  01_restart_docker:
    command: "sudo systemctl restart docker"
  02_create_compose_folder:
    command: "mkdir /opt/compose"
  03_copy_docker_compose:
    command: "cp /var/app/current/app/docker-compose.yml /opt/compose/"
  04_change_directory:
    command: "cd /opt/compose/"
  05_docker_compose_up:
    command: "docker-compose up -d"
```
Here, we're using the `files` section to create a new `daemon.json` file that modifies the memory resources docker is allowed to use, before restarting the daemon using `systemctl`. The rest of the process is similar to the previous example.

For a more thorough dive into this sort of deployment strategy, I highly recommend looking into the official Elastic Beanstalk documentation, of course.  Also, "Docker Deep Dive" by Nigel Poulton provides an excellent understanding of docker internals, and understanding docker well will make using it via Elastic Beanstalk much easier. For in-depth knowledge on cloud infrastructure, "Cloud Native Patterns" by Cornelia Davis is also valuable. You will also find great value in exploring the docker compose documentation directly on docker's site. These resources have personally been quite helpful in my understanding and use of these technologies.

In summary, integrating docker compose with .ebextensions involves using the container commands to provision your environment, copy your docker compose file into a working directory, and finally run the `docker-compose up` command. By effectively using `.ebextensions`, you can maintain your docker orchestration while still leveraging the managed services of Elastic Beanstalk. This provides a lot of flexibility and power for complex application deployments. It requires a bit more setup than standard EB deployments, but in the long run, this makes for much more manageable and scalable setups.
