---
title: "Is config.ru required to run a Sinatra app inside a Docker container?"
date: "2025-01-30"
id: "is-configru-required-to-run-a-sinatra-app"
---
The necessity of `config.ru` for a Sinatra application within a Docker container hinges on the chosen execution method.  My experience deploying numerous Sinatra applications, both in production and development environments, has shown that while `config.ru` is conventionally used for Rack-based applications, its presence is not strictly mandatory when leveraging alternative execution strategies within Docker.


**1. Explanation:**

Sinatra, at its core, is a DSL built atop Rack.  Rack defines a standard interface between web servers and Ruby frameworks.  The `config.ru` file (Rackup configuration file) acts as the glue, specifying how the Rack application (your Sinatra app) should be mounted and run.  It typically contains a line like `run Sinatra::Application`.  This tells the Rack server (e.g., Puma, Thin, WEBrick) how to bootstrap your Sinatra application.

However, within the constrained environment of a Docker container, we have additional options for application execution. Dockerfiles provide flexibility beyond the traditional Rackup approach. We can execute the Sinatra application directly using a command within the Dockerfile's `CMD` instruction or through a process manager like Supervisor, eliminating the direct reliance on `config.ru`.


**2. Code Examples:**

**Example 1: Traditional Rackup Approach (Requires config.ru)**

This approach mirrors a typical Sinatra deployment outside of Docker.  It relies on the `config.ru` file and a Rack-compatible server like Puma.

```ruby
# config.ru
require './app'
run Sinatra::Application

# app.rb (Sinatra application)
require 'sinatra'

get '/' do
  'Hello from Sinatra!'
end

# Dockerfile
FROM ruby:3.1

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .

CMD ["puma", "-C", "config.puma"]
```

This configuration is straightforward. The Dockerfile installs dependencies, copies the application code, and then executes Puma using a `config.puma` file (not shown, but would contain Puma-specific configurations). This requires the `config.ru` file to correctly bootstrap the Sinatra application.

**Example 2: Direct Execution within the Dockerfile (No config.ru required)**

This method bypasses `config.ru` by directly executing the Sinatra application within the Dockerfile's `CMD` instruction using a suitable server.  This requires a server which can be invoked directly, providing more control over the execution.

```ruby
# app.rb (Sinatra application - same as Example 1)
require 'sinatra'

get '/' do
  'Hello from Sinatra!'
end

# Dockerfile
FROM ruby:3.1

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .

CMD ["ruby", "app.rb", "-p", "4567"] #  Using the built in webserver
```

Here, we leverage Sinatra's built-in WEBrick server for simplicity. The `-p 4567` argument specifies the port. This approach eliminates the need for `config.ru`, significantly simplifying the setup, but it's generally less suitable for production environments due to WEBrick's performance limitations.


**Example 3: Using a Process Manager (Supervisor) (No config.ru required)**

This approach offers better control and robustness, particularly in production settings.  It uses Supervisor to manage the Sinatra application's lifecycle, removing the direct dependency on `config.ru`.

```ruby
# app.rb (Sinatra application - same as Example 1)
require 'sinatra'

get '/' do
  'Hello from Sinatra!'
end

# supervisord.conf
[program:sinatra_app]
command=/usr/bin/ruby app.rb -p 4567
autostart=true
autorestart=true
stderr_logfile=/var/log/sinatra_app.err.log
stdout_logfile=/var/log/sinatra_app.out.log

# Dockerfile
FROM ruby:3.1

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .
COPY supervisord.conf /etc/supervisor/conf.d/sinatra_app.conf

CMD ["/usr/bin/supervisord", "-n"]
```

This example uses Supervisor to manage the Sinatra application.  The `supervisord.conf` file defines the application, enabling automatic start, restart, and logging. The Dockerfile copies this configuration and starts Supervisor, providing a more robust and production-ready solution.


**3. Resource Recommendations:**

* Consult the official Sinatra documentation.
* Refer to the documentation for your chosen Rack server (Puma, Thin, etc.).
* Study the Supervisor documentation for process management best practices.
* Explore the Docker documentation for image building and containerization techniques.


In conclusion, while `config.ru` plays a crucial role in the traditional Rack-based deployment of Sinatra applications, its presence is not always mandatory within a Dockerized environment.  Alternative execution strategies, demonstrated above, provide flexibility and offer more control, depending on deployment needs and scalability requirements.  The choice depends entirely on the desired level of control and the production readiness considerations of your project.  My experience indicates that for simple development or prototyping, the direct execution method is sufficient, but for production deployments, using a process manager like Supervisor provides far greater robustness and maintainability.
