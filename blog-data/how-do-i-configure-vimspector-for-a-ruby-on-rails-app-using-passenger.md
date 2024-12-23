---
title: "How do I configure vimspector for a Ruby on Rails app using Passenger?"
date: "2024-12-23"
id: "how-do-i-configure-vimspector-for-a-ruby-on-rails-app-using-passenger"
---

Okay, let's tackle vimspector configuration for a Ruby on Rails application, specifically one using Passenger. This isn’t always a straightforward setup, and I remember spending a good chunk of time ironing out the wrinkles in a past project. The key here is understanding the interplay between vimspector, the debugging adapter (in this case, debase), and Passenger's application lifecycle.

The primary challenge comes from Passenger's deployment method. It spawns application processes, often in a pool, and this can obscure the connection between your debugger and the code you're trying to examine. The solution lies in configuring the debugger to attach to one of these spawned processes correctly. It’s crucial that you don’t just launch the application through the debugger, since Passenger usually has its own process management routines. We'll focus on *attaching* to a running process.

First, ensure you have the necessary components. You'll need vimspector, of course, properly installed and configured in your vim or neovim setup. Beyond that, you absolutely must have the `debase` gem added to your `Gemfile` in development, and ensure you run `bundle install` to include it:

```ruby
group :development do
  gem 'debase'
end
```

This allows the Ruby debugging server to run and accept connections. Once that's in place, we can get into the crucial part – the `.vimspector.json` configuration file, specifically designed for *attach* scenarios.

Now, let's break down a sample configuration. This isn't a simple copy-paste solution since your exact paths, interpreters, and other environment variables may vary, but it provides a strong foundation:

```json
{
  "configurations": {
    "rails-passenger-attach": {
      "adapter": "ruby",
      "configuration": {
        "request": "attach",
        "remoteHost": "localhost",
        "remotePort": 1234,
        "localRoot": "${workspaceRoot}",
        "remoteRoot": "/path/to/your/rails/app"
      },
      "breakpoints": {
          "exception": "true"
        }
    }
  }
}
```

Let’s dissect this piece by piece. `adapter: "ruby"` specifies that we are using the ruby debugger adapter bundled with vimspector (which, under the hood, manages `debase`). `request: "attach"` is key – we're not starting a new process, we're joining one already running, spawned by Passenger. `remoteHost: "localhost"` and `remotePort: 1234` indicates the server will listen on this network location for debugger connections. The port, `1234`, is a common but arbitrary selection; you can chose a different one, so long as it's consistent across your Passenger configuration.

`localRoot` points to the root of your Rails project on your development machine (vimspector's notion of what to show), and `remoteRoot` to where the app lives on the server. Notice how i've used `${workspaceRoot}` to avoid any hardcoded absolute paths. In my experience working with dockerized environments, these paths require careful consideration. We also added `breakpoints: { "exception": "true" }` to have the debugger break on exceptions.

Now, this isn't enough on its own, you also need to configure Passenger itself to launch your application with the debugger enabled. This is done by adding the following line to your application’s entry point, often `config/environment.rb` or `config/boot.rb`:

```ruby
if defined?(Rails) && Rails.env.development?
  require 'debase/core'
  Debase.start_server('localhost', 1234)
end
```

This snippet only runs when the rails environment is in `development`, preventing a debugger from being started in other environments. It requires `debase/core`, then it starts the server on the selected host and port. Ensure this code is executed very early during your application's startup process.

Finally, let’s add a quick sanity check for when you *aren't* using the debugger in a debugging process:

```ruby
if defined?(Rails) && Rails.env.development? && !ENV['DISABLE_DEBUGGER']
    require 'debase/core'
    Debase.start_server('localhost', 1234)
  end
```

Here, I've added `!ENV['DISABLE_DEBUGGER']`. This will allow you to run your rails server without starting the debug server by setting the environment variable `DISABLE_DEBUGGER=true`. This is useful if you want to run tests or just start the server without the overhead of a debugger.

**Important Notes and Troubleshooting Tips:**

*   **Port Conflicts:** Ensure no other services are running on port 1234, or the port you chose. You can use tools like `netstat` or `lsof` to check.
*   **Firewalls:** If you're working behind a firewall, make sure the debug port is open and accessible.
*   **`remoteRoot` and `localRoot`:** These must match your project’s structure. If there's a mismatch, breakpoints won't be properly set. I once spent hours troubleshooting breakpoints that seemed to get ignored, which were ultimately caused by an improperly set `remoteRoot` due to docker mount mapping nuances.
*   **Passenger Configuration:** Passenger might have its own configurations influencing the Ruby environment. Make sure it isn’t interfering with the debugger’s ability to start or connect. Consult the passenger documentation to make sure all the configurations are correct.
*   **Network Issues:** For containerized apps, ensure your network configuration allows your host to connect to the container’s debug port. This could involve port forwarding or using specific container networking options.

**Resource Recommendations:**

*   **“Debugging with Ruby” by Jeremy McAnally:** This is a great book that goes into depth on debugging Ruby applications. It covers `debase`, which is used by the debugger, making it extremely helpful.
*   **“Passenger Documentation”:** The official Passenger documentation is the best resource for how it operates with Ruby applications. Familiarize yourself with Passenger’s process management to better debug.
*   **Vimspector Documentation:** The official vimspector documentation is excellent, though may not have specific Passenger configurations. Look for the `attach` examples and make sure you fully understand the configurations.

In summary, debugging a Passenger-deployed Rails application using vimspector hinges on correctly configuring vimspector to *attach* to an existing Ruby process spawned by Passenger. This requires proper setup of the debugger server within the Rails application itself, and an accurate configuration of your `.vimspector.json` file. The process can be nuanced, and the specific setup details of your environment (paths, ports, network settings) require careful consideration.
