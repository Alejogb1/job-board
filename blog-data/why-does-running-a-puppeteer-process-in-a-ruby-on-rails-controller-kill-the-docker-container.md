---
title: "Why does running a puppeteer process in a Ruby on Rails controller kill the Docker container?"
date: "2024-12-23"
id: "why-does-running-a-puppeteer-process-in-a-ruby-on-rails-controller-kill-the-docker-container"
---

Alright, let's talk about why a puppeteer process within a rails controller might be doing the unthinkable and taking down your entire docker container. I’ve seen this dance a few times, and it’s usually not pretty. It’s a complex issue rooted in resource management, process lifecycle, and a bit of Docker’s isolation magic.

In my experience, the issue almost never comes down to a bug in puppeteer itself, but more to how we're weaving it into the fabric of our rails application and the underlying container environment. Years ago, I worked on a project that required dynamically generating screenshots of user dashboards for reporting. We integrated puppeteer directly into our rails backend, thinking it would be straightforward. Spoiler alert: it wasn't. We quickly ran into the container-killing scenario, and that’s when the real debugging began.

The core problem stems from the fact that `Puppeteer` is inherently a resource-intensive process. It spins up a full-fledged Chromium browser instance in the background. This browser, even in headless mode, consumes considerable memory, cpu, and file descriptors. When launched within the context of a rails controller action, it's essentially fighting for resources alongside your web server (puma, unicorn, etc.), and other worker processes in the same container.

Here’s the crucial bit: default Docker containers often have limited resources allocated to them, especially memory and cpu. When that `Puppeteer` instance is created, it will use memory, and potentially cause it to go beyond what is set. If the container exceeds these limits, the Docker daemon, seeing it as a rogue process exceeding resources, will often terminate the entire container rather than allowing one service to starve the rest. This is especially common if your rails application is also pushing the container's resource boundaries, which happens quite a bit.

Another factor is how signals are handled inside Docker. When you terminate a process within the container, typically the main process running inside will receive a signal, most likely a `SIGTERM` or `SIGKILL`. These signals are then forwarded to child processes. Depending on how `Puppeteer` is initialized and if it properly handles these signals, your headless browser might not shut down cleanly. If it doesn't gracefully stop and return its resources quickly, this can also lead to container termination. The container runtime is essentially saying "enough is enough".

Let me illustrate this with a few simplified examples to better understand the problem and some strategies to overcome it.

**Example 1: The Naive Approach (and why it fails)**

```ruby
# app/controllers/reports_controller.rb
class ReportsController < ApplicationController
  require 'puppeteer'

  def generate_report
    Puppeteer.launch(args: ['--no-sandbox']) do |browser|
      page = browser.new_page
      page.goto('http://example.com')
      page.screenshot(path: Rails.root.join("tmp", 'report.png'))
    end
    render plain: "Report Generated"
  end
end
```

This looks innocent enough, right? However, this code will likely cause our container to die. We are running the browser directly in the web process context. If multiple requests call this endpoint simultaneously, each call spawns a new `Puppeteer` instance, consuming additional resources and escalating towards a resource-related container termination.

**Example 2: Background Processing with a dedicated worker**

A better strategy is to delegate this task to a background job system, such as `Sidekiq` or `Resque`. This decouples the `Puppeteer` process from your web server's request-response cycle.

```ruby
# app/workers/report_generator_worker.rb
class ReportGeneratorWorker
  include Sidekiq::Worker

  def perform(url, file_path)
    Puppeteer.launch(args: ['--no-sandbox']) do |browser|
      page = browser.new_page
      page.goto(url)
      page.screenshot(path: file_path)
    end
  end
end

# app/controllers/reports_controller.rb
class ReportsController < ApplicationController
  def generate_report
     file_path = Rails.root.join("tmp", "report_#{Time.now.to_i}.png")
     ReportGeneratorWorker.perform_async('http://example.com', file_path)
     render plain: "Report generation started in the background"
  end
end
```

Here, we’re pushing the puppeteer logic to a separate `Sidekiq` worker process. This means each request no longer directly spawns a browser, it simply enqueues the generation task. Workers generally handle resource management far better than the standard web server worker pool. However, depending on the volume of screenshots to be generated, it's also possible to exhaust resources allocated to the Sidekiq process, though less often than the direct process.

**Example 3: Using Docker Resource Limits (with caution)**

While not a code example, this involves configuration. You can fine-tune Docker's resource limits per container using the `docker run` or `docker-compose` commands. For example:
`docker run -m 2g --cpus=2 my-rails-app`
This command limits the container to 2 GB of RAM and 2 CPUs. This is a reactive solution, and will not prevent crashes if the application requests more memory than allocated.

The key here is to understand that docker doesn't magically manage processes for you. It isolates them, and when those isolated processes become too greedy, it steps in to terminate them. We need to be considerate about resource consumption, especially when using resource-intensive tools like `Puppeteer`.

Now, for some further reading that can help you deepen your understanding:

*   **"Operating Systems Concepts" by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne:** This book provides an excellent foundation in process management, resource allocation, and operating system principles.
*   **"Docker in Action" by Jeff Nickoloff:** Dive deep into Docker concepts, resource management, and best practices for building production-ready containers.
*   **The official Docker documentation:** Always a great resource for the most up-to-date information on Docker features, commands, and configuration.
*   **Puppeteer’s official documentation:** Contains specific instructions on how to run puppeteer in various environments, including considerations for headless mode and containerized environments.

In summary, the core issue isn't usually with puppeteer itself. It's about the environment you’re running it in. Direct execution in a web request cycle, without proper resource management or process isolation, almost always leads to container termination. By leveraging background workers and carefully considering container resource limits, you can create a stable and reliable system that uses puppeteer effectively. Always start small, monitor closely and scale resources as needed based on observation, not intuition. It will help avoid the dreaded container death.
