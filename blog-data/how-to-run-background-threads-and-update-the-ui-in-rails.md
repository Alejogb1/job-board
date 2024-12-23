---
title: "How to run background threads and update the UI in Rails?"
date: "2024-12-23"
id: "how-to-run-background-threads-and-update-the-ui-in-rails"
---

, let's tackle this one. Been there, done that, more times than I care to recall. Running background processes and keeping the user interface in sync is a common challenge in web applications, particularly with Rails. It's a dance between asynchronous work and the inherently synchronous nature of browser rendering. The trick is to handle it efficiently without locking up the main thread and, crucially, making sure the UI reflects the updated state in a reliable manner. I've seen more than a few implementations that crashed and burned spectacularly, usually due to lack of proper synchronization or failure to handle exceptions gracefully.

The core issue is this: when a user triggers an action that requires a lengthy process – say, generating a large report, processing a batch of images, or performing some complex data analysis – you absolutely *cannot* do that in the request/response cycle that serves the user's web page. That's how you get timeouts, stalled browsers, and frustrated users. You need to offload that work to a background thread or process. Then, you need a way to signal back to the user’s browser about the status of that process and eventually, the result.

Rails itself doesn’t provide direct, built-in mechanisms for background threading beyond what the underlying Ruby runtime offers. Instead, it encourages the use of external libraries and services that specialize in managing asynchronous workloads.

There are a few ways to handle this situation and I’ll start with the most common, which is using a queuing system combined with websockets. This is my usual starting point for anything moderately complex.

**Approach 1: Active Job with ActionCable (Websockets)**

The quintessential solution for this in modern Rails is to utilize *Active Job* to queue background tasks and *Action Cable* for real-time updates to the user’s browser. Active Job provides an abstraction layer over a variety of queuing backends (like Sidekiq, Resque, or even the database itself for very basic use cases). Action Cable, on the other hand, gives us the websocket framework we need to push updates to the client.

Here's a conceptual breakdown:

1.  **User Action:** The user triggers an action (e.g., clicks a button) on the Rails application, sending a request to your server.
2.  **Background Job Enqueued:** Your Rails controller creates a new *Active Job* and pushes it onto the queue. The controller then quickly renders an initial response back to the user, usually indicating that the background process has started (e.g., "Processing…"). This avoids blocking the UI.
3.  **Worker Process:** A background worker (e.g., a Sidekiq process) picks up the job from the queue and executes it. This can involve intensive computations, file processing, or network requests – all happening outside the main Rails thread.
4.  **Status Updates via Websocket:** Within the worker job, you'd use Action Cable to push status updates to a specific channel that the user's browser is subscribed to. These updates might indicate progress (e.g., "50% complete"), errors, or eventual success with a result.
5.  **Browser Updates:** The javascript on the client-side listens to the websocket channel and dynamically updates the user interface to reflect the ongoing progress or results.

Let me demonstrate with a simplified example. Assume we have a job that processes an image and reports back the dimensions:

```ruby
# app/jobs/process_image_job.rb
class ProcessImageJob < ApplicationJob
  queue_as :default

  def perform(image_path, user_id)
    begin
      image = MiniMagick::Image.open(image_path) # Using MiniMagick to get dimensions
      width, height = image[:dimensions]
      ActionCable.server.broadcast "image_processing_channel_#{user_id}", {status: "complete", width: width, height: height }
    rescue StandardError => e
      ActionCable.server.broadcast "image_processing_channel_#{user_id}", {status: "error", message: e.message }
      Rails.logger.error "Error processing image: #{e.message}"
    end
  end
end
```

```javascript
// app/javascript/channels/image_processing_channel.js

import consumer from "./consumer"

consumer.subscriptions.create({ channel: "ImageProcessingChannel", user_id: document.querySelector('body').dataset.userId }, {
    connected() {
      console.log("connected to channel")
      // Called when the subscription is ready for use on the server.
    },
    disconnected() {
      console.log("disconnected from channel")
      // Called when the subscription has been terminated by the server.
    },
    received(data) {
      console.log("data:", data)
      // Called when there's incoming data on the websocket for this channel
      if (data.status === "complete") {
       document.querySelector('#image-width').innerText = data.width;
       document.querySelector('#image-height').innerText = data.height;
      } else if (data.status === "error") {
        document.querySelector('#error-message').innerText = data.message;
      }

    }
  });

```

```erb
  <!-- app/views/images/show.html.erb -->
  <div data-user-id="<%= current_user.id %>">
      <p>Image processing in progress...</p>
      <p>Width: <span id="image-width"></span></p>
      <p>Height: <span id="image-height"></span></p>
      <p>Error: <span id="error-message"></span></p>
  </div>
```

In this example, the `ProcessImageJob` uses `MiniMagick` (you'd need to add it to your `Gemfile`) to extract image dimensions. It uses the user id to define a unique channel that only they are subscribed to. We could handle errors more gracefully, but this gets the point across. On the client-side, javascript subscribes to that channel and updates the DOM accordingly using selectors based on ids.

**Approach 2: Polling (Simpler, but less efficient)**

Another approach, which I would recommend avoiding for anything that is even moderately complex, is polling. The client periodically sends requests to the server checking the status of the background process. This is simpler to implement (no websockets required), but less efficient and not as responsive. It puts more load on your server and requires the client to keep making requests.

Let's say you're using a simple background task framework like delayed_job (though it is considered a bit old these days but still gets the job done). The approach would look something like this:

1.  **User Action:** The user action starts a job.
2.  **Job Started and ID Returned:** The server starts the job and returns the `job_id` to the browser.
3.  **Client-Side Polling:** Javascript uses `setTimeout` to call a status check endpoint on the server every few seconds, passing the job ID.
4.  **Status Check:** The server checks the status of the job by checking the `delayed_jobs` table.
5.  **Result:** If the job has completed, the server returns the result. The client then updates the UI.

I’m skipping the code example as, realistically, I would almost never recommend this. There are very few situations in modern applications where polling is preferred to websockets.

**Approach 3: Server-Sent Events (SSE)**

Server-Sent Events (SSE) provide a one-way communication channel from the server to the client. It's simpler than websockets, not requiring two-way interaction, and can be a useful middle ground between polling and websockets. The client subscribes to the events, and the server pushes updates as they happen. This would require writing code that manually manages that connection through a `text/event-stream` response type. The key here is that it is still a server-pushed system that can allow the client to receive updates more efficiently than polling.

I would recommend starting with Active Job and ActionCable (approach 1) as it scales really well, and the learning curve is fairly gentle. But understanding the other approaches also helps when encountering legacy systems or when the complexities of websockets are simply not worth it.

**Resources for Further Exploration**

For a deep dive into this area, I'd suggest these resources:

*   **"Concurrent Programming in Ruby" by Matthew Kirk:** An excellent book covering various concurrency patterns and techniques in Ruby, which can be crucial when dealing with background processing.
*   **"Rails 7: A Guide for Modern Web Development" by Jason Swett and Noel Rappin:** An up-to-date book on the latest versions of Rails, including best practices for background jobs and real-time interactions.
*   **The official Sidekiq documentation:** Sidekiq is very common in the Rails community and it is critical to understand it well if you want to write scalable Rails applications.
*   **The official Action Cable guides:**  It's important to understand the intricacies of how ActionCable works so you can build reliable systems.

I hope this provides a clear picture of how to tackle background threads and UI updates in Rails. Remember, handling this well is about much more than just making things work. It's about creating responsive and reliable experiences for your users. Good luck, and feel free to ask if you need more details on something.
