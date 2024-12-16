---
title: "How can I run background threads that update the UI in Rails?"
date: "2024-12-16"
id: "how-can-i-run-background-threads-that-update-the-ui-in-rails"
---

Okay, let's tackle this. I’ve seen this particular scenario play out countless times in various projects, and while the core concepts are fairly straightforward, the implementation details often trip people up. The fundamental problem lies in Rails’ single-threaded nature when handling requests, and the fact that UI updates in web applications are inherently synchronous. Attempting to perform lengthy operations directly in a request cycle blocks the user interface, leading to a poor user experience. So, we need to offload these processes to background threads or processes, ensuring that any interaction with the UI remains fluid.

The challenge, as you've probably guessed, is that these background threads or processes cannot directly manipulate the DOM or modify the UI elements in the user’s browser. That responsibility falls squarely on the main thread that is processing a given user request. Our goal then, isn't to have a background thread *directly* manipulate the UI, but rather to have it update data that the UI then reacts to via a refresh or some similar mechanism.

Over the years, I've explored different architectures to address this. Early on, in the "Wild West" days of Web 2.0, we'd rely heavily on client-side polling, which was inefficient and often overwhelmed the server. Today, we have far better options, leveraging techniques like websockets, server-sent events (SSE), or simply polling judiciously with well-structured server-side processes and proper caching. The key is to understand the specific requirements of your application when choosing a solution.

Let's look at a basic situation where you have a background task that needs to update the UI to indicate progress. I encountered this exact scenario when working on an image processing application some years ago. Initially, users would upload images, and the processing would happen synchronously, leading to frustrating wait times and browser crashes. We needed to move image processing into a background task and show the user a progress bar.

Here’s a fairly straightforward approach using `ActiveJob` in conjunction with Action Cable for near real-time updates:

**Example 1: Using ActiveJob and Action Cable for near Real-Time Updates**

First, let’s define an `ActiveJob` to handle our image processing:

```ruby
# app/jobs/image_processing_job.rb
class ImageProcessingJob < ApplicationJob
  queue_as :default

  def perform(image_id)
    image = Image.find(image_id)
    total_steps = 10  # For example, 10 steps of processing
    (1..total_steps).each do |step|
      sleep 2  # Simulating work. In real world, use the actual image processing methods.
      progress = (step.to_f / total_steps * 100).round
      ActionCable.server.broadcast("image_progress_channel_#{image_id}", { progress: progress })
    end
    image.update(status: 'completed')
    ActionCable.server.broadcast("image_progress_channel_#{image_id}", { progress: 100, status: 'completed' })

  rescue StandardError => e
      image.update(status: 'failed', error_message: e.message)
      ActionCable.server.broadcast("image_progress_channel_#{image_id}", {status: 'failed', error: e.message})
  end
end
```

Next, create an Action Cable channel for our updates:

```ruby
# app/channels/image_progress_channel.rb
class ImageProgressChannel < ApplicationCable::Channel
  def subscribed
    stream_from "image_progress_channel_#{params[:image_id]}"
  end

  def unsubscribed
    stop_all_streams
  end
end
```

On the client side (JavaScript), you would subscribe to this channel and update the UI accordingly:

```javascript
// some_javascript_file.js
const imageId = document.getElementById('image_id').value; // Get it from a hidden field

consumer.subscriptions.create({ channel: "ImageProgressChannel", image_id: imageId }, {
  connected() {
    console.log("Connected to the ImageProgressChannel for image:", imageId);
  },

  disconnected() {
    console.log("Disconnected from the ImageProgressChannel for image:", imageId);
  },
  received(data) {
    console.log("Received Data:", data);
     if (data.progress) {
          document.getElementById("progress-bar").value = data.progress;
          document.getElementById("progress-percentage").textContent = data.progress + '%';
      } else if (data.status) {
          document.getElementById("progress-status").textContent = data.status;
          if (data.status === "completed") {
              document.getElementById("progress-container").classList.add("completed");
          }
          if (data.status === "failed") {
              document.getElementById("progress-container").classList.add("failed");
              document.getElementById("progress-error").textContent = data.error;
          }
      }
  }
});
```

This snippet demonstrates a basic mechanism where the backend actively pushes updates to the UI using websockets.

**Example 2: Polling with Ajax**

Now, let's look at an alternative approach using polling, which might be simpler for less demanding scenarios. I’ve used this successfully in situations where near real-time updates aren't critical, like batch processing tasks.

First, we start the task in a background job and update the database record on completion:

```ruby
# app/jobs/data_processing_job.rb
class DataProcessingJob < ApplicationJob
  queue_as :default

  def perform(data_item_id)
    data_item = DataItem.find(data_item_id)
    sleep 10  # Simulating some long operation
    data_item.update(status: 'processed', result: 'processed successfully')
  rescue StandardError => e
      data_item.update(status: 'failed', result: e.message)
  end
end
```

Then, on the UI side, you use Javascript to poll an endpoint to retrieve the data item’s status using `fetch`.

```javascript
//  some_javascript_file.js
function checkStatus() {
    const dataItemId = document.getElementById('data_item_id').value; // get data id from hidden field
    fetch(`/data_items/${dataItemId}/status`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("status_indicator").textContent = `Status: ${data.status}`;
             if (data.status === 'processed') {
                  document.getElementById("result_indicator").textContent = data.result;
                  clearInterval(pollInterval); //stop polling on completion
             }
             if (data.status === 'failed'){
                 document.getElementById("result_indicator").textContent = data.result;
                  clearInterval(pollInterval); //stop polling on failure
             }

        })
        .catch(error => console.error("Error fetching status:", error));
}
const pollInterval = setInterval(checkStatus, 5000);  // Poll every 5 seconds, adjust as needed
checkStatus(); // run checkStatus immediately to get first status
```

Finally, in the Rails controller, create a method to respond to the polling request:

```ruby
# app/controllers/data_items_controller.rb
class DataItemsController < ApplicationController
  def status
    @data_item = DataItem.find(params[:id])
    respond_to do |format|
      format.json { render json: { status: @data_item.status, result: @data_item.result } }
    end
  end
end

#routes.rb
get '/data_items/:id/status', to: 'data_items#status'
```
This setup demonstrates basic polling; note you will need to build out full `CRUD` for `data_item` in a real application. This offers a less demanding mechanism for updating the user interface, though it does come with the caveats of server overload through excessive polling and delayed responses.

**Example 3: Using Sidekiq and Server-Sent Events (SSE)**

For more demanding and continuous processes where server-side push is beneficial but not as complex as web sockets, server-sent events can be a good compromise.  Let’s say you have a long reporting process. I have used this often to handle generation of large PDFs or CSV files. Sidekiq can handle the background processing, and SSE pushes results.

Here is our worker:

```ruby
# app/workers/report_generation_worker.rb
class ReportGenerationWorker
    include Sidekiq::Worker

    def perform(report_id)
      report = Report.find(report_id)
      #Simulating processing in chunks and writing status to the database.
        total_steps = 10
        (1..total_steps).each do |step|
           sleep 1 #Simulating processing
            progress = (step.to_f / total_steps * 100).round
            report.update(progress: progress, status: 'processing')
            ActionCable.server.broadcast("report_progress_channel_#{report_id}", { progress: progress })
         end
      # Generate the report and store it.
      report.update(status: 'completed', report_url: '/reports/download')
      ActionCable.server.broadcast("report_progress_channel_#{report_id}", {progress: 100, status: 'completed'})
  rescue StandardError => e
      report.update(status: 'failed', error_message: e.message)
      ActionCable.server.broadcast("report_progress_channel_#{report_id}", {status: 'failed', error: e.message})
  end
end

```

Again, we use an `ActionCable` channel to broadcast the updates. The client-side setup would be similar to example one, adapting it to the `report_progress_channel`.

```javascript
//some_javascript_file.js

const reportId = document.getElementById('report_id').value; // Get it from a hidden field

consumer.subscriptions.create({ channel: "ReportProgressChannel", report_id: reportId }, {
  connected() {
    console.log("Connected to the ReportProgressChannel for report:", reportId);
  },

  disconnected() {
    console.log("Disconnected from the ReportProgressChannel for report:", reportId);
  },
  received(data) {
    console.log("Received Data:", data);
     if (data.progress) {
          document.getElementById("progress-bar").value = data.progress;
          document.getElementById("progress-percentage").textContent = data.progress + '%';
      } else if (data.status) {
          document.getElementById("progress-status").textContent = data.status;
          if (data.status === "completed") {
                document.getElementById("download-button").classList.remove("hidden");
          }
          if (data.status === "failed") {
              document.getElementById("progress-container").classList.add("failed");
              document.getElementById("progress-error").textContent = data.error;
          }
      }
  }
});
```

In this example, Sidekiq processes the report generation, while Action Cable streams updates to the client.

For further reading, I’d recommend exploring the following:

*   **"Concurrent Programming in Erlang" by Joe Armstrong:** Though not Rails specific, this book offers excellent insights into concurrency and message passing paradigms.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** This book provides a strong foundation on distributed systems and real-time data processing, which helps in understanding the core challenges in background processing and UI updates.
*   **The Rails documentation for Active Job and Action Cable:** This is essential for mastering the specific implementation within Rails.

Remember that each scenario has its own trade-offs. Understanding your requirements for speed, response time, reliability, and server load is critical. There isn't a one-size-fits-all solution, but these techniques will equip you with practical solutions for the majority of situations.
