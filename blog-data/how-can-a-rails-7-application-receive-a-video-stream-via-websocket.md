---
title: "How can a Rails 7 application receive a video stream via WebSocket?"
date: "2024-12-23"
id: "how-can-a-rails-7-application-receive-a-video-stream-via-websocket"
---

Alright,  I remember a project back in 2021, building a real-time surveillance system—that was a deep dive into precisely this challenge: getting video streamed into a Rails app via WebSockets. It’s certainly not the simplest setup, but completely achievable with some careful planning. You wouldn’t directly push raw video bytes through WebSockets and expect magic; that’s going to lead to performance nightmares. What we need is a strategy.

First off, let's break down the core concepts involved. We’re aiming for a real-time, or near-real-time, transmission. This implies a need to handle the video feed efficiently. Simply put, we aren't sending an entire video file every frame, but rather, encoded chunks of data which our client application, running in the browser, can reconstruct back into a viewable video. The approach i've found to be the most robust involves a process of segmentation, encoding, and then stream forwarding.

My preferred method is to use `mpeg-dash` or `hls` which are adaptive bitrate streaming technologies. In the surveillance project, we used `hls` (HTTP Live Streaming) since it has generally wider browser support compared to `mpeg-dash`. I'll focus on `hls` for this explanation, as the fundamentals are very similar.

The architecture looks roughly like this:

1.  **Video Source:** This could be a camera, a video file, or any system that can produce raw video frames.
2.  **Segmentation and Encoding:** The raw frames are fed into an encoder (typically using something like ffmpeg) which converts them into a format that's suitable for web streaming, i.e., `.ts` (Transport Stream) segments and generates a playlist file (.m3u8).
3.  **WebSocket Server (Rails):** A Rails application establishes a WebSocket server which acts as the intermediary to relay the media stream.
4.  **WebSocket Client (Browser):** The browser receives the stream from the server and utilizes a library like `hls.js` to decode and display the video.

Now, let's get into the code examples. We'll skip the source setup as that depends on the camera or source itself. We will however cover the encoding and websocket portion.

**Example 1: Generating HLS Segments with `ffmpeg`**

This is the shell command we would employ to get video converted and segmented with `ffmpeg`:

```bash
ffmpeg -i rtmp://[your_source_url] \
-c:v libx264 \
-preset veryfast \
-tune zerolatency \
-c:a aac \
-ar 44100 \
-ac 1 \
-f hls \
-hls_time 2 \
-hls_list_size 5 \
-hls_flags delete_segments \
-hls_base_url http://localhost:3000/hls/ \
-hls_segment_filename /path/to/your/public/hls/segment_%05d.ts \
/path/to/your/public/hls/playlist.m3u8
```

Explanation:

*   `-i rtmp://[your_source_url]`: This specifies the input source of the video stream, assuming it's an RTMP stream. Change this to match your video source.
*   `-c:v libx264`:  Sets the video encoder to x264, a widely supported encoder.
*   `-preset veryfast`: This increases encoding speed at the expense of some compression efficiency; a trade-off we often make for near real-time streaming.
*   `-tune zerolatency`: Optimizes the encoder for lower latency streaming.
*   `-c:a aac`: Sets the audio encoder to AAC.
*   `-ar 44100`, `-ac 1`: Set audio sample rate and number of audio channels.
*   `-f hls`: Specifies HLS as the output format.
*   `-hls_time 2`: Sets the segment duration to 2 seconds.
*   `-hls_list_size 5`: Keeps the playlist history to 5 segments. Older segments are deleted.
*   `-hls_flags delete_segments`: Delete old segments.
*   `-hls_base_url http://localhost:3000/hls/`: Sets the base URL for the segments, matching the Rails app setup we are going to implement.
*   `-hls_segment_filename`: The path to save the segment files with a sequential naming scheme.
*   `/path/to/your/public/hls/playlist.m3u8`: The path to save the playlist file.

This command generates `.ts` segments and a `playlist.m3u8` file in the specified public directory of your Rails app (e.g., `/public/hls/`). This playlist file contains the location of the latest segments. These files will need to be accessed by the client application using the base URL specified.

**Example 2: Rails WebSocket Channel setup**

Now, in Rails, I'd implement a channel to handle the websocket connection. Consider this as a part of a `VideoStreamChannel.rb` file under your `/app/channels` directory:

```ruby
class VideoStreamChannel < ApplicationCable::Channel
  def subscribed
    stream_from "video_stream_channel"
    Rails.logger.info("Client subscribed to video_stream_channel")
  end

  def unsubscribed
    Rails.logger.info("Client unsubscribed from video_stream_channel")
  end

  def send_segment(data)
    ActionCable.server.broadcast "video_stream_channel", data: data
    Rails.logger.info("Forwarded segment via WebSocket")
  end
end
```

Here we:

*   `subscribed`: Handles a new client subscribing to the channel. We use a broadcast channel named `video_stream_channel`.
*   `unsubscribed`: Handles client disconnection.
*   `send_segment`: This is the key method. When a video segment needs to be sent to the clients, you'd invoke this method with the segment data, which it then broadcasts via the channel.

**Example 3: Rails Controller to handle the HLS playlist and serve segments**

You'll need a controller to serve the `playlist.m3u8` and `.ts` segments. Here's the code you may implement within `HlsController.rb`:

```ruby
class HlsController < ApplicationController
  def playlist
     send_file Rails.root.join('public', 'hls', 'playlist.m3u8'), type: 'application/vnd.apple.mpegurl', disposition: 'inline'
  end

  def segment
    segment_file = File.join('public', 'hls', params[:file])
    if File.exist?(segment_file)
        send_file segment_file, type: 'video/mp2t', disposition: 'inline'
    else
      head :not_found
    end
  end
end
```

And the routing in `routes.rb` would look like this:

```ruby
  get '/hls/playlist.m3u8', to: 'hls#playlist'
  get '/hls/:file', to: 'hls#segment', constraints: { file: /segment_\d+\.ts/ }

  mount ActionCable.server => '/cable'
```

What the code provides here is:

*   `playlist`: Serves the `playlist.m3u8` file.
*   `segment`: Serves individual `.ts` segments. This ensures the segments are accessible via the `hls_base_url` specified earlier in the ffmpeg command.
*   The routing for the HLS requests and the ActionCable endpoint.

Now, in your browser's JavaScript, you'd connect to the WebSocket channel. You'd also use a library like `hls.js` to play the video. The client would obtain the playlist via an HTTP request to the `/hls/playlist.m3u8` route and hand it over to `hls.js`. The library will take care of fetching segments and rendering the video.

A few closing notes for further study:

*   **Real-time Encoding**: The `ffmpeg` command I gave is a starting point. Tuning encoder settings for latency vs. quality is a critical part of a robust real-time system. Experiment with different presets and parameters.
*   **Scalability**: For larger audiences, you'll need to investigate techniques to scale the video serving layer, possibly with a CDN for distributing the `ts` segments and the `m3u8` playlist file.
*   **Security**: If the video is not public, consider implementing authentication and authorization to ensure only authorized users can access the stream.
*   **Error Handling**: Ensure that you have proper error handling in both server and client side. WebSocket connections may drop, and you should implement connection management to handle this.
*   **Resource Utilization**: Encoding requires significant CPU resources and you need to design your app in a way that can handle concurrent encodings effectively.

For resources, you want to dive into:

*   **The FFMPEG Documentation:** There's no better source to understand what can be done with `ffmpeg`. It’s a deep well of information, especially on various codecs and their parameters.
*   **RFC 8216: HTTP Live Streaming:** This is the official document specifying HLS.
*   **The `hls.js` documentation:** This will be indispensable for client-side video playback.
*   **Action Cable in Rails guides:** Familiarize yourself with the workings of websockets and broadcasting.

Building a real-time video streaming application is complex, but very achievable by methodically separating its components. Start with this base, and gradually add the necessary enhancements.
