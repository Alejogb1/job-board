---
title: "How can AirPlay be prevented from a video playing within a WKWebView?"
date: "2024-12-23"
id: "how-can-airplay-be-prevented-from-a-video-playing-within-a-wkwebview"
---

, let’s dive into this. I've actually encountered this specific scenario a few times, usually in very controlled kiosk-style applications, and it always boils down to carefully managing the interaction between the `WKWebView` and the underlying native media playback mechanisms. Blocking AirPlay isn't a simple toggle; it requires a nuanced understanding of how the web view handles video elements and how iOS exposes (or doesn't expose) controls over their external presentation.

My initial experience, I recall, was working on a digital signage application where we absolutely could not allow users to mirror the content to personal screens – think very sensitive financial data on public display. The default behavior, of course, was that any video element embedded in the web view was readily available for AirPlay, and that was unacceptable. So, we embarked on a bit of an adventure to lock that down.

The key to understanding this is recognizing that `WKWebView` itself doesn’t natively provide a direct "block AirPlay" switch. Instead, we need to intervene at the point where the video is being played and effectively disable the mechanism that triggers AirPlay. This mechanism is tied to the `AVPlayer` or `AVPlayerViewController` (or equivalent) instances that the web view internally manages when rendering media.

The most reliable approach, in my experience, is to leverage the `allowsAirPlay` property on the underlying `AVPlayer` instances as much as we can influence it from within the context of the web view. However, direct access to these player instances is a no-go. We can't just reach in and set `allowsAirPlay = false`.

The strategy, then, relies on a few key techniques:

1.  **HTML5 Video Attributes:** First, and this is often the most straightforward, we can attempt to influence the behavior via the video element itself in our HTML. If the server (or if we control the served HTML) provides the HTML, we can start there. The HTML5 `<video>` tag supports several attributes, but none directly block AirPlay. However, we can influence how playback initiates which will, in turn, change what `AVPlayer` instances the underlying `WKWebView` ends up creating. For example:

    ```html
    <video controls disablePictureInPicture  autoplay id="myVideo" src="yourvideo.mp4"></video>
    ```

    While attributes such as `disablePictureInPicture` might seem related, they primarily control PiP behavior, not AirPlay. `controls` allows for the default UI, which might expose AirPlay options, which we would rather avoid. But this code provides a controlled video element, and we can attempt to start it programmatically. This will be important when we move into controlling the `AVPlayer` indirectly. If we serve this HTML and use an id, then we can communicate with it from the native iOS code.

2.  **JavaScript-based Interception:** This involves injecting javascript into the `WKWebView` at appropriate times to intercept the playback events and prevent the system from initiating AirPlay. This gives us a bit more influence over the media playback. The web view loads HTML with the video element, and we inject a script that monitors video events. If a playback attempt starts, we programatically pause the video. We can do this before the system generates a player that supports AirPlay, giving us a fighting chance to change things before it’s exposed to the user:

    ```javascript
    document.getElementById('myVideo').addEventListener('play', function(event) {
        event.preventDefault();
        this.pause();
    });
    ```
    Then, if the video is started from the HTML side, this will immediately pause the video and prevent it from generating an AVPlayer with an active AirPlay. This can also prevent the default video controls (set using the `<video controls>` attribute) from being exposed. We do have to be careful that it does not cause unwanted behaviors for the user. For instance, trying to play the video will appear to do nothing. This is where custom controls come in handy.

3.  **Native iOS Interception (Message Handlers):** This is where we gain significant control. We will leverage JavaScript to start video playback under our explicit control (i.e., by communicating with the native iOS code via messages). This is the most robust way to handle this requirement. Using `WKUserContentController`, we inject a script that listens for user actions. We use this to send a message to the native iOS side when the video should start playing. When we receive the message, we create an `AVPlayer` manually, configure it with `allowsAirPlay = false`, and start the playback. This means the `WKWebView` never makes its own `AVPlayer` object, and we control the media playback object directly.

    Here is an example of the native iOS code that would receive the message:

    ```swift
    import UIKit
    import WebKit
    import AVKit

    class ViewController: UIViewController, WKScriptMessageHandler {

        var webView: WKWebView!
        var avPlayer: AVPlayer?
        var avPlayerViewController: AVPlayerViewController?

        override func viewDidLoad() {
            super.viewDidLoad()

            let contentController = WKUserContentController()
            contentController.add(self, name: "videoHandler")

            let config = WKWebViewConfiguration()
            config.userContentController = contentController

            webView = WKWebView(frame: view.bounds, configuration: config)
            webView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            view.addSubview(webView)

            let url = Bundle.main.url(forResource: "index", withExtension: "html")!
             webView.loadFileURL(url, allowingReadAccessTo: url)

        }

         func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
                if message.name == "videoHandler", let videoURLString = message.body as? String, let videoURL = URL(string: videoURLString) {
                    
                    self.avPlayer = AVPlayer(url: videoURL)
                    self.avPlayer?.allowsAirPlay = false
                    
                    self.avPlayerViewController = AVPlayerViewController()
                    self.avPlayerViewController?.player = self.avPlayer
                     
                    present(avPlayerViewController!, animated: true) {
                        self.avPlayer?.play()
                    }
                }
            }
    }
    ```
    Here, we load an HTML file. When the html side sends a message to the "videoHandler", it will trigger this code, which loads the video and presents it in an `AVPlayerViewController`. The `allowsAirPlay` is explicitly set to `false`. The HTML side javascript might look something like this, which triggers the native iOS player:

    ```html
     <video id="myVideo" src="yourvideo.mp4" style="display:none;"></video>
      <button onclick="startVideo()">Play Video</button>

        <script>
            function startVideo(){
                var videoUrl = document.getElementById('myVideo').src;
                window.webkit.messageHandlers.videoHandler.postMessage(videoUrl);
            }
        </script>
    ```

In this example, we hide the HTML video element and load it using an explicit button. The user clicks the button, and the javascript sends a message to the native iOS code. The native iOS code receives the video URL and creates an `AVPlayer` using that URL, allowing us to control the `allowsAirPlay` property. It’s a more involved setup, but it offers the most reliable method for preventing AirPlay in this specific context.

Regarding resources, I’d strongly recommend diving deep into Apple’s documentation on `WKWebView`, `WKUserContentController`, `AVPlayer`, and `AVPlayerViewController`. The *Programming with AVFoundation* guide is a cornerstone resource. Another insightful book is *Advanced iOS Application Architecture* by Mark Moeykens. These will give a good foundation to build a strong understanding of how to manipulate media playback.

In short, blocking AirPlay in `WKWebView` is a careful game of control. It's about intercepting playback attempts, understanding how web elements interact with native components, and crafting a strategy that aligns with the specific application requirements. Start with the basics, and then work towards a more comprehensive solution to effectively mitigate any unintended external playback.
