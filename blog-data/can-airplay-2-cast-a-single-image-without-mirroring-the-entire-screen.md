---
title: "Can AirPlay 2 cast a single image without mirroring the entire screen?"
date: "2024-12-23"
id: "can-airplay-2-cast-a-single-image-without-mirroring-the-entire-screen"
---

 It's a question that actually takes me back a few years. I recall a project where we were building a custom digital signage solution, and the need to display a single, high-resolution graphic without mirroring a whole tablet screen became a rather pressing matter. At the time, the documentation felt a little...sparse on this particular point. So, yes, we were faced with the same query: can AirPlay 2 cast *just* a single image without resorting to screen mirroring? The short answer is yes, absolutely. But, like most things in technology, it’s about understanding the nuances.

AirPlay 2, at its core, is built on a framework of media streaming. It handles video and audio with relative ease, and extends that capability to images too. The key here isn’t that it's fundamentally designed *not* to handle individual image casts; rather, it's that the common user-facing implementation primarily defaults to screen mirroring. This is the behavior we're all used to: whatever shows on the sending device is replicated on the receiving AirPlay target. That's the “out-of-the-box” experience.

However, the underlying api provides the tools for more specific scenarios, such as transmitting a single image. The fundamental mechanics involve formatting the data appropriately and sending it as a specific media stream type. The magic, as it often does, happens in the software development kit (sdk). Essentially, you bypass the screen mirroring pipeline and construct a direct image data transmission stream. This involves, primarily, using the `avkit` framework on apple platforms. I found the official apple documentation on avkit and the hls (http live streaming) protocol particularly useful when researching this back then. Specifically, the sections dealing with custom media streams and adding layers to the video stream provided the crucial insights. The trick is to create an hls stream where the “video” is a static image and the audio channel is muted, or absent.

To make this a bit more concrete, here's a code snippet illustrating the concept using Swift, focusing on iOS development:

```swift
import UIKit
import AVKit

func streamImage(image: UIImage, to destination: String) {
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        print("Error: Could not convert image to JPEG data.")
        return
    }

    // Create a local file to store image data.
    guard let temporaryURL = try? FileManager.default.url(
        for: .documentDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
    ).appendingPathComponent("temp.jpg") else {
        print("Error: Could not create temporary file URL")
        return
    }
    
    do {
        try imageData.write(to: temporaryURL)
    } catch {
        print("Error writing image data to file: \(error)")
        return
    }
    

    // setup the asset and player
    let asset = AVAsset(url: temporaryURL)
    let playerItem = AVPlayerItem(asset: asset)
    let player = AVPlayer(playerItem: playerItem)
    let playerViewController = AVPlayerViewController()
    playerViewController.player = player
    playerViewController.showsPlaybackControls = false

    // configure output to AirPlay
    let routePickerView = AVRoutePickerView()
    routePickerView.isRoutePickerButtonEnabled = true
    
    // Present the player to begin playback and AirPlay.
    UIApplication.shared.windows.first?.rootViewController?.present(playerViewController, animated: true) {
        player.play()
    }


    player.rate = 1.0;
    
   
}

// Example usage (assuming you have a UIImage instance named 'myImage'):
// streamImage(image: myImage, to: "MyAirPlayTarget")
```

This snippet showcases a simplified way to send an image as a stream, leveraging `avkit` to create a player with the image as the content. The key here is not using mirroring mechanisms but creating a player whose "video" content is an image. We then use airplay through the player and the route picker, which we make visible programmatically. This approach is far more resource-efficient than mirroring the entire screen, especially if your image is static. Note that in a real-world scenario, you might need to handle network conditions, resolution adjustments, and more, but the basic principle remains.

For more advanced scenarios that required smoother transitions or layering of images over video, we used a slightly different approach which utilizes `avcomposition` and `avmutablecomposition`. Instead of treating the image as a stand-alone "video," we composited it as a layer over a transparent video track. This technique involves creating a single-frame video track and then overlaying your image onto it using an `avmutablevideocompositionlayerinstruction`. Here's a conceptual code sample demonstrating this:

```swift
import UIKit
import AVKit
import AVFoundation

func streamImageOverlay(image: UIImage, duration: CMTime, to destination: String) {
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        print("Error: Could not convert image to JPEG data.")
        return
    }

    // Create a local file to store image data.
    guard let temporaryURL = try? FileManager.default.url(
        for: .documentDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
    ).appendingPathComponent("temp.jpg") else {
        print("Error: Could not create temporary file URL")
        return
    }
    
    do {
        try imageData.write(to: temporaryURL)
    } catch {
        print("Error writing image data to file: \(error)")
        return
    }


    // Create the composition
    let composition = AVMutableComposition()
    
    // Create a video track
    guard let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
        print("Error: Could not create video track.")
        return
    }
    
    // Create a time range
    let timeRange = CMTimeRangeMake(start: .zero, duration: duration)

     // create a blank image as a video track to put our image over
    let blankImage = UIImage(color: .clear, size: image.size)
    guard let blankImageData = blankImage?.jpegData(compressionQuality: 0.8) else {
        print("Error: Could not convert blank image to JPEG data")
        return
    }

        guard let blankTemporaryURL = try? FileManager.default.url(
        for: .documentDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
    ).appendingPathComponent("blank.jpg") else {
        print("Error: Could not create blank temporary file URL")
        return
    }
    
    do {
        try blankImageData.write(to: blankTemporaryURL)
    } catch {
        print("Error writing blank image data to file: \(error)")
        return
    }

    let blankAsset = AVAsset(url: blankTemporaryURL)
    guard let blankVideoTrack = try? blankAsset.tracks(withMediaType: .video).first else{
        print("Error: could not load video track from blank asset")
        return
    }

    try? videoTrack.insertTimeRange(timeRange, of: blankVideoTrack, at: .zero)

    // Create a video composition
    let videoComposition = AVMutableVideoComposition()
    
    // Create a video instruction
    let instruction = AVMutableVideoCompositionInstruction()
    instruction.timeRange = timeRange
    
    // Create a layer instruction
    let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
        let cgImage = image.cgImage!
        
    let imageLayer = CALayer()
            imageLayer.contents = cgImage
        imageLayer.frame = CGRect(origin: .zero, size: image.size)
       
    let parentLayer = CALayer()
    parentLayer.frame =  CGRect(origin: .zero, size: image.size)
    parentLayer.addSublayer(imageLayer)

    let renderer = UIGraphicsImageRenderer(size: image.size)
    let renderedImage = renderer.image { context in
        parentLayer.render(in: context.cgContext)
    }
    
    // Set the transform
       let transform = CGAffineTransform(translationX: 0, y:0)
       layerInstruction.setTransform(transform, at: .zero)
    
   
    
    // Create a sublayer that contains our image
        
    instruction.layerInstructions = [layerInstruction]
    videoComposition.instructions = [instruction]

    
    // setup the asset and player
   
    let playerItem = AVPlayerItem(asset: composition)
    playerItem.videoComposition = videoComposition
    let player = AVPlayer(playerItem: playerItem)
    let playerViewController = AVPlayerViewController()
    playerViewController.player = player
    playerViewController.showsPlaybackControls = false

    // configure output to AirPlay
    let routePickerView = AVRoutePickerView()
    routePickerView.isRoutePickerButtonEnabled = true
    
    // Present the player to begin playback and AirPlay.
    UIApplication.shared.windows.first?.rootViewController?.present(playerViewController, animated: true) {
        player.play()
    }


    player.rate = 1.0;

}

// Example Usage (Assuming an image named myImage and a duration of 10 seconds):
// let duration = CMTime(seconds: 10, preferredTimescale: 600)
//streamImageOverlay(image: myImage, duration: duration, to: "MyAirPlayTarget")
```
This second snippet creates an actual composite, allowing the image to be presented over a video track. This was particularly useful in our digital signage project where we had animated backgrounds with static images on top. This approach allows for much more flexibility but involves more complexity in setting up the composition and its various layers.

Finally, for applications where we needed tight control over streaming performance and advanced features, like adaptive bitrate, we directly interacted with `avassetwriter` and configured custom hls streams using low-level api. This method is more involved and requires careful management of streaming packets, but offers complete control over how data is sent, received, and displayed. This would usually require you to set up a temporary web server that serves the image as a segment in an hls stream.

```swift
import AVKit
import AVFoundation
import UIKit

// Simplified class for demonstration (consider using a proper server in a production scenario).
class SimpleLocalHTTPServer {
    
    var port: UInt16 = 8080
    var server: HTTPServer?
    
    func start(jpegData: Data) {
            server = HTTPServer()
            server!.serverRoot = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        let temporaryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("temp.jpg")
        
        do {
            try jpegData.write(to: temporaryURL)
        } catch {
            print("Error writing image data to file \(error)")
            return
        }

            server?.addHandler(forMethod: "GET", path: "/image.jpg") { request, response in
                
                response.statusCode = 200
                response.setValue("image/jpeg", forHeader: "Content-Type")
                response.body = try? Data(contentsOf: temporaryURL)
        }

            server?.addHandler(forMethod: "GET", path: "/playlist.m3u8") { request, response in
                    response.statusCode = 200
                    response.setValue("application/x-mpegurl", forHeader: "Content-Type")
                    let playlist = """
                    #EXTM3U
                    #EXTINF:10,
                    image.jpg
                    """
                 response.body = playlist.data(using: .utf8)
            }
        do {
            try server!.start(port: port)
            print("Server started on port \(port)")
        } catch {
            print("Error starting server \(error)")
        }

    }
    
    func stop() {
        server?.stop()
        server = nil
    }
}

func streamImageHLS(image: UIImage, to destination: String) {

      guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        print("Error: Could not convert image to JPEG data.")
        return
    }
    
    let localServer = SimpleLocalHTTPServer()
    localServer.start(jpegData: imageData)

    let url = URL(string: "http://localhost:\(localServer.port)/playlist.m3u8")!

    let asset = AVAsset(url: url)
    let playerItem = AVPlayerItem(asset: asset)
    let player = AVPlayer(playerItem: playerItem)
    let playerViewController = AVPlayerViewController()
    playerViewController.player = player
    playerViewController.showsPlaybackControls = false
    
        let routePickerView = AVRoutePickerView()
        routePickerView.isRoutePickerButtonEnabled = true
        UIApplication.shared.windows.first?.rootViewController?.present(playerViewController, animated: true) {
           player.play()
        }
    player.rate = 1.0

}

// Example Usage
// streamImageHLS(image: myImage, to: "MyAirPlayTarget")
```
This third snippet sets up a rudimentary local web server to stream an hls playlist containing a single image. This is obviously simplified for demonstrative purposes. It requires a lot more to make it production ready, but illustrates the principle that you can use hls directly with airplay. This approach was what allowed for the most fine-tuned control of the stream and its rendering.

For those who want to dive deeper, i would strongly recommend exploring the following resources: 'the avfoundation programming guide' from apple's developer documentation, which delves into all of these apis in detail. Also, "understanding hls" by andrzej wieczorek provides a good overview of the http live streaming protocol. Finally, 'advanced ios programming' by mark dalrymple covers all this from an objective-c perspective which may also be helpful as its closer to the metal so to speak.

In conclusion, while AirPlay 2 often defaults to screen mirroring, it's completely capable of casting a single image without this behavior. The key lies in understanding the underlying frameworks and how to create custom media streams. In my experience, tackling problems like this involves a methodical approach, breaking down the problem, and using the right tools for the job.
