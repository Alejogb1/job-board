---
title: "How can I color key video footage on an ARKit SCNPlane using GPUImage?"
date: "2025-01-30"
id: "how-can-i-color-key-video-footage-on"
---
Achieving selective color manipulation of video footage within an ARKit scene, specifically on an `SCNPlane`, using GPUImage requires a careful orchestration of several components. My experience building interactive AR applications highlights the need for a robust pipeline to handle the real-time nature of video processing and AR rendering. The core challenge stems from the fact that `GPUImage` operates on pixel data, while `SCNPlane` displays a texture derived from either a `CVPixelBuffer` or an `UIImage`. Bridging this gap involves converting the video frame into a `CVPixelBuffer`, passing it through `GPUImage` filters, and then rendering the processed output as a texture onto the `SCNPlane`.

The fundamental workflow involves the following steps: obtaining a video frame, converting it to a suitable input format for GPUImage, applying desired color adjustments with GPUImage, converting the GPUImage output back to a texture suitable for SceneKit, and finally updating the `SCNPlane` material's contents with this new texture. This process must occur on a per-frame basis to maintain real-time video rendering.

First, the video capture source, typically from `AVCaptureSession`, needs to output `CVPixelBuffer` objects. This ensures compatibility with both `GPUImage` and `CoreVideo`. Specifically, the video output delegate method, `captureOutput(_:didOutput:from:)`, will provide the frames, which we then need to send to the `GPUImage` pipeline.

Second, the `CVPixelBuffer` is wrapped within a `GPUImagePicture`. This allows the `GPUImage` processing chain to ingest the video frame. To implement color manipulation, one of the many available `GPUImage` filters can be used. For example, `GPUImageColorMatrixFilter` can be used to perform complex color transformations or adjustments like saturation, brightness, or specific channel alterations.

Third, the output of the `GPUImage` processing must be converted back into a texture compatible with SceneKit. This typically means rendering the filtered output from `GPUImage` into a new `CVPixelBuffer` (which can be achieved with a `GPUImageFramebuffer`) or directly to an OpenGL texture using the `GPUImage` framework's APIs. Finally, this texture is then assigned to the `SCNPlane`’s material `diffuse.contents` property. It’s crucial to ensure the target texture’s pixel format matches the material requirements.

Here are three code examples illustrating different aspects of this process:

**Example 1: Basic Video Capture and GPUImage Setup**

```swift
import AVFoundation
import GPUImage
import SceneKit

class VideoProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var videoCaptureSession: AVCaptureSession!
    var camera: AVCaptureDevice!
    var videoOutput: AVCaptureVideoDataOutput!
    var filter: GPUImageColorMatrixFilter!
    var gpuImageVideoCamera: GPUImageVideoCamera!
    var previewView: SCNView!
    var displayPlane: SCNPlane!
    var displayNode: SCNNode!
    
    init(preview: SCNView, plane: SCNPlane, node: SCNNode) {
        super.init()
        self.previewView = preview
        self.displayPlane = plane
        self.displayNode = node
        setupCamera()
        setupGPUImage()
    }

    func setupCamera() {
        videoCaptureSession = AVCaptureSession()
        camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        
        guard let camera = camera else { return }
        
        let videoInput = try! AVCaptureDeviceInput(device: camera)
        videoCaptureSession.addInput(videoInput)

        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        
        videoCaptureSession.addOutput(videoOutput)
        
        videoCaptureSession.startRunning()
    }
    
    func setupGPUImage() {
       gpuImageVideoCamera = GPUImageVideoCamera(session: videoCaptureSession)
        filter = GPUImageColorMatrixFilter()
        filter.setInputRotation(kGPUImageNoRotation, at: 0)
        gpuImageVideoCamera.addTarget(filter)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // process frames with gpuImage
          let outputImage = filter.imageFromCurrentlyProcessedOutput()
        
          guard let ciImage = outputImage else { return }
            let texture = createTexture(from: ciImage)
       
            DispatchQueue.main.async {
                 self.displayPlane.materials.first?.diffuse.contents = texture
            }
     }

  private func createTexture(from image: CIImage) -> MTLTexture? {
        let context = CIContext()
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
             return nil
        }
       
        let textureLoader = MTKTextureLoader(device: previewView.device!)
        do {
           let texture = try textureLoader.newTexture(cgImage: cgImage, options: nil)
           return texture
       } catch {
           print("Error loading texture \(error)")
        }
        return nil
    }

}
```

This example establishes a basic video capture setup using `AVCaptureSession` and configures a `GPUImageVideoCamera` to receive those frames. It also sets up a `GPUImageColorMatrixFilter`, although it doesn't apply any specific color adjustments yet. The key part is the `captureOutput` delegate method, which receives each video frame and then converts the `GPUImage` output to a `MTLTexture` which can be applied to the `SCNPlane`. It uses a helper function `createTexture` to handle texture creation.

**Example 2: Applying a Color Matrix Filter**

```swift
    func setupGPUImage() {
      gpuImageVideoCamera = GPUImageVideoCamera(session: videoCaptureSession)
       filter = GPUImageColorMatrixFilter()
       filter.setInputRotation(kGPUImageNoRotation, at: 0)
      
       let colorMatrix = GPUImageColorMatrix()
        colorMatrix.colorMatrix =  matrixForContrast(contrast: 2)
        gpuImageVideoCamera.addTarget(colorMatrix)
       
        colorMatrix.addTarget(filter)

   }

    func matrixForContrast(contrast: CGFloat) -> [CGFloat] {
         let scale : CGFloat = contrast
         return [ scale,    0,     0,     0,    0,
                  0,   scale,     0,     0,    0,
                  0,     0,   scale,     0,    0,
                  0,     0,     0,     1,    0 ]
     }
```

In this extension to the first example, a `GPUImageColorMatrixFilter` is configured to increase the contrast. The  `matrixForContrast` helper function generates the `colorMatrix` and sets it on the `colorMatrix` filter.  This shows how to manipulate the color matrix to achieve specific effects. This demonstrates how to chain filters, so the video passes through the `GPUImageColorMatrix` to modify the color values before finally passing through the `GPUImageColorMatrixFilter`, which doesn't actually do anything but is required as a terminal filter. The values in the contrast matrix can be altered to create various effects. The `colorMatrix` property expects an array of 20 values as input and this function provides a 4x5 matrix with only the scale changed.

**Example 3: Applying a specific color filter to achieve a monochrome effect**

```swift
    func setupGPUImage() {
       gpuImageVideoCamera = GPUImageVideoCamera(session: videoCaptureSession)
        filter = GPUImageColorMatrixFilter()
        filter.setInputRotation(kGPUImageNoRotation, at: 0)
       
        let colorMatrix = GPUImageColorMatrix()
        colorMatrix.colorMatrix = matrixForMonochrome(color: UIColor.red)
        gpuImageVideoCamera.addTarget(colorMatrix)
        colorMatrix.addTarget(filter)
   }


    func matrixForMonochrome(color: UIColor) -> [CGFloat] {
            var red: CGFloat = 0
            var green: CGFloat = 0
            var blue: CGFloat = 0
            var alpha: CGFloat = 0
            color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

            return [
              0.2126 * red + 0.7152 * green + 0.0722 * blue,   0.2126 * red + 0.7152 * green + 0.0722 * blue,  0.2126 * red + 0.7152 * green + 0.0722 * blue, 0, 0,
              0.2126 * red + 0.7152 * green + 0.0722 * blue,  0.2126 * red + 0.7152 * green + 0.0722 * blue, 0.2126 * red + 0.7152 * green + 0.0722 * blue, 0, 0,
               0.2126 * red + 0.7152 * green + 0.0722 * blue,  0.2126 * red + 0.7152 * green + 0.0722 * blue, 0.2126 * red + 0.7152 * green + 0.0722 * blue, 0, 0,
               0,        0,        0,   1, 0 ]

        }
```
This final extension demonstrates the creation of a monochrome filter using a different `colorMatrix`. The `matrixForMonochrome` function calculates the correct matrix to convert an RGB image to a monochromatic one using the given colour by taking the weighted average of the RGB values and setting the RGB components to this value, preserving the alpha value. These examples demonstrate that with the correct matrix values various effects can be produced.

For further learning on these topics, I recommend exploring the official Apple documentation for `AVFoundation`, `CoreVideo`, and `SceneKit`. The `GPUImage` library documentation and sample projects provide excellent insights into filter implementation and usage. Additionally, researching linear algebra related to color matrix transformations is beneficial to better understand how to design specific color filter effects. These resources offer a comprehensive learning path for mastering real-time video processing within ARKit.
