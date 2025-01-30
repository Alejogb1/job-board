---
title: "How can GPUImage2 be used in Flutter?"
date: "2025-01-30"
id: "how-can-gpuimage2-be-used-in-flutter"
---
GPUImage2, a powerful framework for image and video processing on iOS, can be integrated within a Flutter application via platform channels, requiring careful handling of asynchronous operations and data transfer. My experience has shown that achieving smooth, efficient real-time effects necessitates a strong understanding of both Flutter's platform interaction mechanism and GPUImage2's core architecture. Direct interop is impossible; therefore, I've consistently employed native bridge implementations.

The primary challenge lies in the disparate execution environments. Flutter, operating in its Dart virtual machine, cannot directly interact with Swift code executing GPUImage2. Platform channels, in this context, act as the communication bridge. Flutter initiates a request, which marshals data into a format understood by iOS, executes the GPUImage2 pipeline, and then marshals the results back to Flutter. This three-step process is the foundation of any successful integration. We must consider that data movement, particularly with pixel data, introduces overhead and should be carefully optimized.

**Implementation Strategy**

My approach generally revolves around creating a custom Flutter widget encapsulating a `MethodChannel` and an underlying native view using `UIKitView`. The Dart side handles user input and configurations, such as filter selections or parameters. These directives are passed to the native side, where a `GPUImageVideoCamera` or `GPUImagePicture` instance is configured, the necessary filters are appended, and the resulting frames are rendered into an `EAGLContext`, typically associated with the custom `UIView`. The rendered output is then delivered back to Flutter via the platform channel, sometimes as an encoded byte array.

Here are some concrete code examples, showcasing different parts of this process:

**Code Example 1: Dart (Flutter) - Initializing the Native View and Sending a Filter Command**

```dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class GpuImageView extends StatefulWidget {
  const GpuImageView({super.key});

  @override
  State<GpuImageView> createState() => _GpuImageViewState();
}

class _GpuImageViewState extends State<GpuImageView> {
  static const MethodChannel platform = MethodChannel('gpuimage2.channel');
  final int viewId = 0;

  String _currentFilter = 'normal';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('GPU Image Test')),
      body: Column(
          children: [
        Expanded(
          child: UiKitView(
              viewType: 'gpuimage2.view',
              creationParams: {},
            onPlatformViewCreated: (int id){
               // View has been created, now I can start interacting
            }
          ),
        ),
         Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
             children: [
                ElevatedButton(
                    onPressed: (){
                      _changeFilter("sepia");
                    },
                  child: const Text('Sepia')
                ),
               ElevatedButton(
                 onPressed: (){
                    _changeFilter('normal');
                 },
                  child: const Text('Normal'),
               ),
             ],
         )
      ],
    ),
    );
  }

  Future<void> _changeFilter(String filterName) async {
    if(_currentFilter == filterName){
      return; // avoid unnecessary calls
    }
      try {
        await platform.invokeMethod('changeFilter', {'filter': filterName});
        setState(() {
          _currentFilter = filterName;
        });
      } on PlatformException catch (e) {
        print("Failed to change filter: '${e.message}'.");
      }
  }

}
```

In this example, `GpuImageView` creates a `UiKitView` that represents the native GPUImage2 view and communicates with the native side using a `MethodChannel`. The view type 'gpuimage2.view' corresponds to the native identifier for the platform view. When a button is pressed, the `_changeFilter` function invokes the native method, passing the requested filter as a parameter. This approach facilitates dynamic configuration of the GPUImage2 pipeline. I prefer the `UiKitView` approach to avoid the complexity and performance bottlenecks often found with `Texture` widgets when rendering highly processed video.

**Code Example 2: Swift (iOS) - Setting up the GPUImage2 Pipeline and Receiving the Flutter Command**

```swift
import Flutter
import UIKit
import GPUImage

class GpuImageNativeViewFactory: NSObject, FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger
  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    return GpuImageNativeView(frame: frame, viewId: viewId, messenger: messenger)
  }

  func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
      return FlutterStandardMessageCodec.sharedInstance()
  }

}

class GpuImageNativeView: NSObject, FlutterPlatformView, FlutterMethodCallHandler {
    private var _view: UIView
    private var _gpuImageView: GPUImageView
    private var _camera: GPUImageVideoCamera?
    private var _filter: GPUImageFilter?

    private var _channel: FlutterMethodChannel?
    private var _normalFilter:GPUImageFilter?
    private var _sepiaFilter:GPUImageSepiaFilter?

  init(frame: CGRect, viewId: Int64, messenger: FlutterBinaryMessenger) {
      _view = UIView()
      _gpuImageView = GPUImageView(frame: frame)
      _view.addSubview(_gpuImageView)

      _normalFilter = GPUImageFilter()
      _sepiaFilter = GPUImageSepiaFilter()

    super.init()

      _channel = FlutterMethodChannel(name: "gpuimage2.channel", binaryMessenger: messenger)
      _channel?.setMethodCallHandler(self)

      setupCamera()
      startCamera()
  }


  func view() -> UIView {
    return _view
  }

  func onMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
      switch call.method {
      case "changeFilter":
          if let args = call.arguments as? [String: Any], let filterName = args["filter"] as? String{
              changeFilter(filterName)
              result(nil)
          }else{
             result(FlutterError(code: "INVALID_ARG", message: "Arguments are invalid", details: nil))
          }
      default:
        result(FlutterMethodNotImplemented)
    }
  }


  private func setupCamera(){
    _camera = GPUImageVideoCamera(sessionPreset: AVCaptureSession.Preset.hd1280x720.rawValue, cameraPosition: .front)
    _camera?.horizontallyMirrorFrontFacingCamera = true
      _camera?.outputImageOrientation = .portrait
  }

    private func startCamera(){
      // I will need a capture group if I want to do custom effects, but this is beyond the demo
        if let cam = _camera{
            cam.addTarget(_normalFilter!)
            _normalFilter!.addTarget(_gpuImageView)
            _camera!.startCapture()
        }

    }


    private func changeFilter(_ filterName: String){
        if(_filter != nil){
          _filter?.removeAllTargets()
        }

        switch filterName{
        case "normal":
             _camera!.removeAllTargets()
             _camera!.addTarget(_normalFilter!)
            _normalFilter!.addTarget(_gpuImageView)
           _filter = _normalFilter
            break
        case "sepia":
            _camera!.removeAllTargets()
            _camera!.addTarget(_sepiaFilter!)
           _sepiaFilter!.addTarget(_gpuImageView)
           _filter = _sepiaFilter
            break
        default:
            // Fallback to normal
           _camera!.removeAllTargets()
            _camera!.addTarget(_normalFilter!)
            _normalFilter!.addTarget(_gpuImageView)
           _filter = _normalFilter
        }
    }

}

```
This Swift code defines the `GpuImageNativeViewFactory`, which is responsible for creating instances of `GpuImageNativeView`. `GpuImageNativeView` manages the `GPUImageVideoCamera`, sets up a default `GPUImageFilter`, and handles incoming method calls from Flutter. Upon receiving the 'changeFilter' method call, the active filter is switched, modifying the GPUImage2 pipeline. `EAGLContext` management is handled by `GPUImageView`, a component that abstracts away the complexities of managing the OpenGL ES context for real-time rendering. I've made this view a `FlutterPlatformView` to guarantee I am painting in the native space, minimizing the overhead associated with transferring textures back and forth to Flutter's rendering layer.

**Code Example 3: Swift (iOS) - Registering the View Factory**

```swift
import UIKit
import Flutter

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
     let controller : FlutterViewController = window?.rootViewController as! FlutterViewController
       let gpuImageFactory = GpuImageNativeViewFactory(messenger: controller.binaryMessenger)
       controller.registrar(forPlugin: "gpuimage2-plugin")?.register(
        gpuImageFactory,
           withId: "gpuimage2.view"
      )
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
```

This snippet demonstrates the final step of registering the `GpuImageNativeViewFactory`. Within the `AppDelegate`, I retrieve the `FlutterViewController` and register the factory using a unique identifier (`"gpuimage2.view"`). This identifier is matched on the Flutter side using `UiKitView`.  This ensures the platform view is made available to flutter

**Resource Recommendations**

To achieve a robust and efficient integration, I recommend carefully studying the following areas:

1. **Flutter Platform Channels:** Understand the bidirectional communication mechanism, specifically handling asynchronous operations and data marshalling. The official Flutter documentation provides a detailed overview.

2. **GPUImage2 Documentation:** Review the API documentation to understand filter configurations, video handling, and performance considerations. This is essential for optimizing your image processing pipeline.

3. **Native View Integration:** Deeply understand the use of `UiKitView` or `Texture` (although in this specific use case, I would avoid texture rendering for performance reasons.) in Flutter. Explore how they relate to native views and their lifecycles.

4. **Memory Management:** Pay particular attention to managing resources on the native side, especially when dealing with pixel data, `GPUImage` objects and the `EAGLContext`. Proper memory management is critical to avoid memory leaks and crashes.

By addressing these key areas, one can build a robust and performant bridge between Flutter and GPUImage2, leveraging the power of GPU-accelerated image processing within a Flutter application. I recommend iterative development, starting with a simple filter and then adding complexity, carefully profiling performance at each stage.
