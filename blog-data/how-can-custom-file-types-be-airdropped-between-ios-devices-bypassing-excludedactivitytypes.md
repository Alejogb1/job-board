---
title: "How can custom file types be AirDropped between iOS devices, bypassing `excludedActivityTypes`?"
date: "2024-12-23"
id: "how-can-custom-file-types-be-airdropped-between-ios-devices-bypassing-excludedactivitytypes"
---

Alright, let's talk about AirDropping custom file types, specifically how to navigate around those pesky `excludedActivityTypes` and get your files where they need to go. I've bumped into this particular challenge a few times, most notably during a project where we were developing a custom annotation tool for medical imaging data. We had these proprietary file formats that, frankly, iOS didn't understand natively, and sharing them via the standard AirDrop mechanism was... frustrating, to say the least. The built-in `excludedActivityTypes` were definitely a hurdle we needed to overcome.

The core issue here revolves around the *interaction* between your application and the system’s sharing mechanism. When you attempt to AirDrop a file, the system examines its type (identified by its UTI, or Uniform Type Identifier) and then looks at the default and custom activities available for that type. If your custom UTI isn't recognized by the standard activities or has been deliberately excluded, the file won't show up in the share sheet for direct AirDrop, or might be restricted to a specific activity rather than a generic AirDrop option.

The first step is always to ensure your app *correctly* declares the custom UTI. This is handled through your `Info.plist` file. You need to specify the supported document types and associated UTIs under the `CFBundleDocumentTypes` key. Within each document type, specify the appropriate extensions, the declared UTI, and most importantly, the `LSTypeIsPackage` flag if your file is actually a directory masquerading as a single file (which is quite common). Incorrectly setting this up is the primary reason why files don't appear or why they appear but are not shareable via AirDrop.

Here's a snippet of how your `Info.plist` might look for, let’s say, a custom file type `.mdf` (Medical Data File):

```xml
<key>CFBundleDocumentTypes</key>
<array>
    <dict>
        <key>CFBundleTypeName</key>
        <string>Medical Data File</string>
        <key>CFBundleTypeRole</key>
        <string>Editor</string>
        <key>LSHandlerRank</key>
        <string>Owner</string>
        <key>LSItemContentTypes</key>
        <array>
            <string>com.example.medicaldatafile</string>
        </array>
         <key>CFBundleTypeExtensions</key>
        <array>
          <string>mdf</string>
        </array>
    </dict>
</array>

<key>UTExportedTypeDeclarations</key>
<array>
    <dict>
        <key>UTTypeIdentifier</key>
        <string>com.example.medicaldatafile</string>
        <key>UTTypeDescription</key>
        <string>Medical Data File</string>
        <key>UTTypeConformsTo</key>
        <array>
            <string>public.data</string>
        </array>
         <key>UTTypeTagSpecification</key>
        <dict>
            <key>public.filename-extension</key>
            <array>
                <string>mdf</string>
            </array>
        </dict>
    </dict>
</array>
```

Notice that `com.example.medicaldatafile` is the actual UTI and that both `CFBundleDocumentTypes` and `UTExportedTypeDeclarations` correctly declare it along with its associated extension `.mdf`. Failing to declare this in both places is a frequent mistake I've seen. The `UTTypeConformsTo` is also important; specifying `public.data` makes your type act as generic binary data. If it's more structured, you might conform to a more specific type.

Now, let's move on to bypassing `excludedActivityTypes`. The crucial thing to understand is that `excludedActivityTypes` are designed to *restrict* activities, not dictate availability. If your app isn't actively offering the data for standard AirDrop, excluding specific activities will not somehow make it appear. To effectively share your custom file type using AirDrop, you primarily need to use `UIActivityViewController`. You need to provide a `UIActivityItemSource` that produces an object for the `UIActivityViewController` to share. That source needs to know how to provide the custom data.

Here is a Swift snippet illustrating the approach:

```swift
import UIKit
import MobileCoreServices

class MedicalDataActivityItemSource: NSObject, UIActivityItemSource {
    let fileURL: URL

    init(fileURL: URL) {
        self.fileURL = fileURL
    }

    func activityViewControllerPlaceholderItem(_ activityViewController: UIActivityViewController) -> Any {
        // Placeholder can be anything; we'll use a URL for simplicity
        return fileURL
    }

    func activityViewController(_ activityViewController: UIActivityViewController, itemForActivityType activityType: UIActivity.ActivityType?) -> Any? {
        // Now we provide the actual file URL for sharing
        if activityType == .airDrop {
           return fileURL
        } else {
            // Handle other activity types as needed. For example: email, messages
           return nil
        }
    }

    func activityViewController(_ activityViewController: UIActivityViewController, subjectForActivityType activityType: UIActivity.ActivityType?) -> String {
        // Optionally provide a subject
        return "Medical Data File"
    }

     func activityViewController(_ activityViewController: UIActivityViewController, dataTypeIdentifierForActivityType activityType: UIActivity.ActivityType?) -> String? {
         return kUTTypeData as String
    }
}


func shareMedicalData(from viewController: UIViewController, fileURL: URL) {
        let itemSource = MedicalDataActivityItemSource(fileURL: fileURL)

        let activityViewController = UIActivityViewController(activityItems: [itemSource], applicationActivities: nil)
       activityViewController.excludedActivityTypes = [] // We remove exclusions so standard AirDrop is shown

        viewController.present(activityViewController, animated: true, completion: nil)
}

// somewhere in your view controller or relevant code location
func initiateAirDrop(){
    // assume fileURL points to your custom file
    let fileURL = URL(fileURLWithPath: "your/file/path/custom.mdf")
     shareMedicalData(from: self, fileURL: fileURL)
}

```

The crucial part is that we’re providing the *file URL* as the item to share, and within `itemForActivityType`, we're specifically handling the `.airDrop` case, ensuring that a valid data source is present for the system to use for AirDrop.  Also notice we set the `excludedActivityTypes` to an empty array. This step ensures that the share sheet presents all available sharing options, including AirDrop. I’ve often seen developers mistakenly try to *force* airDrop through excluding other activities, which is backwards. You need to explicitly make the data shareable, then optionally limit other activities.

Lastly, let’s explore handling receiving the custom file on the destination device.  You need to ensure your app is registered to handle the custom file type. You'll primarily achieve this via your `AppDelegate`. You should implement `application(_:open:options:)` delegate method. Here's how that code may look:

```swift
import UIKit

class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        return true
    }

    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey : Any] = [:]) -> Bool {
        // Handle opening the file
        print("received file at url: \(url)")
        // Example - trigger a processing action or present a view controller
       handleReceivedFile(at: url)
       return true
    }

    func handleReceivedFile(at url:URL){
       // here your logic to process the received file

       // example: move file to your app's documents directory
       let fileManager = FileManager.default
       do {
             let documentsDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
             let destinationURL = documentsDirectory.appendingPathComponent(url.lastPathComponent)
              if fileManager.fileExists(atPath: destinationURL.path){
                   try fileManager.removeItem(at: destinationURL)
              }
             try fileManager.moveItem(at: url, to: destinationURL)
            print("file moved to local storage: \(destinationURL)")

        }catch {
            print("error while handling file transfer \(error)")
        }

     }


    // ... other delegate methods
}
```

This snippet demonstrates the fundamental logic for handling received files. Importantly, the system passes the file's location to your app's `open` delegate method. You will need to write logic here to appropriately move, parse, and handle the data accordingly.

For more detailed understanding of UTIs, Apple’s “Uniform Type Identifiers Overview” documentation is crucial.  To master the intricacies of the share sheet, I recommend diving deep into the `UIActivityViewController` documentation and the `UIActivityItemSource` protocol. Also, for in-depth knowledge on how files are handled by iOS it is recommended to read "System File Handling and Usage" from Apple's developer documents. These references provide comprehensive guidance beyond the scope of this discussion and will significantly strengthen your understanding of these concepts.

In summary, to AirDrop custom file types effectively, you must accurately declare your UTIs, use `UIActivityViewController` with a properly configured `UIActivityItemSource`, ensure `excludedActivityTypes` isn’t preventing the AirDrop activity from appearing, and finally, implement file receiving logic within the app delegate.  Getting this process right initially will save significant debugging time and make for a much better user experience.
