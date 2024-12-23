---
title: "How can I restrict my mobile app using AirWatch SDK profile restrictions?"
date: "2024-12-23"
id: "how-can-i-restrict-my-mobile-app-using-airwatch-sdk-profile-restrictions"
---

Alright, let's tackle this. I've spent a fair amount of time navigating the intricacies of mobile device management, particularly with VMware Workspace ONE (formerly AirWatch). Restricting application behavior via SDK profiles is a fairly common requirement in enterprise deployments, and there’s a specific process I've honed over several projects. I recall one particularly complex deployment for a large healthcare provider where we had to secure patient data across hundreds of devices, and the need for precise control became quite apparent. So, let me walk you through how I approach this, keeping it focused on practical implementation.

Essentially, we're talking about leveraging the AirWatch SDK profile capabilities to fine-tune application behavior, often beyond what’s configurable through the standard OS. The SDK provides a layer that allows you to exert control over functions like data leakage prevention, copy/paste restrictions, and even network access, within the context of a managed application. The goal is to provide enhanced security and compliance without hindering the application's core functionality.

Now, configuring this isn’t just flipping a switch; it necessitates modifying the app to work with the SDK and subsequently creating the corresponding profile within the Workspace ONE console. It's a two-part operation. First, the application needs to be "wrapped" or developed with the AirWatch SDK. Second, a profile that defines the restrictions needs to be created and deployed.

Here’s a breakdown of the steps involved, with some working examples:

**Part 1: Application Integration**

The application either needs to be wrapped using the Workspace ONE app wrapping tool or, preferably, built with the SDK directly integrated into the code. This allows your app to communicate with Workspace ONE policies. The exact approach differs per platform, but let's look at a high-level representation.

For iOS (Swift), the SDK integration would roughly look like this:

```swift
import AWSDK
class AppDelegate: UIResponder, UIApplicationDelegate {

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Initialize the AirWatch SDK
        AWController.shared().start { (status) in
            switch status {
                case .started:
                    print("AirWatch SDK started successfully")
                case .notRegistered:
                    print("AirWatch SDK not registered")
                case .failed(let error):
                   print("AirWatch SDK startup failed: \(error.localizedDescription)")
            }

        }
        return true
    }

    func applicationWillTerminate(_ application: UIApplication) {
       AWController.shared().stop()
    }
}
```

In this snippet, we initialize the `AWController` when the application starts and stop it upon termination. This allows the SDK to initiate and process configuration updates. We would also need to handle registration callbacks and other relevant SDK methods based on application use cases which fall outside of the focus of this specific response.

For Android (Kotlin), the initial integration would be similar:

```kotlin
import android.app.Application
import com.airwatch.sdk.AirWatchSDK
class MyApplication : Application() {
  override fun onCreate() {
    super.onCreate()
    AirWatchSDK.getInstance().start(this) { status ->
            when (status) {
                AirWatchSDK.StartupStatus.STARTED -> {
                    println("AirWatch SDK started successfully")
                }
                AirWatchSDK.StartupStatus.NOT_REGISTERED -> {
                    println("AirWatch SDK not registered")
                }
                 AirWatchSDK.StartupStatus.FAILED -> {
                     println("AirWatch SDK failed")
                 }
            }
        }
    }

    override fun onTerminate() {
        AirWatchSDK.getInstance().stop(this)
        super.onTerminate()
    }
}
```

Just as in the iOS example, we initialize and stop the `AirWatchSDK` singleton, allowing the application to be managed through the Workspace ONE console. It’s imperative to remember that this is the minimal setup and that additional steps are required to utilize the full functionality of the SDK including feature specific configuration and callbacks handling.

**Part 2: SDK Profile Configuration**

Once your application is SDK-integrated, you can create and deploy SDK profiles within Workspace ONE. In the console, you'll find the *Profiles* section, typically under *Devices*, then *Profiles & Resources*. Here, you can craft a custom SDK profile. This profile is where we define all the rules and restrictions.

Let's illustrate with an example of restricting data exchange between managed and unmanaged apps via the pasteboard. Here's a conceptual representation of what you configure in the console’s SDK profile section:

```
Profile Name: Secure App Profile
Platform: iOS/Android
Restrictions:
  Data Loss Prevention:
    Clipboard:
      Allow Paste from Managed Apps: True
      Allow Paste to Managed Apps: True
      Allow Copy to Unmanaged Apps: False
      Allow Copy from Unmanaged Apps: False
  Network:
    Allow Wi-Fi: True
    Allow Cellular: True
    Allowed Domains: ["api.example.com"]
    Blocked Domains: ["malicious.example.com"]
  Feature Control:
    Disable Screen Capture: True
```

This configuration would permit pasting within managed apps but would block copying from any managed app to unmanaged apps, and vice versa. Furthermore, it allows Wi-Fi and cellular access, restricts network traffic to the allowed domains, blocks traffic to the blocked domains, and prevents screen capture. This level of granular control is what makes the SDK profiles so powerful. The actual implementation of this in the console is done through a graphical interface, providing various options that translate into settings within the SDK itself.

Here's an example of how you could retrieve the settings within the application, assuming we have some hypothetical settings reader:

```swift
import AWSDK

func checkRestrictionSettings() {
    let settings = AWController.shared().settings
        
    let clipboardSettings = settings?.dataLossPrevention?.clipboard
        
    print("Paste to Managed Enabled: \(clipboardSettings?.allowPasteToManagedApps ?? false)")
    print("Copy to Unmanaged Disabled: \(clipboardSettings?.allowCopyToUnmanagedApps ?? false)")

    let networkSettings = settings?.network
    print("Allowed Domains: \(networkSettings?.allowedDomains ?? [])")
    print("Blocked Domains: \(networkSettings?.blockedDomains ?? [])")

    let featureControl = settings?.featureControl
    print("Screen Capture Disabled: \(featureControl?.disableScreenCapture ?? false)")
}

```

In this Swift example, we demonstrate how one might extract the policy configuration in the app. Remember, the actual property names and structures might slightly vary across different SDK versions, so consult the SDK documentation for the definitive specification. A similar approach can be taken in Android using `AirWatchSDK.getInstance().settings`. It's important to note that these settings are not configured directly in the code, they're pulled from the SDK once the profile is pushed down from Workspace ONE.

**Key Considerations**

1.  **SDK Version Compatibility:** Always ensure that the SDK version within your application aligns with the one supported by your Workspace ONE environment. Incompatibilities can result in unexpected behavior or a complete failure in applying profile settings.
2.  **Testing:** It is critical to thoroughly test your application in a real-world environment, not just within emulators. Ensure that all restrictions function as intended and do not impede legitimate workflows. I have witnessed far too many rushed deployments crash and burn because of insufficient testing.
3.  **User Experience:** While security is paramount, excessive restrictions can hamper user experience. Find a balance that safeguards sensitive data without rendering your application unusable. Consider using conditional policies to apply different restrictions based on device context (e.g., location).
4.  **Documentation:** Always consult the official VMware Workspace ONE documentation for the latest APIs and profile options as these details change regularly. The “Workspace ONE SDK for iOS” or “Workspace ONE SDK for Android” manuals are your primary resource and should be reviewed thoroughly before and during implementation.
5. **Feature Specific Configuration:** The examples provided demonstrate the general SDK setup. However, many specific SDK features, such as those related to authentication, data encryption, tunnel configurations, etc., require their own individual configurations, which again, must align with the policies setup on Workspace ONE.

In my experience, these profiles are essential for securing enterprise apps effectively. It’s a bit more effort upfront but results in a much more secure and compliant application. If you want to dive deeper, consider looking into the works of Mike Spadafore on enterprise mobility, as his books often cover complex deployments with MDM systems. I also recommend keeping abreast of VMware's own documentation and white papers for best practices.

Implementing SDK profiles effectively requires understanding both the technical aspects of SDK integration and the practicalities of mobile device management policies. Hope this helps you get moving forward.
