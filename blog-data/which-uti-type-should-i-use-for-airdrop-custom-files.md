---
title: "Which UTI type should I use for AirDrop custom files?"
date: "2024-12-23"
id: "which-uti-type-should-i-use-for-airdrop-custom-files"
---

Alright, let's tackle this. Instead of diving straight into a recommendation, let's first frame the problem and why choosing the correct uniform type identifier (uti) matters for AirDrop, especially when dealing with custom file types. I’ve definitely been down this road before, most notably when I was building a collaborative drawing app where we needed to share project files. The challenge was ensuring that the recipient’s app could open our files seamlessly. The solution ultimately came down to properly defining our uti.

The core issue is that AirDrop relies on utis to determine which applications can handle the received files. Think of a uti as a unique, system-wide identifier for a specific type of data. It's how macOS and iOS understand that a '.jpg' is an image and opens it with the Photos app. For standard file types, the system has pre-defined utis. However, when you introduce custom file formats, like those used in my past drawing app project, you need to define a custom uti to ensure a smooth user experience. Using the incorrect or generic uti can lead to scenarios where the receiving device either doesn't recognize the file type, opens it with the wrong application, or, worse, fails to open it altogether. It’s more than just file extensions; it's about semantic understanding by the operating system.

For AirDrop custom files, specifically, you’re not just dealing with a simple file transfer. You’re also asking the system to associate your custom data with an application that can parse and present it. So, a suitable uti isn't merely a matter of personal preference, it's a crucial part of defining how your data interacts with the receiving environment.

Now, let’s talk about specific recommendations. If your custom file format is a variation of an existing file type – say you're extending a text format or have some special metadata embedded in an image – consider *conforming* to an existing uti. This doesn’t mean renaming your file. Conforming involves declaring that your uti *inherits* from a system-provided uti, indicating your data's fundamental nature. For example, if your custom file contains structured text, you might conform to `public.plain-text` or a more specific variation like `public.rtf`. This is generally a good start if your format is broadly similar to an established one. It allows the system to rely on existing applications and mechanisms to some extent, while still offering some level of distinction for your custom handling.

However, if your file format is genuinely unique and doesn’t align with any known format, you’ll need to define a new, *custom* uti. This involves creating a new uti identifier and declaring the relationship of your file type to the application that can handle it. This provides full control and ensures the correct association with your application on the receiving device. This process has more steps, but the benefit is a clean separation and dedicated behavior for your file format.

Let’s get into some concrete code examples. First, a scenario where conforming is appropriate. Imagine a configuration file, `.configx`, which contains key-value pairs in plain text but is slightly different from a general `.txt` file due to specific formatting requirements. We can conform to the base plain-text uti.

```objectivec
// Example 1: Conforming to public.plain-text
#import <MobileCoreServices/MobileCoreServices.h>

NSString* utiForConfigx() {
  return (NSString*)kUTTypePlainText; // Using the constant for public.plain-text
}

// Code to create and send a file: Assume configURL is a NSURL object pointing to your file.
- (void)sendConfigFileViaAirDrop:(NSURL *)configURL {
    UIActivityViewController *activityViewController =
      [[UIActivityViewController alloc] initWithActivityItems:@[configURL] applicationActivities:nil];

    // Get the UTI and set it on the activity
    NSString* uti = utiForConfigx();
    [activityViewController setValue:@[uti] forKey:@"activityItemTypes"];

    [self presentViewController:activityViewController animated:YES completion:nil];
}
```

In this snippet, `kUTTypePlainText` is the existing, system-defined uti for plain text. By setting it as the activity item type, we tell AirDrop that this file is, essentially, a form of text, making it more likely to be openable by compatible applications (if none of them are directly associated with your custom application).

Now, let's consider the case where we have a truly custom file format – say, a `.drawfile` used in our hypothetical drawing app, which stores drawing data in a binary format. In this case, we need to define our own custom uti. This is done through `Info.plist` of your application.

```xml
<!-- Example 2: Info.plist definition of a custom UTI -->
<key>CFBundleDocumentTypes</key>
<array>
  <dict>
    <key>CFBundleTypeName</key>
    <string>Drawing File</string>
    <key>CFBundleTypeIconFiles</key>
    <array>
        <string>draw_icon</string>
    </array>
    <key>CFBundleTypeRole</key>
    <string>Editor</string>
    <key>LSHandlerRank</key>
    <string>Owner</string>
    <key>LSItemContentTypes</key>
    <array>
      <string>com.yourdomain.drawfile</string>
    </array>
  </dict>
</array>
<key>UTExportedTypeDeclarations</key>
<array>
  <dict>
    <key>UTTypeIdentifier</key>
    <string>com.yourdomain.drawfile</string>
    <key>UTTypeDescription</key>
    <string>Drawing File</string>
    <key>UTTypeConformsTo</key>
    <array>
        <string>public.data</string>
    </array>
    <key>UTTypeTagSpecification</key>
    <dict>
      <key>public.filename-extension</key>
      <array>
        <string>drawfile</string>
      </array>
    </dict>
  </dict>
</array>
```

Here, `com.yourdomain.drawfile` is our custom uti, declared in the `UTExportedTypeDeclarations` section. The `CFBundleDocumentTypes` section links that uti to the application itself. This tells the system that our app is the primary editor for files with the `drawfile` extension.

Now we can use the new custom uti in our application. The next code example showcases how to do this in the same way as before:

```objectivec
// Example 3: Using the custom UTI for a drawfile.

NSString* utiForDrawFile() {
  return @"com.yourdomain.drawfile"; // Returns the custom UTI string
}

// Code to create and send a file: Assume drawFileURL is a NSURL object pointing to your file.
- (void)sendDrawingFileViaAirDrop:(NSURL *)drawFileURL {
    UIActivityViewController *activityViewController =
        [[UIActivityViewController alloc] initWithActivityItems:@[drawFileURL] applicationActivities:nil];

    // Get the UTI and set it on the activity
    NSString* uti = utiForDrawFile();
    [activityViewController setValue:@[uti] forKey:@"activityItemTypes"];

     [self presentViewController:activityViewController animated:YES completion:nil];
}
```

In this last example we use the newly created `com.yourdomain.drawfile` uti in the application. This will tell AirDrop that only applications that have claimed support for this custom uti will be able to receive and interact with this file.

Important considerations here: When deciding between conforming and declaring a new uti, think about the *semantic* nature of your data. Is it fundamentally an existing format with some adjustments, or is it something entirely new? This choice impacts whether the system can rely on some level of existing support.

For further reading, I recommend Apple's documentation on Uniform Type Identifiers, specifically the *Document Type Declaration Reference* and *Uniform Type Identifiers Reference*. Also, the *Core Services Programming Guide*, specifically the sections related to file systems and uniform type identifiers, provides a comprehensive understanding. Another highly useful resource is *Advanced Mac OS X Programming* by Mark Dalrymple; while older, it has an excellent section about the Core Services framework including UTIs. These resources will provide a comprehensive understanding of the underlying systems and help in correctly handling your custom file types.

In closing, remember that the correct uti choice for AirDrop isn't just about making your app work; it's about creating a seamless and predictable user experience. By making informed decisions about your uti, you’re building a more stable, reliable, and user-friendly product. My experience with the collaborative drawing app emphasized how much a deep understanding of this system matters for the smooth functioning of even seemingly simple features like file sharing.
