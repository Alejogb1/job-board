---
title: "How can a native program be run within Adobe AIR?"
date: "2025-01-30"
id: "how-can-a-native-program-be-run-within"
---
Adobe AIR, while offering cross-platform application development through ActionScript and JavaScript, inherently operates within a sandbox environment. Consequently, direct execution of native code presents a challenge. Overcoming this limitation necessitates the use of AIR Native Extensions (ANEs), custom extensions written in native languages (like C++, Java, or Objective-C) that provide a bridge between the ActionScript/JavaScript virtual machine and the underlying operating system's functionalities. These extensions are essentially shared libraries or dynamic link libraries (.dll, .so, .dylib) wrapped in a specific structure that AIR can understand and integrate.

Developing an ANE involves several distinct phases, beginning with the native code implementation, proceeding to packaging that code with a descriptor file and then finally, consuming that package within an AIR application. The primary problem to solve lies in managing the communication flow between the sandboxed AIR environment and the unsandboxed native environment. I have, through multiple projects, encountered the common pitfalls and best practices involved with such an endeavor.

The core principle revolves around establishing clear communication channels. Native methods, defined within the native extension, are exposed as callable functions from ActionScript or JavaScript. When a function in the AIR application invokes a native function, the AIR runtime marshals the parameters, sends them to the native side and, after execution of the native function, receives the return value, marshaling that back to the ActionScript side. This marshaling process is crucial for type safety and ensures that data passed between different environments are interpreted correctly. Data types typically need to be mapped between equivalent representations available in both ActionScript/JavaScript and the specific native environment. For instance, a number in ActionScript might need to be converted to an integer in C++. I have personally observed instances where incorrectly mapping data types led to memory corruption and unpredictable application crashes.

Consider a scenario where we need an AIR application to fetch the system's CPU architecture (x86, x64, ARM). We can't directly get this information via AIR’s APIs. This requires a native extension.

**Example 1: C++ Native Extension (Windows)**

This example illustrates a basic C++ implementation for a Windows platform. This section would be part of the platform-specific implementation in a full ANE project.

```cpp
#include <windows.h>
#include <string>
#include <sstream>
#include "ANECommon.h" // Assumed header for AIR common type definitions

FREObject getCPUArchitecture(FREContext ctx, void* funcData, uint32_t argc, FREObject argv[]) {
    SYSTEM_INFO sysinfo;
    GetNativeSystemInfo(&sysinfo);
    std::string archStr;

    switch (sysinfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            archStr = "x64";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            archStr = "x86";
            break;
        case PROCESSOR_ARCHITECTURE_ARM:
            archStr = "ARM";
            break;
        case PROCESSOR_ARCHITECTURE_ARM64:
            archStr = "ARM64";
            break;
        default:
            archStr = "Unknown";
            break;
    }

   // Convert the string to FREObject for return to AIR
    FREObject result;
    const uint8_t* strData = reinterpret_cast<const uint8_t*>(archStr.c_str());
    uint32_t strLen = static_cast<uint32_t>(archStr.length());
    if(FRENewObjectFromUTF8(strLen + 1, strData, &result) != FRE_OK){
        return nullptr;
    }
    return result;

}

// Extension initialization and deinitialization functions would go here
```

In this C++ code, I retrieve the system's architecture using Windows API calls and construct a string representation. Critically, this string must be converted to a `FREObject`, a fundamental data type for communicating with the AIR runtime. Failing to correctly convert the data results in data not being retrieved correctly on the ActionScript side. The `ANECommon.h` header (not shown for brevity) includes necessary definitions for `FREObject` and related AIR runtime structures. Note the inclusion of a case for each potential processor architecture, crucial for accurate reporting.

**Example 2: ActionScript/JavaScript Usage**

The ActionScript or JavaScript within the AIR application then uses this extension to query the processor architecture. I will show example ActionScript.

```actionscript
package
{
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.events.Event;
	import flash.utils.ByteArray;
    import com.example.MyNativeExtension; // Assuming this is your ANE class

	public class Main extends Sprite
	{
		private var outputText:TextField;
		private var _extension:MyNativeExtension;

		public function Main():void
		{
			if (stage) init();
			else addEventListener(Event.ADDED_TO_STAGE, init);
		}

		private function init(e:Event = null):void
		{
			removeEventListener(Event.ADDED_TO_STAGE, init);
			// entry point
			outputText = new TextField();
			addChild(outputText);
			outputText.x = 20;
			outputText.y = 20;

		    try{
			 _extension = new MyNativeExtension();
              var cpuArch : String = _extension.getCPUArchitecture();
			    outputText.text = "CPU Architecture: " + cpuArch;
            }catch(e:Error){
                outputText.text = "Error Loading Native Extension" + e;
            }
		}
	}
}
```

In the ActionScript snippet, the key part is the initialization of `MyNativeExtension` class (a representation of the native extension) and invocation of its `getCPUArchitecture()` function. The returned string, representing the architecture, is displayed in the text field. It is critical to catch exceptions in the instantiation or usage of the extension, to gracefully handle scenarios where loading the ANE fails.

**Example 3: ANE Descriptor XML**

The ANE also includes an XML file (extension.xml) that describes the extension and declares native libraries to load and methods the extension exposes.

```xml
<extension xmlns="http://ns.adobe.com/air/extension/1.0">
  <id>com.example.MyNativeExtension</id>
  <versionNumber>1</versionNumber>
  <platforms>
    <platform name="Windows-x86">
      <applicationDeployment>
        <nativeLibrary>MyNativeExtension.dll</nativeLibrary>
          </applicationDeployment>
       <runtimeVersions>
          <version>2.0</version>
      </runtimeVersions>
    </platform>
	    <platform name="Windows-x86-64">
      <applicationDeployment>
        <nativeLibrary>MyNativeExtension64.dll</nativeLibrary>
          </applicationDeployment>
       <runtimeVersions>
          <version>2.0</version>
      </runtimeVersions>
    </platform>
        <platform name="Android-ARM">
            <applicationDeployment>
                <nativeLibrary>libMyNativeExtension.so</nativeLibrary>
            </applicationDeployment>
        <runtimeVersions>
            <version>2.0</version>
        </runtimeVersions>
        </platform>
          <platform name="Android-ARM64">
            <applicationDeployment>
                <nativeLibrary>libMyNativeExtension64.so</nativeLibrary>
            </applicationDeployment>
            <runtimeVersions>
                <version>2.0</version>
            </runtimeVersions>
        </platform>
         <platform name="iOS-ARM">
            <applicationDeployment>
                <nativeLibrary>libMyNativeExtension.a</nativeLibrary>
            </applicationDeployment>
            <runtimeVersions>
                <version>2.0</version>
            </runtimeVersions>
        </platform>
         <platform name="iOS-ARM64">
            <applicationDeployment>
                <nativeLibrary>libMyNativeExtension64.a</nativeLibrary>
            </applicationDeployment>
            <runtimeVersions>
                <version>2.0</version>
            </runtimeVersions>
        </platform>
  </platforms>
</extension>
```

This descriptor provides the AIR runtime with the necessary information to identify and integrate the native extension. Note how different `platform` entries specify different native libraries (.dll, .so, .a) for various operating systems and architectures. Including the runtime version is considered a best practice to guarantee compatibility between the extension and the AIR runtime. This aspect has been a point of failure in several situations during development where the extension was incompatible with runtime version.

The process of creating a functional ANE further involves packaging these components into a single .ane file, which is essentially a zip archive containing all necessary components. The packaging typically utilizes the ADT (Adobe AIR Developer Tool) with appropriate parameters defining the included files and the location of the XML descriptor file. I encountered difficulties numerous times because the structure of the .ane file was incorrect which prevented proper loading.

For resources, I'd recommend consulting the Adobe AIR documentation on creating native extensions. The official Adobe documentation provides both theoretical and step-by-step instructions. Also the community forums related to ANEs are an additional valuable resource, with detailed examples and answers to common issues. Finally, reviewing open-source ANE projects on platforms like Github can provide practical examples and implementation strategies, as it often reveals patterns that might not be immediately obvious in documentation. Building ANEs is a complex process with many nuances. Careful attention to the details of data marshaling, platform-specific build processes, and the XML descriptor are necessary for reliable extension creation.
