---
title: "How to implement Android Face ID authentication in React-Native?"
date: "2024-12-16"
id: "how-to-implement-android-face-id-authentication-in-react-native"
---

Alright, let’s talk about implementing face id authentication in a react-native application on android. It's a nuanced process, certainly not a 'one-size-fits-all' solution, and one I've tackled a few times over the years. My experience dates back to when the android biometric api was initially introduced, which made things, shall we say, *interesting* from a developer’s standpoint. Let's break it down, focusing on the android side because that’s where the heavier lifting happens in this context.

The core of implementing face id (or more accurately, biometric authentication, as android doesn’t strictly delineate facial recognition from fingerprint or other methods) in react-native involves bridging the javascript realm with the native android biometric prompt api. You won't find a single straightforward react-native library to accomplish this without native code interaction, and that's where things get a bit more detailed.

First, we need to understand the android biometric library. It operates on the concept of a `biometricprompt`, which handles the user interaction with the biometric hardware. This prompt displays a system dialog for authenticating, handling the biometric acquisition process safely and securely. It’s not something you implement yourself, rather, it's a structured api designed to prevent malicious interference and expose biometric information. Our job in react-native is to initiate this prompt and then appropriately handle its results within the react-native javascript environment.

The starting point is creating a native module in your android project. Within this module, we’ll use the `biometricprompt` api which is part of the `androidx.biometric` package. Make sure your `build.gradle` (module-level) has the dependency:
`implementation "androidx.biometric:biometric:1.2.0"` or a later version, to ensure you’re up-to-date.

Here’s the general flow of the native module code using kotlin – a strongly preferred language for android development these days:

```kotlin
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.*
import java.util.concurrent.Executor

class BiometricModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "BiometricModule"

    private val executor: Executor = ContextCompat.getMainExecutor(reactContext)
    private var promise: Promise? = null

    @ReactMethod
    fun authenticate(promptTitle: String, promise: Promise) {
        this.promise = promise

        val biometricManager = BiometricManager.from(reactContext)
        when (biometricManager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG)) {
            BiometricManager.BIOMETRIC_SUCCESS -> {
                showBiometricPrompt(promptTitle)
            }
            BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE, BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE -> {
              promise.resolve("biometric_not_available")
            }
            BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED -> {
                promise.resolve("biometric_not_enrolled")
            }
            else -> {
               promise.resolve("unknown_error")
            }
        }

    }

    private fun showBiometricPrompt(promptTitle: String) {
        val biometricPrompt = BiometricPrompt(
            currentActivity!!,
            executor,
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    super.onAuthenticationError(errorCode, errString)
                     promise?.resolve("authentication_error_$errorCode")
                    promise = null

                }

                override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                    super.onAuthenticationSucceeded(result)
                     promise?.resolve("authentication_success")
                    promise = null

                }

                override fun onAuthenticationFailed() {
                    super.onAuthenticationFailed()
                     promise?.resolve("authentication_failed")
                     promise = null

                }
            }
        )

        val promptInfo = BiometricPrompt.PromptInfo.Builder()
            .setTitle(promptTitle)
            .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_STRONG)
            .setNegativeButtonText("Cancel")
            .build()

        biometricPrompt.authenticate(promptInfo)
    }


}
```

Let me unpack this a bit. `BiometricModule` is our react-native bridge. The `authenticate` method is exposed as a react-native function call. Inside, we first check if biometric authentication is even available on the device using `BiometricManager.canAuthenticate()`. If available, we display the biometric prompt. The `biometricprompt` callback handles the success, failure, and error cases, resolving the react-native promise with appropriate string indicators. If the biometric hardware isn't available or if biometric authentication isn't setup, it resolves the promise with appropriate messages. This way, we can handle all cases in the javascript side. I've seen so many implementations neglect to handle cases where biometrics aren’t available or configured, leading to very confusing user experiences.

Next, we'll create the bridge file for the native module. This goes in the `android/app/src/main/java/<your_package>/` directory. We then need to register this module in our `MainApplication.java` file, similar to any custom native module.

Now, let’s see how this would be used in javascript within a react-native app.

```javascript
import { NativeModules } from 'react-native';

const { BiometricModule } = NativeModules;

const authenticateWithBiometrics = async (title) => {
    try {
        const result = await BiometricModule.authenticate(title);
        switch (result) {
            case "authentication_success":
                console.log('Authentication success');
                return true;
            case "authentication_failed":
                console.log("Authentication failed");
                return false;
            case "biometric_not_available":
                console.log("Biometric hardware not available");
                return false;
            case "biometric_not_enrolled":
                console.log("Biometrics not enrolled");
                return false;
            default:
                if (result.startsWith("authentication_error")) {
                    const errorCode = result.split("_")[2];
                    console.log(`Authentication Error code ${errorCode}`)
                     return false;
                 }
                console.log("unknown error");
                return false;
        }

    } catch (error) {
        console.error("Error authenticating:", error);
        return false;
    }
};


//Example usage
async function handleBiometricAuth(){
    const isAuthenticated = await authenticateWithBiometrics("Authenticate with Face ID");
    if(isAuthenticated){
        //do something after successful authentication
    } else {
       //handle authentication error

    }
}
```

Here, we import the native module, then call the `authenticate` method we defined in our java module. The method then returns a promise that resolves based on the callback from the native side. We check for various cases and return the corresponding status to the caller function. This is a very simplified version. In reality, you might want more granular error handling or a more user-friendly flow. Also, note that we do not have a way to specify if a PIN/Password can be used as a fallback. This capability is only exposed in the api as of API 30.

Lastly, let's illustrate how you can integrate this into a React component to actually trigger it.

```jsx
import React from 'react';
import { Button, View, Text } from 'react-native';
import { authenticateWithBiometrics } from './biometric';

const AuthScreen = () => {
    const handleBiometricAuthentication = async () => {
      const authSuccess = await authenticateWithBiometrics('Please authenticate to access content');

      if (authSuccess) {
        // Navigate to main app screen or perform authenticated action.
        console.log('Authentication Successful, navigating')
      } else {
        // Handle authentication failure (e.g., display error message).
         console.log('Authentication Failed, retry')
      }
    };

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
             <Text style={{ marginBottom: 20, fontSize: 18 }}>
              Click the button below to authenticate with Biometrics
            </Text>
            <Button title="Authenticate" onPress={handleBiometricAuthentication} />
        </View>
    );
};

export default AuthScreen;

```

This very straightforward example demonstrates a basic component that houses a button. When pressed, it triggers the `handleBiometricAuthentication` function which invokes the biometric authentication flow. This approach ensures a smooth integration of the native functionality into the user interface.

For further study on this topic, I’d recommend starting with the official android documentation on the `androidx.biometric` library. It’s well-written and provides the most accurate details. Additionally, reading *Android Programming: The Big Nerd Ranch Guide* can be beneficial in understanding more of the fundamentals of native android development and architecture. For react-native, familiarize yourself with the documentation on creating native modules, it’s foundational knowledge for this type of task. Understanding android’s security model, particularly around data handling for biometrics, is crucial, and resources like the android security documentation can help clarify those aspects. Avoid third party libraries with complex underlying implementations, if possible. It's much better to have full control, security and transparency on such a sensitive process.

Implementation details can shift based on the precise versions of the `androidx` library and your particular target sdk version. Always ensure your dependencies are current and your code aligns with the latest apis. This example provides a baseline and is not meant to be production ready without appropriate testing, security considerations and robust handling of edge cases. It took a couple of tries on my first project, but these types of experiences are the best learning opportunities, in my opinion.
