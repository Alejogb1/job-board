---
title: "How do I implement Android Face ID authentication in React-Native?"
date: "2024-12-23"
id: "how-do-i-implement-android-face-id-authentication-in-react-native"
---

,  I've navigated the intricacies of biometrics in mobile apps quite a few times now, and implementing face id authentication in react-native presents its own set of interesting challenges. It's not a straightforward plug-and-play affair, unfortunately. We'll need to bridge the gap between javascript and the native android layer, using a combination of native modules and some clever react-native components.

When I first dealt with this, back in my fintech days, we had a very specific requirement for high-security login. The client insisted on biometric authentication as the primary method, so understanding the underlying mechanisms became crucial. We discovered that relying solely on generic third-party packages wasn't sufficient; custom solutions were required for reliable error handling and seamless user experience. Let’s go through the steps involved and what I've found most useful based on experience.

The crux of implementing face id on android, or, more accurately, biometric authentication encompassing face unlock, fingerprint, and other similar methods, involves utilizing the android biometric library. This library handles the device-specific hardware interactions and security aspects. In react-native, we can't directly access it from javascript, so we need a bridge—a native module—to expose the necessary functionality.

Here’s a rough breakdown of what the native module needs to do:

1.  **Check for biometric support:** before attempting any authentication, verify if the device actually supports biometric authentication.
2.  **Initiate biometric prompt:** display the system-level authentication dialog, which would be the fingerprint scanner, face unlock, or other forms, depending on user device settings.
3.  **Handle authentication results:** upon successful or failed authentication, provide feedback back to react-native’s javascript layer.
4.  **Manage key creation and storage:** secure data storage using android's keystore system if needed, though for basic authentication purposes, it might not be required.

Now, let's get to some code. Here's a hypothetical java implementation of a native module (simplified for brevity):

```java
package com.yourapp;

import android.app.Activity;
import android.content.Context;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.biometric.BiometricManager;
import androidx.biometric.BiometricPrompt;
import androidx.core.content.ContextCompat;
import com.facebook.react.bridge.*;

import java.util.concurrent.Executor;


public class BiometricModule extends ReactContextBaseJavaModule {

    private Executor executor;
    private BiometricPrompt biometricPrompt;
    private Promise authPromise;

    BiometricModule(ReactApplicationContext context) {
        super(context);
        executor = ContextCompat.getMainExecutor(context);
    }

    @NonNull
    @Override
    public String getName() {
        return "BiometricModule";
    }


    @ReactMethod
    public void canAuthenticate(Promise promise) {
        BiometricManager biometricManager = BiometricManager.from(getReactApplicationContext());
        int canAuthenticate = biometricManager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG | BiometricManager.Authenticators.DEVICE_CREDENTIAL);
        promise.resolve(canAuthenticate == BiometricManager.BIOMETRIC_SUCCESS);

    }

    @ReactMethod
     public void authenticate(String title, String subtitle, String description, Promise promise) {
         Activity currentActivity = getCurrentActivity();
        if (currentActivity == null) {
            promise.reject("NoActivity", "Activity is null");
            return;
        }

        authPromise = promise;


        BiometricPrompt.PromptInfo promptInfo = new BiometricPrompt.PromptInfo.Builder()
                .setTitle(title)
                .setSubtitle(subtitle)
                .setDescription(description)
                .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_STRONG | BiometricManager.Authenticators.DEVICE_CREDENTIAL)
                .build();


        biometricPrompt = new BiometricPrompt(currentActivity, executor, new BiometricPrompt.AuthenticationCallback() {
            @Override
            public void onAuthenticationError(int errorCode, @NonNull CharSequence errString) {
                super.onAuthenticationError(errorCode, errString);
                if(authPromise != null){
                    authPromise.reject(String.valueOf(errorCode), errString.toString());
                }

                authPromise = null;

            }

            @Override
            public void onAuthenticationSucceeded(@NonNull BiometricPrompt.AuthenticationResult result) {
                super.onAuthenticationSucceeded(result);
                if(authPromise != null){
                 authPromise.resolve(true);
                }
                authPromise = null;
            }

            @Override
            public void onAuthenticationFailed() {
                super.onAuthenticationFailed();
                 if(authPromise != null){
                   authPromise.reject("AuthenticationFailed", "Authentication failed");
                }
                authPromise = null;

            }
        });
         biometricPrompt.authenticate(promptInfo);

    }
}
```

Next, you would need to bridge this to javascript. We create a typescript file that interacts with our newly defined native module:

```typescript
// BiometricModule.ts
import { NativeModules, Platform } from 'react-native';

const { BiometricModule } = NativeModules;

interface Biometric {
  canAuthenticate(): Promise<boolean>;
  authenticate(title: string, subtitle: string, description: string): Promise<boolean>;
}


const biometric: Biometric = {
  canAuthenticate: async (): Promise<boolean> => {
      if (Platform.OS !== 'android') {
          return Promise.resolve(false);
      }

      return BiometricModule.canAuthenticate();
    },

  authenticate: async (title: string, subtitle: string, description:string): Promise<boolean> => {
      if (Platform.OS !== 'android') {
          return Promise.reject("not supported");
      }
    return BiometricModule.authenticate(title, subtitle, description)
    }

};

export default biometric;
```

Finally, here's how you might use it in a react-native component:

```tsx
// MyComponent.tsx
import React, { useState, useEffect } from 'react';
import { View, Text, Button, Alert } from 'react-native';
import biometric from './BiometricModule';

const MyComponent: React.FC = () => {
  const [biometricAvailable, setBiometricAvailable] = useState(false);

  useEffect(() => {
    const checkBiometricAvailability = async () => {
      const available = await biometric.canAuthenticate();
      setBiometricAvailable(available);
    };

    checkBiometricAvailability();
  }, []);


  const handleBiometricAuth = async () => {
      try{
          const success = await biometric.authenticate(
              'Authenticate',
              'Confirm your identity',
              'Please authenticate to continue',
          );
          if(success){
              Alert.alert('Success','User Authenticated!');
          }
      }
       catch(e){
          if(e instanceof Error){
              Alert.alert('Error', `Authentication Failed: ${e.message}`);
          }
           else{
               Alert.alert('Error', `Authentication Failed`);
           }

        }
  };


  return (
    <View>
      {biometricAvailable ? (
        <Button title="Authenticate with Biometrics" onPress={handleBiometricAuth} />
      ) : (
        <Text>Biometrics not available</Text>
      )}
    </View>
  );
};

export default MyComponent;
```

**Important Caveats:**

1.  **Error Handling:** The sample java code has basic error handling, but you need to handle scenarios like authentication failures, biometric hardware errors, and user cancellation gracefully. The callback implementation shows a basic idea of this.
2.  **Permission:** Ensure you’ve included the appropriate permissions in your `AndroidManifest.xml` file, primarily `USE_BIOMETRIC`.
3.  **Security:** Consider the security implications. if you’re storing sensitive data based on this authentication, use android's keystore system for encrypting the data with a key that's protected using biometrics. Look into `KeyStore` API and how to pair that with the `BiometricPrompt` API. This adds an extra layer of complexity but is vital for secure applications.
4.  **Testing:** Biometric authentication is heavily reliant on hardware, and testing on emulators can be inconsistent. Always test on real android devices across various manufacturers to ensure the user experience remains consistent.
5.  **Backward compatibility:** Make sure you check for device api level and use the correct APIs based on the level. The androidx biometric library helps with most, but a solid strategy to handle older device gracefully is important.

**Recommendations for Further Study**

To deepen your knowledge further, i'd strongly advise consulting:

*   **Android Developer Documentation:** Pay close attention to the official documentation on the androidx biometric library, the keystore system, and security best practices. The android developer website is comprehensive, and there's a lot of useful information there.
*   **"Android Security Internals" by Nikolay Elenkov:** A fantastic book for really understanding the security aspects and intricacies of the android operating system, from the kernel level up to the application frameworks.
*   **"Pro Android Security" by David Chandler et al.:** This text is quite a comprehensive guide, covering topics from general application security, threat modeling, to detailed implementations, particularly useful if you're working on applications that deal with sensitive data.
*   **OWASP Mobile Security Project (MASVS):** The Open Web Application Security Project (OWASP) has great resources and guidelines for mobile app security best practices.
*  **Source code:** looking at the source code of various open source apps that use similar technologies could really give you a sense of best practice.

Implementing biometric authentication, particularly face id, on android in react-native is a multi-faceted task. This example gives a foundation to begin, but understand that this is a journey of learning and evolving with the platforms, and being well read is a must. You will need to adapt this to the specific needs of your application. Focus on security, error handling, and providing a smooth, user-friendly authentication experience. Good luck with your project.
