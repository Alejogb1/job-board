---
title: "How can I lock screen orientation in a React Native iOS app?"
date: "2024-12-23"
id: "how-can-i-lock-screen-orientation-in-a-react-native-ios-app"
---

Okay, let's delve into screen orientation locking in React Native for iOS. It's a fairly common requirement, but achieving it seamlessly across all devices and use cases can sometimes feel… well, like it requires a little finesse. I remember wrestling with this a few years back on a client project – a real estate app, as it happened – where we needed the main map view to always be in landscape, while the other sections had to be locked to portrait. We ended up opting for a flexible approach that allowed us to control this on a per-screen basis.

The core concept lies in interacting with the native iOS functionality through React Native bridge. Simply put, we can't directly manipulate the `UIDevice` orientation from within Javascript land. Instead, we'll use native modules to do the heavy lifting. So, where do we begin?

The crucial point here is understanding that we don't want a blanket, application-wide, lock. That's almost always problematic. Ideally, you'd need a granular control, allowing specific screens or components to dictate their orientation. The most straightforward way to do this is by implementing a native module that exposes methods to lock or unlock the screen orientation dynamically, responding to javascript calls within your React Native app.

Let's first examine the native code. We'll create an objective-c module. This module will essentially provide us a bridge to interact with the iOS system's `UIDevice` and `UIApplication` related settings:

**Objective-C Module (OrientationModule.m):**

```objectivec
#import <React/RCTBridgeModule.h>
#import <UIKit/UIKit.h>

@interface OrientationModule : NSObject <RCTBridgeModule>
@end

@implementation OrientationModule

RCT_EXPORT_MODULE();

RCT_EXPORT_METHOD(lockToPortrait)
{
  dispatch_async(dispatch_get_main_queue(), ^{
     if (@available(iOS 16.0, *)) {
        UIWindowScene *currentScene = (UIWindowScene *)[UIApplication sharedApplication].connectedScenes.anyObject;
        [currentScene.requestGeometryUpdateHandler setAllowedOrientations:UIInterfaceOrientationMaskPortrait];

     }else{
         [[UIDevice currentDevice] setValue: [NSNumber numberWithInteger: UIInterfaceOrientationPortrait] forKey:@"orientation"];
         [UIViewController attemptRotationToDeviceOrientation];

     }

  });
}

RCT_EXPORT_METHOD(lockToLandscape)
{
  dispatch_async(dispatch_get_main_queue(), ^{
      if (@available(iOS 16.0, *)) {
          UIWindowScene *currentScene = (UIWindowScene *)[UIApplication sharedApplication].connectedScenes.anyObject;
           [currentScene.requestGeometryUpdateHandler setAllowedOrientations:UIInterfaceOrientationMaskLandscape];

       }else {
          [[UIDevice currentDevice] setValue: [NSNumber numberWithInteger: UIInterfaceOrientationLandscapeRight] forKey:@"orientation"];
          [UIViewController attemptRotationToDeviceOrientation];
      }

  });
}


RCT_EXPORT_METHOD(unlockOrientation)
{
  dispatch_async(dispatch_get_main_queue(), ^{
    if (@available(iOS 16.0, *)) {
         UIWindowScene *currentScene = (UIWindowScene *)[UIApplication sharedApplication].connectedScenes.anyObject;
         [currentScene.requestGeometryUpdateHandler setAllowedOrientations:UIInterfaceOrientationMaskAll];
       }
      else {
         [[UIDevice currentDevice] setValue: [NSNumber numberWithInteger: UIInterfaceOrientationUnknown] forKey:@"orientation"];
         [UIViewController attemptRotationToDeviceOrientation];
       }


  });
}

@end
```

Important points in this code snippet:

1.  **`RCTBridgeModule` Protocol:** This protocol is crucial for exposing native functionality to the React Native environment.
2.  **`RCT_EXPORT_MODULE()`:**  This macro registers the class as a native module.
3.  **`RCT_EXPORT_METHOD(...)`:** These macros expose the `lockToPortrait`, `lockToLandscape`, and `unlockOrientation` functions, making them callable from JavaScript.
4.  **`dispatch_async(dispatch_get_main_queue(), ...)`:** Native iOS methods modifying the UI MUST be executed on the main thread. This ensures that no UI elements are unexpectedly updated outside the main UI loop.
5. **iOS 16 specific handling:** The code now includes handling for devices running iOS 16 and later which uses the requestGeometryUpdateHandler on UIWindowScene to modify allowed orientations.

**Linking the Native Module to React Native:**

Once you have created the `OrientationModule.m` (and the necessary header file `OrientationModule.h`), you'll need to ensure React Native can find and use it. Usually, this involves adding the file to your Xcode project (in the Libraries folder) and updating the `AppDelegate.mm` file with a necessary entry point. Assuming this is handled, you can jump over to how we use this in our Javascript code.

**React Native Implementation:**

Here is a simple way we can use this inside our react native project, we will use a custom hook to make this functionality available in our components:

```javascript
// useOrientation.js
import { NativeModules, Platform } from 'react-native';
const { OrientationModule } = NativeModules;

const useOrientation = () => {
  const lockToPortrait = () => {
    if (Platform.OS === 'ios') {
      OrientationModule.lockToPortrait();
    }
  };

  const lockToLandscape = () => {
    if (Platform.OS === 'ios') {
      OrientationModule.lockToLandscape();
    }
  };

  const unlockOrientation = () => {
    if (Platform.OS === 'ios') {
      OrientationModule.unlockOrientation();
    }
  };
    
  return { lockToPortrait, lockToLandscape, unlockOrientation };
};

export default useOrientation;

```

This `useOrientation` hook abstracts away the complexities of interacting directly with the native module. In any of your React Native components, you can use this hook and lock/unlock the orientation based on the specific screen needs:

```javascript
// ExampleComponent.js
import React, { useEffect } from 'react';
import { View, Text, Button } from 'react-native';
import useOrientation from './useOrientation';

const ExampleComponent = () => {
  const { lockToLandscape, unlockOrientation } = useOrientation();

  useEffect(() => {
    lockToLandscape();

    return () => {
        unlockOrientation(); //cleanup on unmount
    };
  }, [lockToLandscape, unlockOrientation]); //dependancy array, should never change

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>This Screen is Locked in Landscape</Text>
    </View>
  );
};

export default ExampleComponent;

```

In this `ExampleComponent`, we've used the `useEffect` hook to lock the orientation to landscape when the component mounts and unlock it when the component is unmounted. This ensures proper behaviour during navigation transitions, it is critical to clean up on component unmount to avoid unexpected behaviour.

**Important Considerations**

*   **iOS Version:** As you saw in the objective-c code, we need to adjust how we lock the orientation depending on the iOS version (iOS 16+ or below). This is because the way allowed orientations are specified changed in iOS 16, and we must use `requestGeometryUpdateHandler` and `UIWindowScene` to achieve the same result.
*   **User Experience:** Never force the user into a particular orientation unless it is absolutely necessary for functionality. Consider implementing orientation changes with smooth transitions rather than sudden shifts.
*   **Android:** While this response focuses on iOS, the approach is conceptually similar on Android using native modules, but you would have to look at `Activity` and its related methods.
*   **Testing:** Always test your orientation locking on real devices and different iOS versions since the simulator sometimes doesn’t accurately reflect device behavior, especially when rotation is involved.

**Recommendations for Further Reading**

For a deeper understanding, I'd highly recommend exploring the following resources:

1.  **Apple's official documentation on `UIDevice` and `UIWindowScene`**: This will give you the official perspective and the inner workings of how screen orientation is handled on iOS.
2.  **"Pro iOS Apps with React Native" by Jonathan Magnan:** This book has a complete detailed section on native module creation with a great understanding of the bridge.
3.  **React Native Documentation:** You need to know the basics of Native Modules in the documentation if you are going to create your own.

By following the steps above and consulting the suggested materials, you should be able to implement fine-grained screen orientation control in your React Native iOS application. Remember, flexibility and user experience should be at the forefront of your development approach.
