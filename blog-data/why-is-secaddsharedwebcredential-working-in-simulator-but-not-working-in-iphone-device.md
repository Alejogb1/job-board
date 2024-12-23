---
title: "Why is SecAddSharedWebCredential working in Simulator but Not working in iPhone Device?"
date: "2024-12-15"
id: "why-is-secaddsharedwebcredential-working-in-simulator-but-not-working-in-iphone-device"
---

alright, so, you're seeing secaddsharedwebcredential behaving differently between the simulator and an actual iphone device, right? yeah, i've been there. it's a classic head-scratcher when things work perfectly in the simulated world but go sideways when deployed to a real device. let me break down what’s probably happening and what i’ve learned the hard way over the years, trying to get these shared web credentials to behave.

the first thing to understand is that the security framework on ios, while conceptually the same across the simulator and the device, has some practical differences. the simulator environment is far more forgiving. it's essentially a sandbox running on your mac, so it doesn't have all the real-world constraints of a hardware device. your actual iphone, on the other hand, operates under much stricter security rules that are deeply tied to the secure enclave and hardware-level protections.

secaddsharedwebcredential, specifically, is involved in saving web credentials that can be used across multiple apps from the same developer team and associated domains. the whole premise is built around the idea of securely sharing credentials, and that's where some device-specific stuff starts to matter. on a simulator, often, these checks are either bypassed or not as rigorously enforced, but on a real iphone device, apple's security checks are fully active.

let’s talk common culprits:

first, entitlement issues. this is probably the most common pitfall. for secaddsharedwebcredential to work, your application needs to have the correct entitlements configured. this is not just a 'make sure you enabled the keychain sharing' thing; it's very specific to shared web credentials. your app's entitlements file (usually `your_app_name.entitlements`) needs to include the `com.apple.developer.associated-domains` entitlement and the correct `webcredentials` domain.

here’s a simple example of how to configure it (this is often under the hood or in xcode’s signing settings but you need to confirm this):

```xml
<!doctype plist public "-//apple//dtd plist 1.0//en" "http://www.apple.com/dtds/propertylist-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>com.apple.developer.associated-domains</key>
        <array>
            <string>webcredentials:yourdomain.com</string>
        </array>
    </dict>
</plist>
```

you'll want to check xcode, under the signing and capabilities tab of your project settings, and make sure associated domains is added, and that your webcredentials configuration is correctly set up there. if you're manually handling entitlements files, you also have to confirm that xcode did not add a new one and instead is using the correct file you configured. this has happened to me and i felt silly after 3 hours.

second, associated domains not properly configured. it is not enough that you have the entitlements, your associated domain needs to be correctly configured on the server side. this is often overlooked as developers assume the client side is the culprit. your server needs to serve a file `apple-app-site-association` at `https://yourdomain.com/.well-known/apple-app-site-association`. this file is a json, and the iphone will look for this specific file and location to check the domains associated with the app. this file has to be served with a mime type of `application/json` or the device will simply ignore it and no errors will be thrown. i lost a good night of sleep debugging this so don’t ignore it.

the contents of the json should be like this:

```json
{
  "applinks": {
      "apps": [],
        "details": [
            {
                "appID": "teamid.bundleid",
                "paths": [
                    "/ *"
                ]
            }
        ]
    },
    "webcredentials": {
        "apps": [
             "teamid.bundleid"
         ]
    }
}
```

replace `teamid` with your development team id and `bundleid` with your application's bundle identifier. this should match precisely the entitlement identifier you configured in your entitlements file. if either of these identifiers is not correctly set the authentication will silently fail and you will be scratching your head wondering what is happening.

now, there is a subtle but important thing to remember, you have to use the full app identifier, so something like this: “teamid.your.bundle.id”. and the key is that this needs to match *exactly* the one you use to sign your app. also, confirm that you have https configured correctly because if you don’t the configuration won’t be considered and the handshake will fail. it is like trying to talk to someone through a broken phone. it just won't work.

third, the actual code for saving the credential. i've had cases where the code looked but subtle bugs were causing the credential saving to fail. while the simulator is forgiving, the device is not. for instance the service parameter on your call has to be fully equal to the configured service type in your credentials configuration.

```swift
func saveWebCredential(username: string, password: string, service: string, completion: @escaping (bool, error?) -> void) {
  let query = [
    ksecclass as string: ksecclassinternetpassword,
    ksecattrserver as string: "yourdomain.com", // this needs to match your domain
    ksecattraccount as string: username,
    ksecattrservice as string: service, // this needs to be the exact same string configured on the server
    ksecvalueData as string: password.data(using: .utf8)!
  ] as [string : any]

  let status = secadd(query as cfdictionary)

   if status == errsecsuccess {
        completion(true, nil)
   } else {
      completion(false, secerror(status: status))
    }
}
```

check that the `ksecattrserver` is the same domain of your `webcredentials`, and that `ksecattrservice` is what you expect. a small typo here will make the save fail. the error codes are not always that clear but you can use `secerror(status: status)` to get an error that gives you slightly more information. and if you get a weird error like `-34018` it usually means that something went wrong with the associated domains or that the credentials are not correctly configured on the keychain.

fourth, keychain access groups. this is less common but still can trip you up if you're sharing keychain items between multiple apps that are not using shared web credentials. make sure that the keychain access groups are configured correctly for your app or you will run into trouble. if you are doing something more advanced you might have to consider using `ksecattraccessgroup` if that is the case the app access group has to match across the apps that want to share the credentials.

lastly, debugging on a real device is a bit different than in the simulator. you need to ensure that the device is connected to your mac, xcode is using the correct provisioning profile, and the app has been properly installed. check the device's console logs using xcode to help you pinpoint more specific error messages. i know that staring at the screen looking for error logs might not be the most exciting part of coding but it is a necessary one.

also, when dealing with security related issues i found it useful to consult apple’s documentation about the keychain services. the documentation can be quite dense but there is no better way to learn. i would recommend reading the "keychain services programming guide", it will save you a lot of time and headaches.

finally, remember to rebuild your app after changes. i know this sounds basic, but sometimes we get lost in the code and we forget the basics. and yeah, sometimes, the issue is that i needed to reboot the device and things would just magically work after. it’s like the device also needed a “have you tried turning it off and on again?”.

in my experience, these are the most common reasons why things work on the simulator but not on a device. take a careful look at each of these areas and you'll likely find the issue. hope this helps, and let me know if you're still having problems.
