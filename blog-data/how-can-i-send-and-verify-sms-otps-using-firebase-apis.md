---
title: "How can I send and verify SMS OTPs using Firebase APIs?"
date: "2024-12-23"
id: "how-can-i-send-and-verify-sms-otps-using-firebase-apis"
---

Alright, let’s get into the specifics of handling SMS one-time passwords (OTPs) with Firebase. This is something I’ve dealt with extensively over the years, and it’s evolved quite a bit. I remember back in my early days, we had to roll our own solutions, managing SMS gateways and dealing with all the complexities that came with it. Thankfully, Firebase simplifies a lot of that now. The key lies in Firebase Authentication's phone number sign-in feature, which provides built-in OTP mechanisms. Let's break it down step-by-step, focusing on both the sending and verification aspects, and avoiding any of the fluff.

First, the core idea revolves around the `signInWithPhoneNumber` method, typically on the client side of your application. It starts the process by requesting an OTP to be sent to the user’s provided phone number. Then, you receive a verification id which you'll use later. The Firebase backend handles the SMS sending, so you don't need to worry about the intricacies of carrier networks. This is a significant improvement over older approaches. After receiving the OTP, the user inputs it, and you use that to complete the verification and authenticate the user.

Now, let's look at this practically. Here's a simplified JavaScript example, assuming you're working in a web application environment. I’ll provide analogous snippets in other common environments later:

```javascript
import { getAuth, RecaptchaVerifier, signInWithPhoneNumber } from "firebase/auth";

const auth = getAuth();

function sendVerificationCode(phoneNumber) {
  window.recaptchaVerifier = new RecaptchaVerifier('sign-in-button', {
        'size': 'invisible'
      }, auth);

  const appVerifier = window.recaptchaVerifier;

  signInWithPhoneNumber(auth, phoneNumber, appVerifier)
  .then((confirmationResult) => {
    window.confirmationResult = confirmationResult;
    // Store the confirmation result, often in your state management system
    console.log("SMS sent successfully with verification id:", confirmationResult.verificationId);
    // Proceed to the UI that prompts for the OTP input
  })
  .catch((error) => {
    console.error("Error sending SMS:", error);
    // Handle the error appropriately, perhaps informing the user
  });
}

// example usage:
// sendVerificationCode("+15551234567")
```

In this snippet, you'll see the use of a `RecaptchaVerifier`. Firebase uses this to prevent bots from abusing the service. The `signInWithPhoneNumber` function takes the authentication instance, the user's phone number, and the app verifier. Upon success, a `confirmationResult` is returned, which contains the `verificationId`. This verification id is critical for the next stage. Crucially, store this value in your application's state because you'll need it to complete the authentication process later. Note, I've added comments on error handling - this is an essential aspect often overlooked.

Moving on, let's address the verification step. After the user enters the OTP sent via SMS, this code executes:

```javascript
import { getAuth, signInWithCredential, PhoneAuthProvider } from "firebase/auth";


function verifyCodeAndSignIn(otpCode, confirmationResult) {
  const auth = getAuth();
  const credential = PhoneAuthProvider.credential(confirmationResult.verificationId, otpCode);

  signInWithCredential(auth, credential)
    .then((userCredential) => {
        const user = userCredential.user;
        console.log("User signed in:", user);
        // Redirect the user or update application state as necessary
    })
    .catch((error) => {
        console.error("Error verifying OTP:", error);
        // Handle the error appropriately, inform the user, and allow to re-enter if needed
    });
}

// example usage
//verifyCodeAndSignIn("123456", window.confirmationResult)
```

This snippet takes the OTP code entered by the user and the previously stored `confirmationResult`, uses the `PhoneAuthProvider.credential` to construct a credential using verificationId and code, then attempts to sign in the user using this credential with `signInWithCredential` method. If the OTP matches the expected verification code on the backend, the promise resolves and provides you with an instance of `userCredential`. Otherwise, it will return an error with the reason.

Now, let’s look at this from an Android point of view. Here’s an example using Kotlin:

```kotlin
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.PhoneAuthOptions
import com.google.firebase.auth.PhoneAuthProvider
import java.util.concurrent.TimeUnit

fun sendVerificationCodeAndroid(phoneNumber: String, activity: Activity) {
    val auth = FirebaseAuth.getInstance()

    val options = PhoneAuthOptions.newBuilder(auth)
        .setPhoneNumber(phoneNumber) // Phone number to verify
        .setTimeout(60L, TimeUnit.SECONDS) // Timeout and unit
        .setActivity(activity) // Activity (for callback binding)
        .setCallbacks(object : PhoneAuthProvider.OnVerificationStateChangedCallbacks() {

        override fun onVerificationCompleted(credential: PhoneAuthCredential) {
                // This callback will be invoked in case of instant verification
                // This happens when the phone number being verified is the
                // one which is signed into current device (or sometimes through
                // google play services)
                // Automatically sign in the user using credential here, but in
                // this example we just want to verify OTP so skip it.
        }


        override fun onVerificationFailed(firebaseException: FirebaseException) {
            // Handle verification failure
           Log.e("OTP Verification Error", firebaseException.message.toString())
        }

        override fun onCodeSent(verificationId: String, token: PhoneAuthProvider.ForceResendingToken) {
            // Store the verificationId and also allow re-sending code here
            Log.d("OTP Sent", "OTP send for verificationId: $verificationId")
            // Prompt user to enter OTP
            // Pass the verificationId when user enters OTP
        }
    }).build()

    PhoneAuthProvider.verifyPhoneNumber(options)
}

fun verifyCodeAndSignInAndroid(otpCode: String, verificationId: String, context: Context) {
  val auth = FirebaseAuth.getInstance()
  val credential = PhoneAuthProvider.getCredential(verificationId, otpCode)

  auth.signInWithCredential(credential)
      .addOnCompleteListener { task ->
          if (task.isSuccessful) {
              val user = auth.currentUser
              // Handle successful sign in
              Log.d("OTP Verified", "User sign in successful : ${user?.uid}")
          } else {
              // Handle error
              Log.e("OTP Verification Error", task.exception.toString())
              Toast.makeText(context, "Error verifying OTP: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
          }
      }
}

//example usage
//sendVerificationCodeAndroid("+15551234567", this)
// verifyCodeAndSignInAndroid("123456", "storedVerificationId", applicationContext)
```

This example demonstrates how to accomplish the same outcome on Android, using Kotlin. Notice the use of `PhoneAuthProvider.OnVerificationStateChangedCallbacks` which handles various lifecycle events including on-completion and code sending. The approach for signing in using OTP is almost similar where we form the credential object with verification id and code.

Now, a few things to keep in mind. First, you need to configure Firebase correctly in your project. Make sure you've enabled phone number sign-in in your Firebase console. You'll need to set up your SHA-1 fingerprints (and SHA-256 for Android) for Android and also have the bundle ID for iOS configured in Firebase, and ensure that your recaptcha keys are properly configured for web. Also, pay attention to error handling. Network issues, invalid phone numbers, and incorrect OTPs will all throw errors. It's crucial to handle these gracefully and provide helpful feedback to the user.

From a security perspective, always treat the `verificationId` with care. Do not expose it in the UI or store it in a vulnerable way. Consider using robust state management in your application to handle these variables.

For further exploration, I'd highly recommend the official Firebase documentation, which is comprehensive and kept up-to-date. Also, the following papers on authentication principles might be helpful in understanding the "why" behind these steps: "Security Engineering: A Guide to Building Dependable Distributed Systems" by Ross Anderson, and the "Handbook of Applied Cryptography" by Alfred J. Menezes, Paul C. van Oorschot, and Scott A. Vanstone can shed light on some of the underlying cryptography principles. Understanding these fundamentals will deepen your grasp of how Firebase's mechanisms function.

Finally, remember that the UX matters significantly in this flow. Clear error messages, proper input fields, and timely feedback are crucial. A poorly designed OTP flow can frustrate your users and impact the overall experience. That said, by leveraging the Firebase APIs, you are on the path to a simplified and secure authentication process, but always remember that attention to detail and understanding fundamentals are crucial for success.
