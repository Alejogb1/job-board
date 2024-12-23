---
title: "Where are Android keystore keypairs stored?"
date: "2024-12-23"
id: "where-are-android-keystore-keypairs-stored"
---

Let's tackle the specifics of where Android keystore keypairs reside. It’s a topic I've had to navigate thoroughly, particularly back in my mobile security days building custom authentication mechanisms for a fintech app – a project that certainly tested my understanding of the Android security model.

Fundamentally, when we talk about Android keystore, we’re not discussing one single physical storage location; rather, we’re referencing a secure subsystem that operates via a combination of hardware and software elements. The precise implementation varies depending on the Android version and the underlying hardware. Understanding this nuance is essential for any developer dealing with sensitive cryptographic material.

At its core, the Android keystore system functions as a *credential vault*. Instead of the application directly accessing the private key material, the operating system provides an api that handles all the secure storage and cryptographic operations. This separation of concerns significantly enhances security, as it limits the exposure of the sensitive key material to applications. In earlier Android versions, typically everything up to Android 5.1 (api level 22), keystore relied more heavily on software-based storage. The keys, though stored securely within the application’s sandbox, were more susceptible to attacks exploiting vulnerabilities at the operating system layer. Often, keys were derived from user passwords or passcodes and encrypted using these derivatives. While relatively secure against basic attacks, they were not impervious, especially given advanced exploits targeting kernel vulnerabilities.

Starting from Android 6.0 (api level 23), google introduced the hardware-backed keystore (also called secure element or trusted execution environment (tee) depending on the hardware vendor). In this model, cryptographic operations happen within dedicated hardware that's isolated from the main operating system. The key material, after generation, never leaves the secure hardware. The primary goal of hardware-backed keystore is to protect keys from compromised systems. If an adversary manages to gain root privileges on the operating system or even re-flashes the entire device, keys stored in a hardware-backed keystore remain inaccessible. These secure hardware environments are generally proprietary to the hardware manufacturers, meaning that there will be slight variations across different devices.

Now, let’s address how these keys are actually stored and accessed. When an application requests a key to be generated or imports an existing key, the keystore service uses key management daemons to interact with the keystore hardware. For software-backed keystore, keys are typically stored in encrypted files within the application's private storage area, generally under `/data/user_de/<user id>/<package name>/`. The precise location is not guaranteed to be consistent across versions or devices, hence developers should use the keystore apis and not rely on specific path. The encryption of these files use a key which is derived from the device lock pattern, pin, or password, combined with other hardware-specific keys to make the storage sufficiently secure against attackers. When hardware-backed keystore is used, key material is stored by the secure hardware component, and the system only maintains pointers (handles) to the keys. When an application requests to use a key (e.g., for signing or decryption), the system then uses this pointer to direct the request to the secure hardware to perform cryptographic operations without exposing the actual key. The results are passed back to the application.

To illustrate these concepts, let’s look at some code examples, remembering this is simplified and ignores error handling for brevity:

**Snippet 1: Key Generation Using Software-Backed Keystore**

```java
import java.security.*;
import java.security.spec.RSAKeyGenParameterSpec;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import javax.crypto.KeyGenerator;

public void generateSoftwareKey(String alias) throws Exception {
    KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
    keyStore.load(null);

    if (!keyStore.containsAlias(alias)) {
        KeyGenerator keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore");
        KeyGenParameterSpec keyGenSpec = new KeyGenParameterSpec.Builder(alias,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_ECB)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
                .setKeySize(2048) // example key size
                .build();
        keyGenerator.init(keyGenSpec);
        keyGenerator.generateKey();
    }
}

```
This snippet demonstrates creating an rsa keypair in a software backed keystore. The key material gets generated and stored using encryption based on the lock screen pin or password, within the app’s dedicated storage directory. The api itself abstracts the location away from the user.

**Snippet 2: Key Generation Using Hardware-Backed Keystore (Attempt)**
```java
import java.security.*;
import java.security.spec.RSAKeyGenParameterSpec;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import javax.crypto.KeyGenerator;

public void generateHardwareKey(String alias) throws Exception {
    KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
    keyStore.load(null);

    if (!keyStore.containsAlias(alias)) {
        KeyGenerator keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore");
        KeyGenParameterSpec keyGenSpec = new KeyGenParameterSpec.Builder(alias,
                        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_ECB)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
                .setKeySize(2048)
                .setIsStrongBoxBacked(true) // This line asks for a hardware-backed key
                .build();

        keyGenerator.init(keyGenSpec);
        keyGenerator.generateKey();

    }
}
```
This snippet mirrors the first, but attempts to request a hardware backed key by setting the `setIsStrongBoxBacked(true)` attribute. The underlying system attempts to fulfill the request, but if a hardware backed keystore is unavailable, the framework reverts to software. This behavior is implicit, and the application receives no indication on the storage medium. Therefore, it's essential to verify the capabilities and key characteristics if a hardware-backed key is required.

**Snippet 3: Checking if a key is hardware backed**
```java
import java.security.*;
import java.security.spec.RSAKeyGenParameterSpec;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import java.security.cert.Certificate;
import android.security.keystore.KeyInfo;
import javax.crypto.KeyGenerator;


public boolean isKeyHardwareBacked(String alias) throws Exception {
    KeyStore keyStore = KeyStore.getInstance("AndroidKeyStore");
    keyStore.load(null);
    if (!keyStore.containsAlias(alias)) {
       return false;
    }
    Certificate cert = keyStore.getCertificate(alias);
    if(cert == null){
        return false;
    }
    PublicKey publicKey = cert.getPublicKey();
    if(!(publicKey instanceof java.security.interfaces.RSAKey)){
        return false;
    }
    KeyInfo keyInfo = ((android.security.keystore.RSAKey)publicKey).getKeyInfo();

    return keyInfo.isInsideSecureHardware();


}
```
This snippet shows a method to check if an existing key stored in the keystore is actually hardware backed using the `getKeyInfo()` method and `isInsideSecureHardware()`. This verification is crucial for ensuring proper security. If an application requires the hardware keystore and it is not present, alternative approaches are needed, often involving error messages or graceful fallback to less secure methods.

For deeper understanding and more formal documentation, I highly recommend reviewing the Android developer documentation on [Android Security](https://developer.android.com/training/articles/security) and specifically the section on [Android Keystore System](https://developer.android.com/training/articles/keystore). In particular, pay close attention to the [KeyGenParameterSpec](https://developer.android.com/reference/android/security/keystore/KeyGenParameterSpec) class. For a more academic approach, the [“Security and Privacy in Mobile Systems”](https://www.springer.com/gp/book/9783030153785) book by Dimitrios Zissis provides a comprehensive discussion of security architectures on mobile platforms. Additionally, consider the [“Security Engineering”](https://www.cl.cam.ac.uk/~rja14/book.html) by Ross Anderson for a broader understanding of security principles applicable to the field. Also consider the [“Handbook of Applied Cryptography”](https://cacr.uwaterloo.ca/hac/) for a comprehensive understanding of cryptography.

In short, the Android keystore is not about a single file; it's an ecosystem with variable storage methods depending on several factors. The keys themselves may be software-protected on lower-end devices or stored in tamper-resistant hardware on premium devices. What’s paramount is that developers should *always* interact with the keystore using its apis, never directly attempting to find or modify the underlying storage locations. This approach is the best way to ensure security and forward compatibility in the Android ecosystem.
