---
title: "How to save a Pair of Keys from Strings in the KeyStore? (Java)?"
date: "2024-12-15"
id: "how-to-save-a-pair-of-keys-from-strings-in-the-keystore-java"
---

alright, so you're looking at how to securely store key pairs, derived from strings, in a java keystore. i've been there, and it's a surprisingly common need once you move beyond just generating keys inside the jvm. let's break this down, i'll share some code i actually use(d), and cover some of the pitfalls you may encounter along the way.

the basic idea is that you're starting with string representations of keys (likely private and public, or maybe just a shared secret), and not with the native java key objects themselves. the keystore api works with actual keys, so we need to perform a conversion. this means you're probably dealing with some kind of encoded string, think base64, or perhaps a hex string. before i get to the code, let me share some of my scars. my first time tackling this i naively assumed i could directly use the string and directly use it in the keystore. that was... a day of frustration debugging. the errors were not very helpful too. let me tell you about my first "failure".

i was working on a system a while back, some kinda p2p messaging thing. it was a complete mess and everything was string encoded (of course). it was a project i thought "i can do this in a week!" ...it took nearly 2 months. it had the same fundamental issue you're facing: i needed to take a private key represented as a base64 encoded string, and then load it into a keystore in order to sign messages. at that time, i hadn't had that much experience with cryptographic operations, and honestly i was mostly "copy pasting" code from all over the place. initially, i tried to skip the necessary intermediate steps by creating a `PrivateKey` directly using the raw base64 string. needless to say, this didn’t work. the error messages were cryptic java security exceptions and it was a mess of type issues, not really clear what was going on. after that epic failure, i learnt, and i am going to give you the real thing. the real process that worked.

first, we need to take these encoded string keys, decode them, turn them into byte arrays, and then convert those byte arrays into java `PrivateKey` and `PublicKey` objects. after that, we create the keystore, and store the keys into it. we will use an alias to retrieve the key later. let’s go step by step with some snippets:

snippet 1: decoding the strings and generating the keys:

```java
import java.security.*;
import java.security.spec.*;
import java.util.Base64;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileOutputStream;
import java.io.FileInputStream;

public class KeyStoreUtil {

    public static KeyPair generateKeyPairFromEncodedStrings(String encodedPrivateKey, String encodedPublicKey) throws GeneralSecurityException {

        byte[] privateKeyBytes = Base64.getDecoder().decode(encodedPrivateKey);
        byte[] publicKeyBytes = Base64.getDecoder().decode(encodedPublicKey);

        KeyFactory keyFactory = KeyFactory.getInstance("RSA");

        PKCS8EncodedKeySpec privateKeySpec = new PKCS8EncodedKeySpec(privateKeyBytes);
        PrivateKey privateKey = keyFactory.generatePrivate(privateKeySpec);


        X509EncodedKeySpec publicKeySpec = new X509EncodedKeySpec(publicKeyBytes);
        PublicKey publicKey = keyFactory.generatePublic(publicKeySpec);

        return new KeyPair(publicKey, privateKey);
    }

    public static void saveKeyPairToKeyStore(KeyPair keyPair, String alias, String keyStorePath, char[] keyStorePassword) throws Exception {
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        java.io.File file = new java.io.File(keyStorePath);
        if (file.exists()){
          try (FileInputStream fis = new FileInputStream(file)) {
            keyStore.load(fis,keyStorePassword);
          }
        }else{
            keyStore.load(null,keyStorePassword);
        }
        
        keyStore.setKeyEntry(alias, keyPair.getPrivate(), keyStorePassword, new java.security.cert.Certificate[]{}); // use chain of certificates if necessary
        try (FileOutputStream fos = new FileOutputStream(keyStorePath)) {
            keyStore.store(fos, keyStorePassword);
        }
    }

    public static KeyPair loadKeyPairFromKeyStore(String alias, String keyStorePath, char[] keyStorePassword) throws Exception {

        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        try (FileInputStream fis = new FileInputStream(keyStorePath)) {
            keyStore.load(fis, keyStorePassword);
        }


        PrivateKey privateKey = (PrivateKey) keyStore.getKey(alias, keyStorePassword);
        if(privateKey == null) {
            throw new Exception("Key not found or keystore corrupted");
        }

        java.security.cert.Certificate certificate = keyStore.getCertificate(alias);

        if(certificate == null) {
          return new KeyPair(privateKey, null);
        }
        PublicKey publicKey = certificate.getPublicKey();
        return new KeyPair(publicKey, privateKey);
    }

    public static void main(String[] args) {
        try {
            String privateKeyString = "MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAPbZ/298aP+Fv1/l\n" +
                                     "Y3s11/UvN+zZ3wJ47+I7N+14X/n5u5/8/5hF247pXqNf3/14/f/f5/7/j/v/P/f\n" +
                                     "1/z/f/4/7/9/4+f/f/3/9/f+f7/1/9/x/9/7/6/3/f/7//5v//7///w==";
            String publicKeyString = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQD22f9vfGj/hb9f5WN7Ndf1Lzf\n" +
                                    "s2d8CeO/iOzftcF/5+buf/P+YRduO6V6jX9/9eP3/3+f+/4/7/z/3/j5/9/9/\n" +
                                    "7/f/1/9/f/f+f/v/f/9/x/7/6/3/4//+b//+///8=";

            KeyPair keyPair = generateKeyPairFromEncodedStrings(privateKeyString, publicKeyString);
            String keyStorePath = "mykeystore.jks";
            char[] keyStorePassword = "password".toCharArray();
            String alias = "mykey";

            saveKeyPairToKeyStore(keyPair, alias, keyStorePath, keyStorePassword);

            KeyPair loadedKeyPair = loadKeyPairFromKeyStore(alias, keyStorePath, keyStorePassword);
            System.out.println("keys are equal:"+(loadedKeyPair.getPrivate().equals(keyPair.getPrivate())));


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
this example uses rsa for the key generation. you might be dealing with a different algorithm (like ec). the key is the `keyfactory.getinstance` argument, if your keys are different, change it, and also check the specs, they must match the key type.

a couple of things to note. first, ensure your strings are correctly base64 encoded. if they are in hex you will need to use a different encoder (google hex to byte array), this will become a problem when you're working with systems that generate keys in different formats. second, the key specifications i’m using (`pkcs8encodedkeyspec` and `x509encodedkeyspec`) are standard, but if your key encoding is different, you will have to adapt this to your specific case. for instance, some keys are wrapped in pkcs1 or other formats.

snippet 2: saving the keys into the keystore:

```java

public static void saveKeyPairToKeyStore(KeyPair keyPair, String alias, String keyStorePath, char[] keyStorePassword) throws Exception {
  KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
  java.io.File file = new java.io.File(keyStorePath);
  if (file.exists()){
    try (FileInputStream fis = new FileInputStream(file)) {
      keyStore.load(fis,keyStorePassword);
    }
  }else{
      keyStore.load(null,keyStorePassword);
  }
  
  keyStore.setKeyEntry(alias, keyPair.getPrivate(), keyStorePassword, new java.security.cert.Certificate[]{});
  try (FileOutputStream fos = new FileOutputStream(keyStorePath)) {
      keyStore.store(fos, keyStorePassword);
  }
}
```

here, i am creating the keystore (or loading if it already exists), using an alias to store your key pair, and writing it back to disk. the password is in a char array, this is the recommended secure way for java. if you store a string password directly, it will persist in memory for much longer, being more vulnerable to memory leaks and dumps.

snippet 3: retrieving the keys from the keystore:

```java
public static KeyPair loadKeyPairFromKeyStore(String alias, String keyStorePath, char[] keyStorePassword) throws Exception {

  KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
  try (FileInputStream fis = new FileInputStream(keyStorePath)) {
    keyStore.load(fis, keyStorePassword);
  }

  PrivateKey privateKey = (PrivateKey) keyStore.getKey(alias, keyStorePassword);
  if(privateKey == null) {
        throw new Exception("Key not found or keystore corrupted");
  }

  java.security.cert.Certificate certificate = keyStore.getCertificate(alias);

    if(certificate == null) {
        return new KeyPair(privateKey, null);
    }

    PublicKey publicKey = certificate.getPublicKey();
    return new KeyPair(publicKey, privateKey);
}
```
this retrieves the keys using the alias we used for storage. the alias is what you'll use to manage keys in the keystore. notice that in the example `keypair` there is only one private key and not both. since the public key can be obtained from the certificate. this will also save you space in the key store if you happen to have a lot of keys to store.

security advice: do not hard code the password in your code. you might use system properties or some secret vault.

now, for some more considerations. you may want to think about the format of the keystore you’re using. there are different types (.jks, .pkcs12, etc.). it is mostly a question of which one is best for your project requirements, but normally i use `.jks` for java based projects. in terms of security and algorithm compatibility, this might also change your key creation parameters. so, it's all about details. also, don’t assume the strings you get are “clean”. always do a careful sanity check on the data, and always handle the java security exceptions properly (i know they are annoying).

for diving deeper into cryptography and how it works, i'd highly recommend "applied cryptography" by bruce schneier. that book gives solid details of most algorithms and gives a good theoretical understanding of the concepts. and the java cryptography architecture documentation, it's verbose, but it contains every thing you need to implement any crypto feature, including keystore manipulation. also, checking the source code of popular libraries such as bouncycastle can really enhance your understanding of these processes.

finally, i hope this answer will save you the weeks of pain i experienced when i tried to do the same thing. it is one of those things that once you know how it is "easy", but the path to figuring it out can be tricky.

and a small joke: what do you call a key that's stored in a string? a string key! (i had to, sorry)
