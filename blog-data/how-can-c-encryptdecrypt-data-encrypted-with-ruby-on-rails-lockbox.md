---
title: "How can C# encrypt/decrypt data encrypted with Ruby on Rails' Lockbox?"
date: "2024-12-23"
id: "how-can-c-encryptdecrypt-data-encrypted-with-ruby-on-rails-lockbox"
---

Alright, let's talk about cross-platform encryption, specifically tackling the challenge of decrypting data in C# that was encrypted using Ruby on Rails' Lockbox gem. This isn't a theoretical exercise; I've faced this precise interoperability headache more than once across project boundaries and legacy systems. It's a common pitfall when integrating services developed with disparate tech stacks, and getting it wrong leads to data corruption, security vulnerabilities, and lots of late-night debugging sessions.

The core issue here revolves around differing encryption implementations, even if the underlying algorithms are conceptually the same. Lockbox in Ruby on Rails leverages `libsodium` for its heavy lifting, providing an abstraction layer that handles key derivation, nonce management, and authenticated encryption. We need to match that implementation precisely within the C# environment to successfully decrypt the data. Trying to do this without a clear understanding of the underlying processes is asking for trouble.

Here’s a breakdown of the steps, and the critical details, to get this working reliably. Firstly, it’s essential to understand how Lockbox operates: it typically employs authenticated encryption with associated data (AEAD). This usually involves an encryption key (derived from a secret), a nonce (a unique, randomly generated value), and potentially associated data that is not encrypted but is included in the authentication process. The specific AEAD algorithm used is often `XChaCha20-Poly1305`, but it's critical to confirm that within the Lockbox configuration of the Ruby app. Failing to match that precisely will, of course, lead to decryption failures.

Secondly, the key derivation used by Lockbox is vital. It often uses a key derivation function (KDF), typically something like `scrypt` or `argon2`. The parameters for this KDF - salt, work factor, and other configuration values - are absolutely crucial. If you have these wrong in C#, even with the correct key, you won't decrypt the data. You'll need to acquire these parameters from the Lockbox configuration in the Ruby on Rails application. The documentation often defaults to `scrypt`, but this should be verified.

Let me illustrate this with some C# code snippets. We’ll use the `libsodium.net` NuGet package, a C# port of `libsodium`. This package allows us to replicate Lockbox's low-level encryption functions. Also note that there are other libraries that do similar jobs, it's important to pick one and stick with it for consistency, and I tend to use this as it's the closest to the underlying implementation.

**Code Snippet 1: Decryption Implementation**

```csharp
using Sodium;
using System;
using System.Text;

public static class LockboxDecryptor
{
    public static string Decrypt(byte[] encryptedData, string key, byte[] associatedData, byte[] nonce, byte[] salt, int workFactor)
    {
        byte[] keyBytes = Encoding.UTF8.GetBytes(key);
        byte[] derivedKey = null;
        try {
            derivedKey = CryptoScrypt.KeyDerivation(keyBytes, salt, 32, workFactor);
        } catch (Exception ex) {
           Console.WriteLine($"Key Derivation error: {ex.Message}");
            return null;
        }

        if (derivedKey == null)
        {
            Console.WriteLine("Key Derivation failed.");
            return null; // Handle key derivation failure
        }

        byte[] decrypted = null;

        try {
           decrypted = SecretAead.XChaCha20Poly1305Decrypt(encryptedData, associatedData, nonce, derivedKey);
        } catch (Exception ex) {
           Console.WriteLine($"Decryption error: {ex.Message}");
           return null;
        }

        if (decrypted == null) {
            Console.WriteLine("Decryption failed.");
            return null;
        }


        return Encoding.UTF8.GetString(decrypted);
    }
}
```

This snippet is the heart of the decryption process. The `Decrypt` method takes the encrypted data, the encryption key (the secret from your Rails app), associated data, nonce, salt, and work factor. It first derives the actual encryption key using `scrypt` based on those values. Then, it decrypts the data using `SecretAead.XChaCha20Poly1305Decrypt`. Notice the error handling; decryption errors can happen due to key mismatches, incorrect nonce, tampered data, or mismatched parameters.

**Code Snippet 2: Example Usage**

```csharp
public class Program
{
    public static void Main(string[] args)
    {
         // Example values, replace with the data from your lockbox.
        string encryptedDataHex = "b351d00a89c0c2a1a5a2339b4b76...";
        string encryptionKey = "your_secret_key_from_rails";
        string associatedDataHex = "012233445566778899aabbccddeeff";
        string nonceHex = "f0e1d2c3b4a5968778695a4b3c2d1e0f";
        string saltHex = "1234567890abcdef"; //Salt value from rails lockbox
        int workFactor = 16; // Work factor from rails lockbox


        byte[] encryptedData = HexStringToByteArray(encryptedDataHex);
        byte[] associatedData = HexStringToByteArray(associatedDataHex);
        byte[] nonce = HexStringToByteArray(nonceHex);
        byte[] salt = HexStringToByteArray(saltHex);

        string decryptedText = LockboxDecryptor.Decrypt(encryptedData, encryptionKey, associatedData, nonce, salt, workFactor);


        if(decryptedText != null) {
            Console.WriteLine($"Decrypted Text: {decryptedText}");
        } else {
           Console.WriteLine($"Decryption failed.");
        }

    }

      public static byte[] HexStringToByteArray(string hex) {
        return Enumerable.Range(0, hex.Length)
                         .Where(x => x % 2 == 0)
                         .Select(x => Convert.ToByte(hex.Substring(x, 2), 16))
                         .ToArray();
     }
}
```

This snippet shows you how to call the decrypt function. It first converts the hex encoded strings to byte arrays, then passes them to the decryption function along with the key, salt, and work factor. You’ll need to extract the correct hexadecimal representations of your encrypted data, associated data, nonce, and salt from your Ruby on Rails application’s logs or storage. The 'work factor' is also critical and must match. This snippet includes the `HexStringToByteArray` helper function to convert hex strings into byte arrays that can be passed to the crypto functions.

**Code Snippet 3: Handling Different Key Derivation functions**

```csharp
using Sodium;
using System;
using System.Text;

public static class LockboxDecryptor
{
    public static string Decrypt(byte[] encryptedData, string key, byte[] associatedData, byte[] nonce, byte[] salt, int workFactor, string kdf="scrypt")
    {
        byte[] keyBytes = Encoding.UTF8.GetBytes(key);
        byte[] derivedKey = null;

        try {
          if (kdf.ToLower() == "scrypt") {
             derivedKey = CryptoScrypt.KeyDerivation(keyBytes, salt, 32, workFactor);
          } else if (kdf.ToLower() == "argon2") {
           //  This would require a separate Argon2 implementation or a package like Konscious.Security.Cryptography.Argon2
            //  derivedKey = Argon2.Hash(keyBytes,salt, salt.Length, workFactor /* Parallelism */, workFactor /* Memory Size */, 32);
            Console.WriteLine($"Argon2 KDF is not implemented in this example, configure scrypt in rails instead.");
             return null;
         } else {
            Console.WriteLine($"Unknown KDF: {kdf}");
            return null;
         }
        } catch (Exception ex) {
           Console.WriteLine($"Key Derivation error: {ex.Message}");
            return null;
        }

         if (derivedKey == null)
        {
            Console.WriteLine("Key Derivation failed.");
            return null; // Handle key derivation failure
        }

        byte[] decrypted = null;

        try {
           decrypted = SecretAead.XChaCha20Poly1305Decrypt(encryptedData, associatedData, nonce, derivedKey);
        } catch (Exception ex) {
           Console.WriteLine($"Decryption error: {ex.Message}");
           return null;
        }

        if (decrypted == null) {
            Console.WriteLine("Decryption failed.");
            return null;
        }

        return Encoding.UTF8.GetString(decrypted);
    }
}

```

This third snippet extends the first to account for differing key derivation functions. While `scrypt` is commonly used by Lockbox, `argon2` can also be employed. You'll need an Argon2 implementation in C# to fully realize this. This snippet shows a basic implementation of the switch, but it's important to use a proper library for Argon2 if that's what the ruby project is using. The key takeaway is flexibility and understanding that the key derivation is just as critical as the encryption algorithm itself.

For resources, I'd recommend checking out the official `libsodium` documentation first, to understand the underlying primitives. Additionally, "Cryptography Engineering" by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno provides deep theoretical grounding in applied cryptography and would be beneficial. Finally, if Argon2 is used, the official IETF specification RFC9106 would be useful.

In practice, cross-platform crypto isn’t magic. Getting it right involves careful scrutiny of the Ruby/Rails configuration, meticulously matching the algorithms, and paying attention to every single parameter. This isn't something you guess your way through; a methodical approach is key, and the code samples above should help you get started.
