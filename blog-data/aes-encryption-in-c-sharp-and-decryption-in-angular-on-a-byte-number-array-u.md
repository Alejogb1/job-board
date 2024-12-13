---
title: "aes encryption in c sharp and decryption in angular on a byte number array u?"
date: "2024-12-13"
id: "aes-encryption-in-c-sharp-and-decryption-in-angular-on-a-byte-number-array-u"
---

Alright so you're hitting that classic cross-platform AES encryption headache aren't you C# doing the heavy lifting with byte arrays and Angular trying to make sense of it all Yeah I've been there done that got the t-shirt more than a few times Let me tell you it's a common pitfall and it's usually down to a couple of things either the key handling is wonky or the encoding is messing with the data Or maybe even both So let's break this down I'm going to talk from my experience because I've spent way too many late nights debugging this specific scenario

First thing first C# AES using byte arrays is pretty straightforward but you need to be meticulous about the details I remember one time I was working on a secure data transfer project We had these sensor nodes pushing data up to a server and of course we needed to encrypt it The C# backend was generating the encrypted byte arrays and we thought we were golden I swear I spent three days staring at a seemingly random pile of garbage in the frontend until I realized we hadn't properly agreed on the IV initialization vector it was just random chaos

Here's a quick snippet demonstrating how you might set up C# AES encryption for byte arrays this is bare bones folks real world you need to handle exceptions properly and maybe use a key derivation function but let's keep it simple

```csharp
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public class AesHelper
{
    public static byte[] Encrypt(byte[] data, byte[] key, byte[] iv)
    {
        using (Aes aesAlg = Aes.Create())
        {
            aesAlg.Key = key;
            aesAlg.IV = iv;
            ICryptoTransform encryptor = aesAlg.CreateEncryptor(aesAlg.Key, aesAlg.IV);
            using (MemoryStream msEncrypt = new MemoryStream())
            {
                using (CryptoStream csEncrypt = new CryptoStream(msEncrypt, encryptor, CryptoStreamMode.Write))
                {
                   csEncrypt.Write(data, 0, data.Length);
                 }
                 return msEncrypt.ToArray();
            }
        }
    }
}

// Example usage
// byte[] key = Encoding.UTF8.GetBytes("1234567890123456");
// byte[] iv = Encoding.UTF8.GetBytes("abcdefghijklmnop");
// byte[] dataToEncrypt = Encoding.UTF8.GetBytes("This is secret data");
// byte[] encryptedData = AesHelper.Encrypt(dataToEncrypt,key,iv);
```

Notice I am using `Aes.Create` that will give you the best implementation provided by the operating system Also `MemoryStream` is your friend here for converting a data array into a `byte[]` that also does not require that you have to know the actual size of the array before hand You need to be very careful with your `key` and your `iv` they need to be the right size specifically 16 bytes for AES 128 24 for AES 192 and 32 bytes for AES 256 Your IV needs to be 16 bytes as well And make sure you're using the same key and IV for both encryption and decryption or you'll just get gibberish Seriously gibberish

Now over to Angular This is where the fun begins because JavaScript and byte arrays are not always best friends We need to be very meticulous in how we deal with the data received from your C# backend You have to make sure that what you’re sending on the C# side is precisely what you’re expecting on the Angular side no data loss or weird encoding or that kind of stuff If you are not familiar with how encoding works I would suggest you check out "Code: The Hidden Language of Computer Hardware and Software" by Charles Petzold This can give you all the nitty gritty details you need to know

One thing I noticed is that Base64 is a usual suspect in this kind of cross platform issues Many times what happens is that you send over some bytes you assume they are bytes then you encode them in base64 on one side and expect to get the same bytes on the other side. That is not how it works Base64 converts them to text so if you're receiving Base64 encoded data in Angular make sure you decode it back into a byte array before attempting decryption If you send an already base64 string that will not work You'll just be messing with random characters And remember if you send bytes you have to send bytes

Here's some Angular TypeScript code using `crypto-js` which is a very common library I've used it so many times I’ve lost count

```typescript
import * as CryptoJS from 'crypto-js';

export class CryptoHelper {

    static decrypt(encryptedData: number[], key: number[], iv: number[]): string {
        const keyBytes = CryptoJS.lib.WordArray.create(key);
        const ivBytes = CryptoJS.lib.WordArray.create(iv);
        const encryptedBytes = CryptoJS.lib.WordArray.create(encryptedData);

        const decrypted = CryptoJS.AES.decrypt({ ciphertext: encryptedBytes }, keyBytes, {
            iv: ivBytes,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });

        return decrypted.toString(CryptoJS.enc.Utf8);
    }
}

// Example usage:
// const key = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36];
// const iv = [0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70];
// const encryptedData = [23, 194, 184, 184, 25, 13, 9, 52, 43, 42, 117, 191, 15, 109, 191, 227]; // Example encrypted byte array
// const decryptedText = CryptoHelper.decrypt(encryptedData, key, iv);
// console.log(decryptedText);
```

Note how the `WordArray` are used and created I've spent considerable amounts of time on the details. `CryptoJS` expects `WordArray` instead of array of numbers also notice that when we created it we did it from an array of numbers that represent a byte not a string. You also have to use the correct mode here CBC mode is very common but if you were using another one you will have to adjust it And if you do not use the right padding also your decryption will fail It's worth reading the documentation provided by `crypto-js`. I would recommend checking their documentation online. Also make sure the key and the IV are of the same size if not you will have an exception thrown by the library which can be difficult to trace back if not handled properly

Now I am going to show you an example in C# in which I handle the key and the IV creation using the C# libraries This is slightly more complicated but if you plan on using this in production I highly encourage you to use this pattern

```csharp
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public class AesHelperAdvanced
{
    public static byte[] Encrypt(byte[] data, string key)
    {
        using (Aes aesAlg = Aes.Create())
        {
          byte[] keyBytes = Encoding.UTF8.GetBytes(key);
           Array.Resize(ref keyBytes,32);
            aesAlg.Key = keyBytes;
            aesAlg.GenerateIV();
            ICryptoTransform encryptor = aesAlg.CreateEncryptor(aesAlg.Key, aesAlg.IV);
            using (MemoryStream msEncrypt = new MemoryStream())
            {
                using (CryptoStream csEncrypt = new CryptoStream(msEncrypt, encryptor, CryptoStreamMode.Write))
                {
                   csEncrypt.Write(data, 0, data.Length);
                 }
                 var result = new byte[aesAlg.IV.Length + msEncrypt.ToArray().Length];
                 Array.Copy(aesAlg.IV,0,result,0,aesAlg.IV.Length);
                 Array.Copy(msEncrypt.ToArray(),0,result,aesAlg.IV.Length,msEncrypt.ToArray().Length);
                 return result;

            }
        }
    }


 public static byte[] Decrypt(byte[] data, string key)
    {
        using (Aes aesAlg = Aes.Create())
        {
           byte[] keyBytes = Encoding.UTF8.GetBytes(key);
           Array.Resize(ref keyBytes,32);
             aesAlg.Key = keyBytes;

            byte[] iv = new byte[aesAlg.BlockSize / 8];
             Array.Copy(data,0,iv,0,iv.Length);
            aesAlg.IV = iv;

             byte[] encryptedData = new byte[data.Length - iv.Length];
              Array.Copy(data,iv.Length,encryptedData,0,encryptedData.Length);
            ICryptoTransform decryptor = aesAlg.CreateDecryptor(aesAlg.Key, aesAlg.IV);
             using (MemoryStream msDecrypt = new MemoryStream(encryptedData))
            {
                using (CryptoStream csDecrypt = new CryptoStream(msDecrypt, decryptor, CryptoStreamMode.Read))
                {
                   using(MemoryStream resultStream = new MemoryStream()){
                     csDecrypt.CopyTo(resultStream);
                     return resultStream.ToArray();
                   }
                }
            }

        }
    }


}

//Example Usage
//string key = "MySuperSecretKey";
//byte[] dataToEncrypt = Encoding.UTF8.GetBytes("This is secret data");
//byte[] encryptedData = AesHelperAdvanced.Encrypt(dataToEncrypt,key);
//byte[] decryptedData = AesHelperAdvanced.Decrypt(encryptedData,key);
//string decryptedText = Encoding.UTF8.GetString(decryptedData);
//Console.WriteLine(decryptedText)
```

In this case you can see I’m using the C# library to create the initialization vector and appending it in front of the data. This gives the angular side a way to retrieve the correct data to perform the decryption You will also need to be careful in the Angular part to read the bytes. If you need some resource on the inner workings of the cryptographic primitives in use here I would recommend you check out "Serious Cryptography: A Practical Introduction to Modern Encryption" by Jean-Philippe Aumasson it goes deep into the math behind the implementation so you can actually know what is going on behind the scenes

The main thing to remember here is you need to check each step very carefully and make sure you are handling everything at the correct level of abstraction. So you should send bytes and receive bytes You should not be sending text that should be converted into something else that then will be converted back to something else you will not get the same thing If you see some data that look like garbage it most probably means that the encryption is working but the decryption is not or the reverse is happening There's no real magic here just careful attention to the details And I know it's a pain but honestly at this point in my career I have seen these types of issue many many times and I have a couple of tricks up my sleeve which usually involve me staring at the screen for hours until I find what happened and fix it or a co worker comes in and points out the obvious It is funny how things work like that.

Ok that should be enough to get you started If you have any other issues just let me know I might have had the same one already.
