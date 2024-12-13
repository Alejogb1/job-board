---
title: "why is signhash from rsacng dotnetcore different from boncycastle running in?"
date: "2024-12-13"
id: "why-is-signhash-from-rsacng-dotnetcore-different-from-boncycastle-running-in"
---

Alright let's dive into this signhash discrepancy between RSACng and Bouncy Castle on .NET Core it's a classic head-scratcher that I’ve personally wrestled with more times than I’d like to admit

So you’re seeing different output right Specifically when generating a signature with `RSACng` in .NET Core and then trying to verify it using Bouncy Castle it’s failing or vice-versa This isn’t a bug per se but more a difference in how the two libraries handle the underlying mechanics of RSA signing particularly with regards to padding and encoding of the data prior to the actual signing operation

First off let's clarify this isn't a case of one being right and the other being wrong it’s more about them having different defaults and that’s where the problem usually lies In the real world of cryptography things get complex fast

See RSACng which is a .NET wrapper around Microsoft's CryptoAPI and the newer CNG cryptography API tends to be more flexible and allows you to explicitly specify padding schemes and digest algorithms via `RSASignaturePadding` and `HashAlgorithmName` objects

Bouncy Castle on the other hand while also robust might have some quirks or different defaults when it comes to signing and data digestion It expects the data to be structured in a particular way and sometimes assumes a default padding scheme so let's see where most people get stuck the encoding bit

Let’s talk code first this is usually more clarifying I’ll give you a .NET Core example using RSACng:

```csharp
using System;
using System.Security.Cryptography;
using System.Text;

public class RsaCngSigner
{
    public static byte[] SignData(byte[] data, RSAParameters privateKeyParams)
    {
        using (var rsa = new RSACng())
        {
            rsa.ImportParameters(privateKeyParams);
            byte[] signature = rsa.SignData(data, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            return signature;
        }
    }
}
```

This code snippet shows how to sign data using `RSACng` with `PKCS1` padding and `SHA256` hashing which are common choices I found myself using those more than anything else when handling those issues

Now for a Bouncy Castle example which many struggle with:

```csharp
using Org.BouncyCastle.Crypto;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Security;
using Org.BouncyCastle.Crypto.Signers;
using System.Text;

public class BouncyCastleSigner
{
    public static byte[] SignData(byte[] data, RsaKeyParameters privateKeyParams)
    {
        var signer = SignerUtilities.GetSigner("SHA256withRSA");
        signer.Init(true, privateKeyParams);
        signer.BlockUpdate(data, 0, data.Length);
        return signer.GenerateSignature();
    }
}
```

So here Bouncy Castle uses a string specifier "SHA256withRSA" this means it does both hashing and RSA signing together it might seem a bit different but it does the same job behind the scenes If you don't specify padding Bouncy castle might defaults to PKCS1 but the devil is in the details the key part here is in the `RSAParameters` for .NET and `RsaKeyParameters` for Bouncy Castle so they must be correct and the same

One gotcha I've seen many many times is the format of the key itself Make sure that the keys that are provided are in the correct format. Many struggle with the RSA parameter conversion between those two libraries. Here is an helper class:

```csharp
using System;
using System.Security.Cryptography;
using Org.BouncyCastle.Crypto;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Math;

public static class RsaKeyConverter
{
    public static RsaKeyParameters ToBouncyCastlePrivateKey(RSAParameters rsaParams)
    {
        BigInteger modulus = new BigInteger(1, rsaParams.Modulus);
        BigInteger privateExponent = new BigInteger(1, rsaParams.D);
        return new RsaPrivateCrtKeyParameters(modulus, new BigInteger(1, rsaParams.Exponent), privateExponent,
            new BigInteger(1, rsaParams.P), new BigInteger(1, rsaParams.Q),
            new BigInteger(1, rsaParams.DP), new BigInteger(1, rsaParams.DQ),
            new BigInteger(1, rsaParams.InverseQ));
    }
      public static RsaKeyParameters ToBouncyCastlePublicKey(RSAParameters rsaParams)
    {
          BigInteger modulus = new BigInteger(1, rsaParams.Modulus);
        BigInteger publicExponent = new BigInteger(1, rsaParams.Exponent);
        return new RsaKeyParameters(false,modulus,publicExponent);
    }
    public static RSAParameters ToDotNetRSAParameters(RsaKeyParameters rsaKeyParameters)
    {
        RSAParameters rsaParams = new RSAParameters();
        rsaParams.Modulus = rsaKeyParameters.Modulus.ToByteArrayUnsigned();
        rsaParams.Exponent = rsaKeyParameters.Exponent.ToByteArrayUnsigned();

        if(rsaKeyParameters is RsaPrivateCrtKeyParameters privateKey)
        {
            rsaParams.D = privateKey.D.ToByteArrayUnsigned();
            rsaParams.P = privateKey.P.ToByteArrayUnsigned();
            rsaParams.Q = privateKey.Q.ToByteArrayUnsigned();
            rsaParams.DP = privateKey.DP.ToByteArrayUnsigned();
            rsaParams.DQ = privateKey.DQ.ToByteArrayUnsigned();
            rsaParams.InverseQ = privateKey.QInv.ToByteArrayUnsigned();

        }
        return rsaParams;
    }
}
```

This helper class will convert keys between libraries and the key thing here is that if your keys are not in the right format it is the number one reason for different signhashes generated by each library

Now the main reason for sign hash differences is usually due to these common things:

**Padding Differences:** RSACng allows to specify the padding algorithm `RSASignaturePadding.Pkcs1` or `RSASignaturePadding.Pss` Bouncy castle defaults to PKCS#1 v1.5 padding which is less secure and also sometimes called v1.5 padding not to be confused with 2.1

**Hashing Algorithm Differences:** Both allow for various hashing algorithms but they must match. You need to make sure both are using the same hash like SHA256. If you specify SHA-1 on one and SHA256 on the other you'll get different outputs it's that simple really

**Data Encoding:** Ensure your input data is encoded identically before hashing Sometimes people sign with string data not byte data there are implicit conversions in .NET that can alter the bytes that are going to be signed If you sign "hello" string in UTF-8 make sure that the other side also uses UTF-8 if you don't get them synced you'll have a bad time

**Key Format Issues**: As I mentioned above RSAParameters and RsaKeyParameters are different in memory representations You must convert them correctly otherwise you'll be signing with different keys and we all know what that means bad crypto

Now let me tell you a real-world story I once spent a whole weekend debugging an issue like this I had a system where a legacy java backend with Bouncy Castle was signing data for a new .NET Core service which used RSACng and the signatures were failing like they were on vacation. Turns out the Java system was defaulting to `SHA1withRSA` with `PKCS#1 v1.5` padding while I was using SHA256 with no padding and this wasn't even funny I mean I’ve seen more entertaining things at a tax audit it was that tedious to debug

The fix ended up being simple but the debugging was pure hell. I had to force the java system to use the same hash and padding and use a different key parameter and then after an hour of debugging with both systems I managed to get the signatures matched. It was a lesson in how subtle differences in crypto libraries can lead to big headaches

To resolve this issue in your situation you need to do a few things:

1.  **Explicitly Specify Padding:** In your RSACng code use `RSASignaturePadding.Pkcs1` or `RSASignaturePadding.Pss` and in bouncy castle "SHA256withRSA" defaults to PKCS1 v1.5. If you need PSS you must call `PssSigner` explicitly.
2.  **Explicitly Specify Hash Algorithm**: Ensure that in both libraries you're using the same hash algorithm SHA256 is recommended over SHA1 because of security concerns it's generally what you see nowadays.
3.  **Ensure Same Encoding:** Always use byte arrays and ensure the encoding of data is the same when signing. The string "hello" in UTF8 is not the same as the same "hello" in ASCII

For further reading you can consult the following resources instead of links:

*   **Applied Cryptography** by Bruce Schneier is a great book to start understanding the mechanics of crypto algorithms
*   **Handbook of Applied Cryptography** by Alfred Menezes et al gives a very detailed approach and it's a must-read for anyone working with crypto implementations
*   **RFC 8017** which is the standard document that specifies PKCS #1 which is an important reading for both padding and encryption mechanisms

I hope this helps clear things up it's a common issue and understanding these nuances is essential when working with different cryptographic libraries Good luck with your debugging and may the crypto gods be on your side because I’ve seen how cruel they can be
