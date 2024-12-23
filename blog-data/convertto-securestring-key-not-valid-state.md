---
title: "convertto-securestring key not valid state?"
date: "2024-12-13"
id: "convertto-securestring-key-not-valid-state"
---

 so I've seen this rodeo before plenty of times "convertto-securestring key not valid state" yeah that old chestnut hits close to home let me tell you this isn't some obscure corner case its a right of passage for anyone playing with powershell and encrypted strings its like the initiation ritual they forgot to put in the official documentation

This thing pops up when you are trying to convert something to a secure string using a key that's well not in a state where its happy I mean secure strings aren't just some text with a fancy hat they rely on cryptographic APIs under the hood and that key is a crucial player its like the bouncer at a very exclusive encrypted club and if that bouncer isn't feeling it you're not getting in

First thing I did when I smacked into this problem back in my early days I was trying to script an automated deployment thing with sensitive passwords and connection strings you know the usual stuff And man oh man the frustration I went through like trying to debug a perl script with only a magnifying glass and some hopes I was messing with system credentials and passing them around with secure strings as a first step And I keep getting "key not in valid state" I kept checking and rechecking my spelling and tried to understand what exactly a key needs to be in a "valid state"

I had to spend a whole evening deep diving into Windows DPAPI the data protection API to understand it it felt like reading hieroglyphs after coding all day but hey that's what we do right? It turns out a secure string's key which you don't often get to see directly usually its tied to the user account's credentials or some machine level key store Its a bit complex and windows handles the key generation for us and usually handles the lifecycle management but if things get corrupted or are used outside of the context they were generated it goes sideways very very fast

So the error you're getting suggests the key its trying to use to encrypt the string is messed up its either unavailable or its permission is somehow not adequate or its just plain unusable It's less about the string itself and more about the system's ability to access its own encryption materials

Now the good news is I've gone through this enough to have a few reliable workarounds in my arsenal usually its something to do with identity context

Here's a typical scenario imagine your trying to encrypt a secret on one machine and then try and use that encrypted string on another machine or even on the same machine under a different user account this is a textbook example of where this error often shows up cause the DPAPI key is user specific

So let's look at some code snippets I mean we're tech people after all show not tell right?

**Scenario 1 Same User different process**

This might seem simple but often when you start playing with automation and service accounts this bites you

```powershell
# This might fail in some situations even on the same machine
$plaintext = "SuperSecretPassword123!"
$secureString = ConvertTo-SecureString $plaintext -AsPlainText -Force

# Try to use the secure string
$secureString
```
What is happening here is that sometimes the context under which you create the secure string and under which you use can be different even under the same user for example using a scheduled task or a different service instance and therefore resulting in the key not being valid for this new context

**Scenario 2 Inter-machine or cross user credential problems**

```powershell
# DO NOT SHARE A SECURE STRING DIRECTLY ACROSS MACHINES IT WILL FAIL IN ALMOST ALL SITUATIONS
# This WILL ALWAYS FAIL in almost every scenario
$plaintext = "SuperSecretPassword123!"
$secureString = ConvertTo-SecureString $plaintext -AsPlainText -Force
# serialize the secure string as base64
$serializedString = [System.Convert]::ToBase64String([System.Runtime.InteropServices.Marshal]::Copy($secureString,0,0,$secureString.Length))

# later on the other machine
$newSecureString = [System.Runtime.InteropServices.Marshal]::Copy([System.Convert]::FromBase64String($serializedString),0,0,[System.Convert]::FromBase64String($serializedString).Length)
$newSecureString
# you will see the key is not valid message as this is not usable
```

What is happening here is that secure strings are not directly transferable between machines or users. Each user and often process has its own encryption keys meaning that even a serialized string will not work because it will try to decrypt using keys from the new environment

So the question becomes what can we actually do that is a very valid question for anyone who's dealt with secure strings

Well usually we go for asymmetric encryption to get out of this mess its a bit more involved but at the end of the day it solves the problem

**Scenario 3 using asymetric encryption**

```powershell
# Generate a public and private key pair
$rsa = [System.Security.Cryptography.RSACryptoServiceProvider]::new()
$publicKey = $rsa.ToXmlString($false) # Export public key
$privateKey = $rsa.ToXmlString($true) # Export private key

# Encrypt the data with the public key
$plaintext = "MySuperSecretPassword"
$publicKeyObject = [System.Security.Cryptography.Xml.RSAKeyValue]::new()
$publicKeyObject.LoadXml($publicKey)
$publicKeyProvider = [System.Security.Cryptography.RSACryptoServiceProvider]::new()
$publicKeyProvider.ImportParameters($publicKeyObject.ExportParameters())
$encryptedBytes = $publicKeyProvider.Encrypt([System.Text.Encoding]::UTF8.GetBytes($plaintext), $false)
$encryptedString = [Convert]::ToBase64String($encryptedBytes)

# Later you decrypt using the private key you did not share in the first place
$privateKeyObject = [System.Security.Cryptography.Xml.RSAKeyValue]::new()
$privateKeyObject.LoadXml($privateKey)
$privateKeyProvider = [System.Security.Cryptography.RSACryptoServiceProvider]::new()
$privateKeyProvider.ImportParameters($privateKeyObject.ExportParameters())
$decryptedBytes = $privateKeyProvider.Decrypt([Convert]::FromBase64String($encryptedString), $false)
$decryptedText = [System.Text.Encoding]::UTF8.GetString($decryptedBytes)

$decryptedText
# The decryptedText should now have our original "MySuperSecretPassword" string
```

This example uses the RSA algorithm to create a public and private keypair public key is used for encryption and private is used for decryption it allows us to share encrypted data with the public key and only people with the private key can decrypt the data you must be careful in how you store your keys as it has a lot of impact in the level of security of your solution and as a general rule you should always keep the private key safe

Now a bit about resources if you are thinking about reading up on this topic instead of following tutorials you could check "Applied Cryptography" by Bruce Schneier its like the bible of cryptography a bit dense but it will give you a deep understanding of these concepts or if you are looking for something less intense a good book is "Cryptography Engineering" by Niels Ferguson and Bruce Schneier they break down complex concepts into digestible chunks and this is a very important skill in the world of crypto. And yes both are Schneier cause he kinda wrote the book on it in this area.

One more piece of advice do not try to roll your own crypto that is very bad practice unless you are a crypto expert its really easy to make a simple mistake that makes your system super vulnerable. Use well tested secure libraries and crypto implementations such as the one used in the last example. Now if you try to add a new encryption algorithm without being an expert you might find that when you try to decrypt you will get "key not in valid state" cause no one else implemented your encryption algorithm hahaha just joking don't try to implement your own encryption algorithms from scratch I'm just kidding

So in short when you hit "convertto-securestring key not valid state" its not the end of the world its a good prompt to revisit how windows manages keys and how you manage sensitive information in your scripts. Remember identity context is key and that you should never directly transfer secure strings and instead you should try to use asymmetric encryption if you need to transport secured data across systems and always always protect your keys I've lost too many hairs over this issue and hopefully this saves some hairs for you
