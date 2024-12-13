---
title: "convertto-securestring key not valid use problem?"
date: "2024-12-13"
id: "convertto-securestring-key-not-valid-use-problem"
---

Okay so you’re wrestling with `ConvertTo-SecureString` puking about an invalid key huh Been there done that got the t-shirt probably several actually It's a classic PowerShell gotcha a real head scratcher when you first encounter it

Let me break it down from my perspective and maybe it will help you. I remember vividly back in the day probably circa PowerShell v3 maybe v4 we were deploying these little automation scripts for our dev environments. One thing we needed to do was store database connection strings you know passwords the sensitive stuff.

First thought was of course lets do `ConvertTo-SecureString` because you know security. Except it kept throwing this error "key not valid for use in specified state" right in my face. It was super frustrating. I mean the code looked right everything was perfect. I even re-typed the whole thing a couple times because sometimes I make typos you know we all do.

So what’s the deal right? The issue isn't necessarily about the *key* you think you’re using for the encryption it's more about the underlying mechanism that `ConvertTo-SecureString` employs which in general terms is the Data Protection API or DPAPI for short. DPAPI on Windows is tied to the user account and machine on which the secure string is generated. It's not like a simple symmetric key you can just move around.

Essentially you’re trying to use a secure string created under a different context be it a different user account or a different machine entirely. If you try to use the secure string somewhere else that's where you get this error It doesn't like it.

Here is the thing right I didn’t understand back then the implications of DPAPI but I learned the hard way. I was working in a team back then and one time I asked my colleague to run my script on his machine. And then boom error. I spent like a half an hour debugging the script because I thought I broke something in the actual automation part. This was before I realized the error had nothing to do with what I was scripting. It is a security mechanism. A feature if you will. I know I know this can be frustrating.

Here's the usual scenario. You're likely creating a secure string on one machine maybe your dev box and then trying to decrypt it on another maybe a production server or you are trying to run a script in your user context where it was created by a service account. And you think you are using the right key but the key isn’t really a key its a collection of machine and user account settings.

Let me show you what I mean in code. The usual approach that fails sometimes:

```powershell
# Bad Approach Example
$securePassword = ConvertTo-SecureString -String "MySuperSecretPassword" -AsPlainText -Force
# Later in a different context
# $secureString | ConvertFrom-SecureString # This line will often fail
```

This looks like it should work on the face of it but no guarantees. Here's the kicker though the `-AsPlainText` parameter implies you have to store that password as a string before converting it to secure string which is a big no-no if your code is being reviewed by peers. This is a security pitfall in itself.

Now the proper way is usually to use the DPAPI encryption mechanism which is what `ConvertTo-SecureString` implicitly is doing so you do not have to worry about the key you are passing. Now there are ways to encrypt data using specific keys and encryption algorithms but using the built in DPAPI mechanism is the easiest way of working with secure strings which is the main objective of your original question.

So here's how to generate a secure string and also how to use it correctly. Remember this should only be decrypted under the same user account context and same machine which created it.

```powershell
# Generate a secure string
$securePassword = ConvertTo-SecureString -String "MySuperSecretPassword" -AsPlainText -Force
$securePassword | ConvertFrom-SecureString | Out-File -FilePath "securePassword.txt" -Force

#Decrypt secure string
$securePasswordFromFile = Get-Content -Path "securePassword.txt" | ConvertTo-SecureString
$plainTextPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePasswordFromFile))
Write-Output $plainTextPassword
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePasswordFromFile))
```

This is the most simple approach but if you try to run the decrypt part on a different machine it is very likely to not work as mentioned before so keep that in mind. This is not a bug this is a security feature it protects you from other users or service accounts stealing your credentials.

If you need to move a secure string between machines the recommended approach is to use asymmetric encryption using certificates. That is usually a better approach because you do have more control.

Here is another more complex example where you are using certificates which is a good starting point.

```powershell
# Certificate encryption example
# Get the certificate
$cert = Get-ChildItem -Path "cert:\CurrentUser\My" | Where-Object {$_.Subject -like "CN=MyCertificate"}

# Example password to encrypt
$plainTextPassword = "MySuperSecretPassword"
$plainTextBytes = [System.Text.Encoding]::UTF8.GetBytes($plainTextPassword)

# Encrypt
$encryptedPasswordBytes = [System.Security.Cryptography.RSACryptoServiceProvider]::Create($cert.PrivateKey).Encrypt($plainTextBytes,$false)
$encryptedPasswordString = [Convert]::ToBase64String($encryptedPasswordBytes)
Write-Output $encryptedPasswordString

# Decrypt
$encryptedPasswordBytes = [Convert]::FromBase64String($encryptedPasswordString)
$decryptedPasswordBytes = [System.Security.Cryptography.RSACryptoServiceProvider]::Create($cert.PrivateKey).Decrypt($encryptedPasswordBytes,$false)
$decryptedPassword = [System.Text.Encoding]::UTF8.GetString($decryptedPasswordBytes)
Write-Output $decryptedPassword

```

This approach is way more involved but also way more secure and flexible. And it also gives you a lot more control.

Okay here is the thing I am sure you are thinking this is too much why not just put it in a configuration file and get over with it? Well there is this thing called secrets management where all these solutions end up being incorporated into a proper system. A proper secret management system is beyond the scope of the current question.

But there is also the question of "best practice" here and yes storing anything in plain text config files is generally frowned upon. So its good to practice the right way when you are learning new skills. This way you also get to understand how Windows works under the hood which will benefit you down the line.

As for resources look into the Windows documentation on DPAPI it's a good starting point. Also the book "Windows System Programming" by Johnson Hart is excellent it goes deep into the Windows internals and it also covers security topics. There is also a lot of good documentation around certificates in Windows if you need to understand the more complex solution I provided. I also recommend digging deep into cryptography concepts from the book "Cryptography Engineering: Design Principles and Practical Applications" by Niels Ferguson Bruce Schneier and Tadayoshi Kohno.

So yeah that's my experience with that error I hope this helps you out It's not the most straightforward thing to deal with but once you get the hang of it it makes sense. Keep at it you'll nail it.
