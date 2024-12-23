---
title: "how to get rid of noisy warning no suitable checks publisher found?"
date: "2024-12-13"
id: "how-to-get-rid-of-noisy-warning-no-suitable-checks-publisher-found"
---

so you're wrestling with that "no suitable checks publisher found" warning right Been there absolutely done that a few times myself its like a persistent mosquito buzzing in your code compilation process drives you nuts

First off let's be clear This warning is a byproduct of code signing and security protocols it pops up when your compiler or development environment can't verify the legitimacy or origin of certain components or libraries you're using It's not always a showstopper but ignoring it is definitely playing Russian roulette with your app's security and potentially user trust

I've seen this crop up in a bunch of contexts Usually it's when you are working with unsigned binaries or libraries that have not been properly code signed or if your development environment doesn't have the correct certificates installed I've seen it with random DLLs i pulled from some random github repo and when trying to compile a C++ project with a custom built library not signed with a trusted certificate I still remember when i tried to install a random python library and that python library was not signed or was using a broken certificate ah bad memories

Ok lets go through possible fixes lets talk about that as a starting point

**The "Easy" but Not Recommended Way: Bypassing the Checks**

Look I know you're probably itching to make the warning just vanish Sometimes you might think "oh just ignore it its fine" but no please for the love of God don't You *can* suppress these warnings using compiler flags or environment variables Its usually not the answer but i have been guilty of that in my past but learned the hard way to never do that again

Example for gcc like C++ compilers something like this might work
```cpp
#pragma warning (disable: 4190) // This will suppress warning code 4190
//Or something like this if using clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-missing-checks-publisher"
```

Or if we go to the python world you can try to silence the warning output while running a program or while installing a library
```python
import warnings
warnings.filterwarnings("ignore", message="No suitable checks publisher found.*")
# Or perhaps you can try
# python -W ignore::UserWarning your_script.py
```

Please resist this urge as much as you can please I’ve seen it cause more headaches down the line trust me. You're essentially telling your system "just trust everything" which is not a good idea especially in this day and age where security is critical think about supply chain attacks

**The Correct Way Code Signing and Publisher Verification**

Now lets talk about the solution for this pesky error
Ideally you should deal with this by ensuring all the libraries and executables you use are properly code signed Code signing involves a digital signature that verifies the identity of the publisher or the company that created the code The OS then uses this signature to make sure the code is legitimate and hasn't been tampered with

If you are generating your own code which is usually the case when dealing with libraries or internal tools you'll need a code signing certificate These certificates are issued by trusted certificate authorities CAs which are the gatekeepers of digital trust
You usually purchase these certificates from commercial providers like Digicert or Sectigo and then use the tools of your platform to sign your code

On Windows for example you use the `signtool.exe` command line utility to sign executables and DLLs using the certificate and a private key you will use the certificate to sign your library or executable in order to not produce such a warning

Here is a basic example of that

```cmd
signtool sign /f your_certificate.pfx /p your_password /t http://timestamp.digicert.com your_executable.exe
```

or if you are using a mac you might have to use `codesign`

```bash
codesign -s "Your Developer ID" --timestamp your_executable
```

The key part is `/f your_certificate.pfx` in the windows example that is the path to your certificate file and `/p your_password` is the password for the certificate
And for macOS you will need a apple developer id to be able to sign

The timestamp server is important to guarantee the signature will remain valid even if the certificate expires the code will still be considered signed and valid if the signature occurred during the certificate period.

**Other Considerations to Eliminate the Warning**

Now that we covered the most basic scenario of the error i would like to talk about cases that are more complex such as dealing with external libraries

Sometimes the "no suitable checks publisher found" arises with third party libraries That's when you will need to use dependency management tools or build systems with signing capabilities
For maven for example you can enable signature validation in pom files by doing something like this
```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-dependency-plugin</artifactId>
  <version>3.6.0</version>
  <configuration>
    <failOnWarning>true</failOnWarning>
    <strict>true</strict>
  </configuration>
  <executions>
     <execution>
        <id>verify-signatures</id>
        <goals>
          <goal>verify-signatures</goal>
        </goals>
        <phase>validate</phase>
    </execution>
  </executions>
</plugin>
```
 This configuration forces Maven to fail if signatures are missing or invalid for dependencies and it is very useful to use that while setting up CI environment to not use unsigned packages or binaries
Or on a more low level scenario with cmake for example you need to configure a set of tools to check and sign the code on your build process

Also when dealing with windows systems sometimes your computer doesn't have the proper certificate authority installed so you need to explicitly install them for your computer to be able to verify the code

And hey i know it is annoying to fix that problem i understand you its like having to learn that the compiler is also your boss now and you must please it but it's part of making sure your software is safe and your users trust it no matter if it is a internal library or a simple application

**Resources**

Instead of giving you random links I’d recommend diving into these resources for a deeper understanding

*   **"Cryptography Engineering" by Niels Ferguson and Bruce Schneier**: It's the bible of practical cryptography This will give you a strong understanding of how certificates work the security model and the code signing architecture
*   **Official documentation of your operating system on code signing**: Like Microsoft's documentations on `signtool.exe` or Apple's documentation for `codesign` the official documentation gives you all the specific details about the parameters required and best practices
*  **Security blogs from reputed companies like Google Project Zero and Microsoft security**: These blogs will keep you up to date on vulnerabilities and the most used attack vectors in our modern world this will give you a better understanding about the importance of code signing and how it protects you from malicious actors

**Final Thoughts**
 The "no suitable checks publisher found" warning might seem like a minor annoyance but is a security red flag If you ignore it you risk introducing vulnerabilities into your system or application Instead embrace code signing best practices and ensure that all the libraries and executable you use are properly signed

If you need more help please feel free to paste your specific use case with more details like which build system you are using which programming language or if you are using your OS specific tool i will do my best to assist you remember security is everyone's responsibility in this crazy software world but hey dont be like my old coworker who kept ignoring the warnings and said "it works on my machine" and then he got hacked you know what they say about those who live in glass houses dont throw stones... or debug their code in production

Good luck my friend
