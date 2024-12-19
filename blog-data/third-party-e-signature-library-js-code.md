---
title: "third party e-signature library js code?"
date: "2024-12-13"
id: "third-party-e-signature-library-js-code"
---

Okay so you're asking about using a third party JavaScript library for e-signatures right Been there done that man Believe me I've sunk weeks maybe even months into this whole digital signature thing It's never as straightforward as it seems on the surface I mean you start thinking simple javascript easy breezy but then security and browser differences and different formats it gets gnarly real fast

So from what I gather you need a js library that handles electronic signatures not just like a drawing pad on the web page I’m talking about proper cryptographic signatures the kind that actually hold up in court you know the works You're not looking for some doodle thing you want something with proper backend support the type that produces verifiable digital fingerprints

I remember way back when I tried rolling my own thing before I knew better Oh boy was that a bad idea It was in like 2015 I was using vanilla js and some random crypto library to hash things I thought I was hot stuff I even implemented a weird little handshake to simulate signing Oh the hubris The first real auditor came along laughed at the whole setup the security was a joke and I realized that security is hard man that’s why we pay the experts

Now that I've had my fair share of trial and error I can tell you for sure that you should not try to roll your own unless you have a dedicated cryptography team That path leads to madness security holes and existential dread

So let's dive into it I'll give you a few pointers based on what I've learned over the years

First big thing is understand what you're actually signing Is it a PDF Is it some JSON data Is it a blob This matters because you need to format it properly before it goes into the signing process You need to make sure the hashing function is the appropriate one and also make sure that the document that the user has seen the one that is displayed is the same one that is being signed It's really common to have discrepancies and that opens up all kinds of problems later on

So let's say you're working with PDFs a very common use case Then you need a library that can handle PDF manipulation along with the signature part You might want to explore libraries that wrap around some backend system or are fully serverless and have a client side component

Here’s a very simplified example of using a fictional library just for illustration sake this is not production-ready code keep in mind

```javascript
// Assume 'eSignatureLib' is a placeholder for a real library
async function signPdf(pdfBlob, privateKey) {
    try {
      // First a hash is computed to be signed
      const hash = await eSignatureLib.computeHash(pdfBlob, 'SHA-256')
       //then we sign with private key
       const signature = await eSignatureLib.sign(hash, privateKey)

       // Then add it to the pdf
      const signedPdfBlob = await eSignatureLib.embedSignature(pdfBlob, signature,hash)
      return signedPdfBlob
    } catch (error) {
      console.error("Error signing PDF:", error);
      throw error;
    }
  }
// Example usage
async function exampleSignProcess(pdfFile , privateKey)
{
   const fileReader = new FileReader()
   fileReader.onload = async (e)=> {
      const arrayBuffer = e.target.result;
       const pdfBlob = new Blob([arrayBuffer], { type: "application/pdf" });
       const signedBlob = await signPdf(pdfBlob,privateKey)
      // Do what you want with the signed blob
       console.log("pdf was signed")
   }
   fileReader.readAsArrayBuffer(pdfFile)

}

// Now you could call it like so
// exampleSignProcess(document.getElementById("pdfInput").files[0] , privateKey )
```

That’s a very simplistic example but it hits the key points You need to first hash the document and sign the hash then embed that signature in the document it self Of course there are different standards you should follow such as PAdES for PDF documents or XAdES for XML documents so do your homework before implementing it also I did not added all security parameters and all edge cases for the sake of conciseness

Let's move on let’s say you're working with JSON data a typical REST API use case it can be also simplified like so

```javascript
async function signJson(jsonData, privateKey) {
  try {
    // Stringify the JSON data first
    const dataString = JSON.stringify(jsonData);
    //then hash the string
    const hash = await eSignatureLib.computeHash(dataString, 'SHA-256');
    //and sign the hash
    const signature = await eSignatureLib.sign(hash, privateKey);
      // Add it to the json
    const signedData = { ...jsonData, signature: signature };
    return signedData;
  } catch (error) {
    console.error("Error signing JSON:", error);
    throw error;
  }
}

// Example usage
async function exampleJsonSigning(jsonPayload,privateKey){
   const signedJson = await signJson(jsonPayload, privateKey)
  // Do stuff with the signed json payload
   console.log("json was signed")
}

// exampleJsonSigning({name:"John Doe",age:30} , privateKey)

```

Again simplified but it shows the pattern of hashing and then signing and then attaching that to the document That's the core of what you need to do

Now here’s a bit of a curveball handling client-side private keys is a big no no never expose your private key to client side code You'll hear that a lot on the internet because that is the golden rule private keys should never leave the server that you trust

So how do we do this securely? Typically you generate the key on the server and use some API to request a signature You send the hash to the server and then you sign it server side It's a roundtrip but it’s the only safe way unless you are dealing with ephemeral keys for temporary signing sessions

So here’s an example of a client side call that requests signing from a server in a simple manner

```javascript
async function requestSignature(dataToSign) {
    try {
      const response = await fetch('/api/sign', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: dataToSign }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const signedData = await response.json();
      return signedData;
    } catch (error) {
      console.error("Error requesting signature:", error);
      throw error;
    }
  }

  // Example
  // const dataToSign = {message : "hello world"}
 //  const signatureResponse = requestSignature(dataToSign)

```

On the server side you would receive the hash sign it and send it back to the client Now please don't use that `fetch` directly in production consider a more robust API client I use `axios` for these kind of tasks

So what about libraries and resources? Don't go searching for "e-signature-library.js" on google you need to be more specific You should check papers and the books of standard authorities in the field of cryptography and digital security like Bruce Schneier or read the documentation of standard such as XAdES, PAdES or CAdES you'll want to look at libraries that use these standards. If you are interested in the actual math behind it look for research papers about elliptic curve cryptography and digital signature algorithms (like ECDSA) and its implementations

Oh and if you find some open source libraries that have been updated recently with proper community support that's even better It also helps to check if your language of choice has a good crypto library I’ve had great success using python's `cryptography` library on the backend for example It does a lot of heavy lifting so I don't have to think about all the low level details

And always remember even with the best library you need to do some security considerations yourself Like always use HTTPS everywhere and for the love of everything don't store private keys in git or hardcode them in your code that’s what configuration files are for

I once spent 3 days debugging an issue only to realize that a colleague had committed a private key to git We all make mistakes sometimes It was a learning experience to be honest but don’t do that okay

Ok I'm done now this should get you started now go forth and implement all of these new concepts that I just gave you and may your digital signature verification be always successful

Hope that helps and happy coding
