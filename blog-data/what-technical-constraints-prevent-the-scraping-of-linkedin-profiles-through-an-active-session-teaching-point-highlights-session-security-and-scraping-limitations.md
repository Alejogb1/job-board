---
title: "What technical constraints prevent the scraping of LinkedIn profiles through an active session? (Teaching point: Highlights session security and scraping limitations.)"
date: "2024-12-12"
id: "what-technical-constraints-prevent-the-scraping-of-linkedin-profiles-through-an-active-session-teaching-point-highlights-session-security-and-scraping-limitations"
---

my second is: if i were to create a website to get around the need for direct access to data from facebook apis what architecture could i use ? (Teaching point: Discusses API proxies and data management)
and the third is: if i wanted to create a service that detects whether a pdf document has been tampered with what technologies would you suggest and how would they work? (Teaching point: Focuses on digital signatures and hash functions)

ok so linkedin scraping with an active session yeah that's a can of worms really it's less about a single hurdle and more about a carefully constructed wall they put up it's not just a quick block they've layered on protections like a cake of anti-scraping measures the core is all about maintaining user session integrity and preventing automated abuse

first off they use session tokens extensively these aren't your grandma's static cookies each time you log in they generate a unique token tied to your browser and device it's not just the cookie itself it's that plus a bunch of other browser fingerprinting stuff like user agent screen resolution and even subtle details about your javascript environment linkedin then uses this data to build a profile of your session if it detects discrepancies say a bunch of requests coming from a script not a real browser or if the request pattern feels off like super human fast it'll invalidate the session or flag it for review which is usually some kind of captcha or outright ban

they employ rate limiting this is a classic defense and it's not a simple per minute limit either it dynamically adjusts to the user's behaviour if you start making a crazy number of profile requests in a short time they will start returning error codes or slow down responses considerably its like a faucet that gets restricted when you try to draw too much water at once the system uses intelligent algorithms to distinguish between genuine user interaction and bot activity this can include analysing the intervals between requests the consistency of scrolling patterns and even mouse movements all these things they are looking at

also linkedin's backend does extensive server side checks think of it as a gatekeeper that looks at every request if they spot any header inconsistencies unusual referers or data patterns they get suspicious it's not enough just to look like a browser you also have to behave like one and this behaviour is very fine tuned they even look at things like the http methods you are using get vs post and if you are not using them correctly that's a red flag so you need to be mimicing their expected usage

javascript is used to inject dynamic content and to make sure it all runs in the right context if you try to bypass the javascript rendering process and just pull data via raw html it won't work you get the html shell but a lot of the data is dynamically loaded using javascript and you need to execute that javascript to actually get all the data the same way a real browser would and they often change up their javascript code so if your scraper is built on old code then its just gonna break and that means ongoing maintenance

that's not all they can also implement more complex things like bot detection services they work by placing hidden markers on the page sometimes javascript based and if your scraper doesn't interact with these markers as a human would then its again a tell tale sign of it being a bot these markers are often very subtle and change over time making it hard for scrapers to adapt and that goes beyond just the javascript it also includes looking at how the browser is rendered they can analyse how your javascript engine renders the page and they can distinguish between different browsers and different javascript runtimes to flag automated agents

in short linkedin has a multi layered system in place each layer works to prevent and identify automated scraping they're constantly evolving their techniques so it's a continuous cat and mouse game between them and scrapers it's not that they can't be scraped its just very very hard to do it consistently without getting blocked and requires significant engineering effort and maintenance

moving on to your facebook api bypass idea you could use a system that basically acts as an api proxy and that decouples your direct access to facebook you want your application or website to only interact with your own api you build a dedicated backend server which in turn interacts with facebook's apis here's the basic idea

your application frontend it's going to talk only to this intermediary server your own backend never directly to facebook this server is going to authenticate with the facebook api using an application token it's like a key to enter facebook's gate on your behalf the server will receive requests from your application like get user information or post to wall and will translate those requests into facebook api calls and when facebook responds the server will receive the data process and format it in a way that is expected by your application this middle layer is an api proxy it hides facebook's specific api details from your application and gives you a single place to manage all interactions

this server can maintain a persistent connection with facebook this can be very helpful for handling things like real time updates or if you need to do batch processing it saves the overhead of reconnecting with facebook each time a request is made the key here is that your application never sees how you are getting the data it just receives the data it expects from the intermediary server this is also where you can add rate limiting or handle different access credentials you can fine tune access permissions on your proxy side without it impacting the other parts of your system

to give you an idea of how it might look consider a node js example

```javascript
// simple nodejs express server
const express = require('express');
const axios = require('axios');

const app = express();
const port = 3000;

app.get('/api/user/:id', async (req, res) => {
  const userId = req.params.id;
  try {
    const response = await axios.get(
        `https://graph.facebook.com/v12.0/${userId}`, {
        headers: {
          Authorization: `Bearer YOUR_FACEBOOK_APP_TOKEN`
        }
      }
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching data' });
  }
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
```
this is just a barebones example but it illustrates the key concept your frontend makes a get request to `/api/user/:id` it does not know that this server is internally using the facebook graph api this server is managing facebook's request details

you could also use a cloud solution like amazon api gateway or google cloud endpoints these provide a managed layer for api management and you don't have to manage the backend servers directly they can handle things like authentication rate limiting and other api management functions you focus on the custom logic of your api and these services manage the other parts

another crucial thing to think about is caching the idea is that responses from facebook are saved for some period of time in a cache so if same request comes in again later then you dont have to call facebook again you return the result from cache this will reduce load on facebook and also improve response time for your app you could use redis or memcached to implement caching it is very useful if you have popular api calls that are made again and again

also dont forget proper error handling and retry logic because facebook api might fail intermittently in this case you can configure retries on the intermediary server and gracefully handle those errors and notify your application this makes it more stable

lastly you should design your server to handle rate limits properly facebook api comes with its own rate limits and you should respect them or else facebook might block your application so you would have to implement some form of throttling on your end to manage this and not exceed facebook's limits

for more on api design i would suggest looking into the book building microservices by sam newman its a great resource for understanding how to break down systems into smaller more manageable components which is what we did with this server architecture

now for your pdf tampering detection the core idea is to create a digital fingerprint for the document if anything changes in the pdf the fingerprint should change significantly for this you can use hash functions and digital signatures think of them as two different layers of protection

the hash function is basically a one way mathematical function it takes any input data in this case your pdf document and produces a fixed size string of characters its also called checksum now here is the cool part even a tiny change in the pdf will completely change this checksum think of it as a very sensitive fingerprint generator the idea here is that you can generate hash before the pdf goes to whoever you are sending it to and when it comes back you can regenerate the same hash if the hash values match it is very very likely that the content has not changed

a common hash function used is sha256 it's a secure hash function and you can easily compute it using any modern programming language for example python
```python
import hashlib

def calculate_sha256_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# Example Usage
pdf_file_path = 'document.pdf'
hash_value = calculate_sha256_hash(pdf_file_path)
print(f"sha256 hash of the pdf: {hash_value}")
```

this python example reads the pdf in chunks computes the sha256 hash and returns it if the hash value changes then you know the pdf has been changed

now digital signatures this adds another layer of security beyond just hash values digital signatures use cryptographic key pairs a private key used to sign the document and a corresponding public key to verify the signature the idea here is that you as the creator of pdf will sign the document with your private key and anyone who receives it can verify the signature using your public key if the signature is valid then it means that this doc was created by your private key and it wasn't changed by another person. it adds another layer of security.

when creating a digital signature first you compute the hash of pdf using a hash function like sha256 then that hash is encrypted using the private key of the document author the encrypted hash is called a digital signature and is appended to the pdf document so every pdf will have the signature alongside the contents. when verifying the signature you first get the signature from the pdf and decrypt using the public key of the document author you have to make sure that the public key really belongs to the author through trusted mechanisms and that will give you the hash value then at the same time you compute the hash of the pdf document if both the decrypted hash and the calculated hash match then it confirms both the origin and integrity of the document

digital signatures typically use algorithms like rsa or ecc and there are various libraries that support digital signatures in many different languages like python c sharp java

```java
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.*;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;
import org.bouncycastle.util.io.pem.PemReader;
import java.io.FileReader;

public class PdfSigner {

    public static String signPdf(File pdfFile, File privateKeyFile) throws GeneralSecurityException, IOException {

        // Load Private Key
        PrivateKey privateKey = loadPrivateKey(privateKeyFile);

        // Create Signature
        Signature signature = Signature.getInstance("SHA256withRSA");
        signature.initSign(privateKey);

        // Read the pdf content and update it with the signature algorithm
        FileInputStream fis = new FileInputStream(pdfFile);
        byte[] buffer = new byte[1024];
        int len;
        while ((len = fis.read(buffer)) != -1) {
            signature.update(buffer, 0, len);
        }
        fis.close();

        byte[] digitalSignature = signature.sign();
        return Base64.getEncoder().encodeToString(digitalSignature);
    }

    private static PrivateKey loadPrivateKey(File privateKeyFile) throws IOException, InvalidKeySpecException, NoSuchAlgorithmException {
        try (PemReader pemReader = new PemReader(new FileReader(privateKeyFile))) {
           byte[] privateKeyBytes = pemReader.readPemObject().getContent();
            PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(privateKeyBytes);
           KeyFactory keyFactory = KeyFactory.getInstance("RSA");
            return keyFactory.generatePrivate(keySpec);
       }
    }

    public static void main(String[] args) {
        try {
            File pdfFile = new File("document.pdf");
            File privateKeyFile = new File("private.pem");

            String signature = signPdf(pdfFile, privateKeyFile);
            System.out.println("Digital Signature: " + signature);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

this java example is little more involved because it uses bouncy castle libraries for better security which is often needed when dealing with cryptographic operations it will read the private key and the pdf file and sign the pdf content this only covers signing to verify the signature you will need additional code to retrieve the public key and verify the signature but its all very similar to the signing process

to secure digital signatures you really should use hardware security modules they are special hardware devices that store cryptographic keys and do signing and verification operation within the secure boundaries they are designed to be tamper resistant and provide strong level of protection if you are dealing with highly sensitive documents

for in depth reading i would recommend understanding cryptography by christof paar and jan pelzl it provides a good foundation for the cryptography that is used in digital signatures and hashing

i've tried to keep it techy simple no over complicating jargon hope its what you needed
