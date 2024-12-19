---
title: "error reading server preface http2 connection?"
date: "2024-12-13"
id: "error-reading-server-preface-http2-connection"
---

Okay so you're hitting a "error reading server preface http2 connection" right Been there done that got the t-shirt seriously This error is a classic head scratcher especially when you think everything *should* be working

Let's break it down from my experience and what I've seen others wrestle with It almost always boils down to a fundamental mismatch or a hiccup in the initial handshake process between your client and the server when attempting an HTTP/2 connection The preface the error is bitching about is that specific sequence of bytes that says "hey we're talking HTTP/2"

First off this isn't a data-corruption-in-transmission kinda error think more like a "I can't understand you because you're speaking Klingon while I'm expecting English" type of problem That's the vibe we're chasing here

I remember back in my early days I spent a week tearing my hair out over this on a custom embedded system Turns out the damn thing wasn't even configured to use HTTP/2 in the first place I was force-setting the protocol on the client side without checking server support A total facepalm moment believe me

So the first thing to absolutely triple-check is that both your client and your server are actually configured and enabled for HTTP/2 support It's not magic It's just configuration If either side isn't set up correctly that preface will never be valid and you're dead in the water The server needs to be explicitly configured to listen for HTTP/2 connections and your client needs to be set up to send the correct preface

Now the error message is often pretty vague so let's dig a little deeper into common pitfalls we see with this issue The easiest and most common thing is probably just plain old protocol mismatch Make sure your client isn't trying to connect with HTTP/2 when the server only supports HTTP/1.1 and vice versa This is easy to fix just check your server configuration and client connection parameters

Let's do some hypothetical examples cause I think they might help

**Example 1: Node.js client using `node-fetch`**

```javascript
const fetch = require('node-fetch');

async function makeHttp2Request() {
    try {
        const response = await fetch('https://example.com', {
            // Note: http2 might not work out of the box you need an agent
            // see `https://nodejs.org/api/http2.html` for http2 specific setup
            // and https://github.com/nodejs/undici
            // if you need to explicitly use http/1.1 force it
            // see specific documentation for your fetch library
             // agent: new http.Agent() //force http/1.1
             
        });
        if (response.ok) {
            console.log('Response received:', await response.text());
        } else {
            console.error('Request failed with status:', response.status);
        }
    } catch (error) {
        console.error('Error during request:', error.message);
    }
}

makeHttp2Request();
```

In this example if the server is not set up to handle HTTP/2 you are likely going to have this specific error You may need to configure your HTTP Agent or you may need to choose an entirely different library to make the request

**Example 2: Python using `requests`**

```python
import requests

def make_http2_request():
    try:
        # requests does not support http2 natively so this will fail as a request
        # but it can help you make a simple request to test the server
        # the simplest way to test the server is with curl
        # e.g: `curl -v --http2 https://your-server-address`
        response = requests.get("https://example.com", verify=False ) #verify=False is only for testing self-signed certificates do not do this for production
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        print("Response received:", response.text)
    except requests.exceptions.RequestException as e:
        print("Error during request:", e)

make_http2_request()
```
Remember the Python `requests` library doesn't natively handle HTTP/2 so trying this will probably still result in an error You can see what happens if you try it and then you can start considering if the problem is on the server side or the client side

**Example 3: Go client using `net/http`**

```go
package main

import (
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net/http"

	"golang.org/x/net/http2"
)

func main() {
	client := &http.Client{
		Transport: &http2.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true, // Insecure only for test cases
			},
		},
	}


	resp, err := client.Get("https://example.com")
	if err != nil {
		log.Fatal("Request failed: ", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatal("Error reading response: ", err)
	}

	fmt.Println("Response:\n", string(body))
}
```
In Go it is often more specific because `net/http` doesn't support HTTP/2 out of the box You need to specifically add `golang.org/x/net/http2` to make http2 requests like in the example provided This also highlights the fact that the server needs to support the HTTP/2 protocol or this will fail with similar errors we are discussing

Alright another common gotcha is TLS configuration with HTTP/2 The spec requires TLS for HTTP/2 so if you're not using TLS or if your TLS configuration is incorrect you'll see a similar error The client needs to be able to negotiate a compatible TLS version and cipher suite and the server also needs to support that as well You want to check the TLS versions of both server and client they have to be compatible

And let's not forget the importance of SNI Server Name Indication especially if you're hosting multiple domains on the same IP address If the SNI is not sent or is sent incorrectly the server might not know what certificate to present which can cause the preface to be invalid That can lead to connection problems including that vague error you are seeing

Ok so we've covered some main issues Now let's talk about debugging this mess if you still haven't found the problem Curl with the `-v` flag is your best friend for this  That gives you super detailed information about the TLS handshake and HTTP/2 negotiation The data is very low level and it is amazing to figure out the problem If curl works but other clients do not then you know it is client specific and not the server

I once spent a whole day just staring at tcpdumps trying to decipher why the client was behaving like a moron Turns out the client was trying to use an older TLS version that was disabled on the server it was truly a pain in the butt that day But it shows you that network capturing tools like tcpdump wireshark can be incredibly valuable too

Now I know you want real resources and I am not a fan of the "just google it" approach although sometimes that's how I solve things Let me suggest some good reads if you are seriously pursuing this topic If you want to dig into the HTTP/2 protocol specifics I'd recommend reading the RFC 7540 and RFC 7541 those are the actual standards that define the HTTP/2 protocol and are a great resource I do not expect you to read everything but it can be very helpful to understand the specific parts of the connection establishment

And if you want a more practical view check out "High Performance Browser Networking" by Ilya Grigorik it's a great book that covers HTTP/2 and related network protocols extensively and it has helped me several times with these issues It gives you all the best practices and deep insights that are very useful And the last thing I would recommend is to check the nodejs and other library specific HTTP/2 documentation because they can often have specific issues on their implementation which is not exactly standard

Okay last bit a little joke I always tell myself when facing a similar issue when trying to debug server issues "It's always DNS... except when it's not" which is basically true most of the time but it also highlights how complex this thing can be

Anyways hopefully this wall of text helps you out If you have any specific parts of the configuration you are unsure about post it and I'll take another shot at it

Good luck
