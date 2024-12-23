---
title: "How can PKCE OAuth2 be implemented with React and Doorkeeper?"
date: "2024-12-23"
id: "how-can-pkce-oauth2-be-implemented-with-react-and-doorkeeper"
---

Okay, let's tackle this. I've seen more than my share of authorization flows go sideways, and the complexities surrounding implementing pkce with react and doorkeeper are definitely something I've spent a good chunk of time figuring out. It's not inherently difficult, but the devil, as they say, is in the details. The key is understanding the components and how they interact, which is what I’ll break down here.

First off, for those unfamiliar, pkce (proof key for code exchange) is an extension to the authorization code grant flow within oauth 2.0. It adds a layer of security, particularly crucial in single-page applications like those built with React, where the client secret cannot be safely stored. It does this by generating a dynamic secret—a verifier—and a challenge based on it, for each authorization request. This prevents an attacker from using an intercepted authorization code without the associated verifier.

My experience with this goes back a few years, working on an internal dashboard where we decided to move away from the implicit grant due to the well-documented security concerns. We had doorkeeper as our authorization server and, well, let’s just say it was a bit of a journey to get the pkce implementation smooth and reliable with React. It wasn’t as straightforward as some tutorials made it seem. Here’s how we eventually cracked it.

The general flow involves these steps: generating a code verifier and a code challenge on the React side, using the challenge when requesting authorization from doorkeeper, receiving the authorization code, and exchanging it for access and refresh tokens, again on the React side, using the code verifier. The challenge is usually derived using the sha256 hash of the verifier, with a base64 url encoding on the result.

Let's start with the React part. We need a utility function to generate these codes. Here’s a basic example, using the 'crypto' library which you’d install via npm:

```javascript
import { createHash, randomBytes } from 'crypto';

function generateCodeChallengeAndVerifier() {
    const verifier = base64UrlEncode(randomBytes(32)); // 32 bytes of random data
    const challenge = base64UrlEncode(createHash('sha256').update(verifier).digest());
    return { verifier, challenge };
}

function base64UrlEncode(buffer) {
    return buffer.toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
}
```

This first snippet gives you the core logic. `randomBytes(32)` creates a 32-byte random buffer, which we then encode in a base64 url safe format to produce the code verifier. Then, we use sha256 on this verifier and base64 url-encode the result to get the code challenge. These two values are essential to the process.

Now, for the second part: initiating the authorization request. You will typically initiate this process when a user wants to log in, or an unauthenticated user attempts to access a secured resource, which is very common in SPAs.

```javascript
import { generateCodeChallengeAndVerifier } from './utils'; // Assuming the code from the first snippet is in utils.js

function initiateAuthorization(clientId, redirectUri, authorizationEndpoint) {
  const { verifier, challenge } = generateCodeChallengeAndVerifier();
  // Store the verifier in localStorage for later use.
  localStorage.setItem('codeVerifier', verifier);

  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: redirectUri,
    code_challenge: challenge,
    code_challenge_method: 'S256',
    scope: 'read write', // Adjust as needed
  });

  window.location.href = `${authorizationEndpoint}?${params.toString()}`;
}


//Example usage
const clientId = 'your_client_id';
const redirectUri = 'http://localhost:3000/callback';
const authorizationEndpoint = 'https://your_doorkeeper_instance.com/oauth/authorize'; // Your Doorkeeper Auth endpoint

// Example of triggering the initiation
function handleLoginClick() {
    initiateAuthorization(clientId, redirectUri, authorizationEndpoint);
}


```
In this second snippet, we get the verifier and challenge, store the verifier securely (here, using localStorage, but in production, you might consider alternative, more secure storage), then form the authorization url. Crucially, the code challenge and `code_challenge_method` are added to the request. You'll also need to include `response_type` as `code`, and potentially other scopes specific to your application requirements. The `window.location.href` then kicks off the authorization flow, which will redirect the user to the configured redirect url.

The final part involves the callback from doorkeeper, when the user returns to your react application. You will need a component or function to handle this, specifically to use the received code and exchange it for access and refresh tokens.

```javascript
// inside the callback route.
import { useState, useEffect } from 'react';

function handleCallback() {
   const [tokens, setTokens] = useState(null);

  useEffect(() => {
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');
       if (code) {
        const verifier = localStorage.getItem('codeVerifier');
        localStorage.removeItem('codeVerifier');
         fetch('https://your_doorkeeper_instance.com/oauth/token', { // Your Doorkeeper Token endpoint
           method: 'POST',
           headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
           body: new URLSearchParams({
            grant_type: 'authorization_code',
             code,
              redirect_uri: 'http://localhost:3000/callback', //Your redirect url
             client_id: 'your_client_id',
             code_verifier: verifier,
           })
         }).then(res=> res.json()).then(data=> {
             setTokens(data);
             //store the tokens securely, e.g. http only cookie or in memory, or any other secure method based on your security requirements.
             console.log("Tokens received", data);
         });

       }
  },[]);

   return (
    <div>
     { tokens ? <p>Tokens received. Check the console </p> : <p> Waiting for tokens... </p> }
    </div>
  )
}


```
This final snippet illustrates the token exchange phase. We retrieve the authorization code and the stored code verifier, then exchange them with doorkeeper to get the tokens. The tokens should then be handled in your application. In this example, i use `useState` to hold the tokens, for you to handle and implement as needed for your specific application.

One crucial aspect I learned the hard way is meticulously validating state transitions in the process. Make sure you handle errors gracefully, especially situations where the code verifier is missing, or the code exchange fails. Proper logging in your backend can be essential in diagnosing what has gone wrong when issues arise.

For further reading, I highly recommend reviewing the original oauth 2.0 specification (RFC 6749) alongside RFC 7636, which specifically outlines the PKCE extension. Understanding the foundational concepts is key to truly grasping these authentication protocols. In addition, "OAuth 2 in Action" by Justin Richer and Antonio Sanso is invaluable for a deeper dive into OAuth 2.0 complexities, and "Programming Web Security" by  Markus Jakobsson and Zulfikar Ramzan is useful for thinking about secure practices for token handling in SPA environments. You may also want to consider delving deeper into the details of the implementation of doorkeeper, and how this relates to implementing pkce and the authorization flows.

Implementing pkce oauth2 with react and doorkeeper does require a solid understanding of the security implications, but it doesn’t have to be a complex process if you approach it step-by-step. Take your time, test thoroughly at each stage, and it'll become a robust part of your authentication flow. Remember to focus on secure storage of your tokens and your client id, especially in a client-side environment.
