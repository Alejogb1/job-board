---
title: "Why does the React fetch API return a 302 status code and invalid cookie?"
date: "2024-12-23"
id: "why-does-the-react-fetch-api-return-a-302-status-code-and-invalid-cookie"
---

Alright, let's unpack this rather thorny issue of React fetch requests encountering 302 redirects and invalid cookies. I've seen this crop up a few times over the years, and it’s often a confluence of subtle configuration problems, not necessarily a straightforward bug in React itself.

The core of the issue typically stems from a mismatch between the expected client-side request behavior and what the server is actually doing. A 302 status code, by its very definition, indicates a temporary redirect. The server is essentially telling the client, "the resource you're looking for is now located at this other address." Coupled with an apparent invalid cookie issue, we're usually dealing with server-side handling of authentication or session management not playing well with how `fetch` operates.

First, consider that `fetch` does not, by default, follow redirects if the request method isn't GET or HEAD, or if you have manually set the `redirect` option to `manual`. If the original fetch request is a POST, PUT, DELETE, or PATCH, and the server responds with a 302, the `fetch` api doesn't automatically resend the *body* of the request to the redirected url. This can create problems, particularly with authentication-heavy api endpoints.

The "invalid cookie" part is even more nuanced. Cookies are generally managed via the `set-cookie` header in http responses. When you fetch to a server, the server may want to set a cookie to track session state. The problem arises if the redirect leads to a domain or subdomain that’s different from where the original request was made, or if there are subtle differences in the path of the requested url. The browser itself will not automatically send or overwrite cookies across domains by default without special configuration, for security reasons. This means that a cookie set at, say, `api.example.com/auth` may not be considered valid at `app.example.com` after a redirect because of differing domain attributes, or potentially incorrect `path` attributes, set on the cookie itself.

Furthermore, server-side logic that uses a 302 as part of a session creation or authentication routine may not be set up to handle the nuances of client-side javascript applications doing the fetching. For instance, if a user authenticates, the server might try to set a session cookie and *then* redirect the user to the final resource. If the cookie setting or path attribute on the cookie is misconfigured, or if the redirect target doesn't anticipate the initial request originating from a client application that might not automatically follow redirects or forward the cookie, this problem becomes apparent. Let’s take a look at some scenarios and corresponding fixes.

**Scenario 1: CORS Issues and Redirects**

Suppose a backend on `api.example.com` attempts to set a cookie and then redirects to `app.example.com`. Here's a scenario where `fetch` would likely struggle:

```javascript
// Example React Fetch Request with issues

const handleFetch = async () => {
  try {
    const response = await fetch('https://api.example.com/login', {
       method: 'POST',
       headers: {
          'Content-Type': 'application/json'
       },
       body: JSON.stringify({username:'testuser', password:'testpassword'})
     });

    if(response.ok) {
       const data = await response.json()
       console.log("success", data)
    } else {
       console.error("fetch error", response.status)
    }


  } catch(error) {
    console.error("Network Error", error);
  }
}

//Backend server response
// http 302 to https://app.example.com, includes set-cookie header, but cookie is not configured to be sent cross domain

```
In this case, you’d see a 302, but the cookie isn't set, either because the cookie's `domain` attribute isn't specified properly for the subdomain/domain transition, or the cross-origin request fails for CORS issues and security measures on the browser. The fix here would involve configuring the cookie to have a broader `domain` and potentially setting the `sameSite` attribute as needed on the cookie server side, as well as ensuring CORS headers are properly configured on the api server. You might need the api server to set `Access-Control-Allow-Origin` to the origins that will send requests, and `Access-Control-Allow-Credentials` to `true` to allow the sending and receiving of cookies cross-origin.

**Scenario 2: Manual Redirect Handling & Cookie Storage**

Sometimes the best option is to manually handle redirects when they’re not standard. Let’s demonstrate:

```javascript
// React Fetch Request with Manual Redirect Handling
const handleFetchWithManualRedirect = async () => {
  try {
     let currentUrl = 'https://api.example.com/login';
     let response;
     let redirectCount = 0;

    while(true) {

        response = await fetch(currentUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
             },
             body: JSON.stringify({username:'testuser', password:'testpassword'}),
             redirect: 'manual'
            });

            if (response.status === 302) {
                const redirectUrl = response.headers.get('location')
                console.log('Redirecting to:', redirectUrl)
                currentUrl = redirectUrl
                redirectCount++;
                if (redirectCount > 5) {
                    throw new Error('Too many redirects');
                }
            } else {
                if(response.ok){
                   const data = await response.json()
                   console.log('Successful fetch', data)
                }else{
                   console.error("fetch error", response.status)
                }

                break;
            }
    }



    //Process cookie here manually if it was sent

     if(response.headers.get('set-cookie')) {
        const cookie = response.headers.get('set-cookie')
        document.cookie = cookie;
        console.log('cookie set manually', cookie)

     }

  } catch(error) {
    console.error("Network error:", error)
  }

}
```
Here, by setting `redirect: 'manual'`, we explicitly instruct `fetch` not to follow the redirect automatically. We then capture the `location` header, make sure we’re not looping endlessly, and re-fetch the new url. In addition, the code demonstrates manually extracting the `set-cookie` header from the response and setting it on `document.cookie`, which allows the browser to maintain cookie state across redirects.

**Scenario 3: Server-side Session Handling**

Sometimes, the problem is on the server. Let’s say the server side logic expects the client-side to automatically resend the body on a 302 from a non GET/HEAD. This is a common misconfiguration, as per HTTP spec, and is not standard behavior.

```javascript

 // React Fetch request expecting auto resend of body for 302s
const handleFetchIncorrectServer = async () => {
    try{
      const response = await fetch("https://api.example.com/protected",{
          method: "POST",
          headers: {
             'Content-Type': 'application/json'
          },
          body: JSON.stringify({data: "some data"})
      })

        if(response.ok) {
         const data = await response.json()
         console.log("fetch success", data)
        }else{
            console.error("fetch error", response.status)
        }

    } catch(error) {
        console.error('network error', error)
    }
}

//Backend server response
// http 302 to https://api.example.com/protected-redirect. Server expects the client to POST body to redirect url, and will fail if not present.

```

Here, a client-side `fetch` api will get a 302, will not forward the post data, and thus will get another error or fail state from the redirect URL. The solution is to update the *server-side* logic to accept GET requests for redirects, or to forward the body of the request to the redirect url, or to avoid the redirect and provide data directly in the first response.

**Resources for Further Reading**

To get a firmer grasp of these concepts, I'd suggest diving into:

*   **RFC 2616 (HTTP/1.1 Specification):** While superseded by newer RFCs, it lays the foundational rules for http status codes and headers, offering excellent historical context.
*   **"HTTP: The Definitive Guide" by David Gourley and Brian Totty:** This book is a comprehensive resource on HTTP, covering all aspects from request formats to server configurations. It provides in-depth explanations of status codes, cookies, and related topics.
*   **Mozilla Developer Network (MDN) web docs:** Search for `fetch api`, `http headers`, `cookies`, and `CORS` on MDN. It’s an invaluable resource for practical, up-to-date information.

In summary, encountering a 302 and an "invalid cookie" with the React fetch API is usually a sign of misaligned server configurations, CORS issues, incorrect cookie attributes, or a mismatch in how client-side `fetch` handles redirects versus server-side expectations. By carefully reviewing your server responses, configuring your cookies correctly, and understanding how `fetch` behaves, you should be able to tackle these situations effectively. Remember, careful monitoring of the network requests, especially the headers, is key. I've found that using browser dev tools or a proxy like Charles or Fiddler, will always yield the most relevant information to pinpoint the exact problem.
