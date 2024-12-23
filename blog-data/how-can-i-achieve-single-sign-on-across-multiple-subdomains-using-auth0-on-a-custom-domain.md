---
title: "How can I achieve single sign-on across multiple subdomains using Auth0 on a custom domain?"
date: "2024-12-23"
id: "how-can-i-achieve-single-sign-on-across-multiple-subdomains-using-auth0-on-a-custom-domain"
---

Let’s tackle single sign-on across subdomains with Auth0 on a custom domain. It's a situation I’ve personally navigated multiple times, and while conceptually straightforward, the devil, as they say, is in the implementation details. We’re not talking theoretical here; this is based on battles fought and lessons learned in the trenches of real-world projects. It's not insurmountable, but there are nuances to respect for a seamless user experience.

Essentially, the core of achieving single sign-on across subdomains rests on establishing a shared authentication context. Auth0, thankfully, makes this achievable primarily through cookie management and proper configuration. The challenge isn’t so much with Auth0 itself, but rather with ensuring the cookies used for authentication are visible and accessible across all relevant subdomains. The primary hurdle often stems from browser security policies regarding cookie scoping.

Here’s the crux of the issue: by default, cookies are scoped to the domain that sets them. This means a cookie set on `app.mydomain.com` is not automatically visible to `api.mydomain.com` or `blog.mydomain.com`. To enable SSO, we need to elevate the cookie scope so all these subdomains can share the authentication information. This can be done by setting the cookie’s domain attribute correctly to cover the parent domain – in our case, `mydomain.com`.

Let's break down the steps and considerations, and I'll provide some illustrative code snippets. We are assuming you have Auth0 already configured with a custom domain.

**Step 1: Configuring Auth0 for Shared Domain Cookies**

Within Auth0, you’ll need to confirm your custom domain is configured correctly, often done within the custom domain section of the Auth0 dashboard. Importantly, you need to ensure the cookie settings in your Auth0 tenant are set to allow sharing across subdomains. This isn't typically a setting exposed directly in the user interface but is configured during domain setup (or sometimes via support). You might need to check their documentation on custom domains and cookie attributes for the latest approach, but fundamentally, it's about informing Auth0's auth mechanisms to set cookies at the domain level.

**Step 2: Implementing Shared Cookie Logic at the Application Level**

Next, we need to ensure your applications respect these shared cookies. This usually involves configuration in the application’s login handler or the relevant middleware. If your applications utilize the `auth0-spa-js` library, the default cookie handling is typically enough once configured in Auth0 but the configuration is often overlooked and can present tricky symptoms. If working with backend applications, ensure you are setting the cookie's domain attribute to `.mydomain.com` rather than to the subdomain explicitly. This is key to allowing all subdomains to have visibility.

Here’s an example in Javascript showing a simplified cookie setting mechanism, commonly used within node-based backends when utilizing a library that requires manually setting the cookie. Note: This snippet isn't Auth0 specific; it's an illustrative example of setting the correct domain attribute for cookie sharing:

```javascript
// Example Node.js backend cookie setting
const setAuthCookie = (res, token) => {
  res.cookie('auth_token', token, {
    httpOnly: true,
    domain: '.mydomain.com', // Setting to parent domain
    secure: true,            // Usually, you'd set this to true in production
    sameSite: 'None',        // This is usually needed for cross-domain requests
    path: '/',             // Makes the cookie accessible from all paths
    maxAge: 3600 * 1000 // 1 hour in milliseconds
  });
};

// Usage example within a login endpoint
app.post('/login', async (req, res) => {
  // Imagine getting the token from Auth0
  const token = "some_auth_token_from_auth0";
  setAuthCookie(res, token);
  res.send({ message: 'Logged in successfully!' });
});
```

Notice the critical parts: the `domain` is set to `.mydomain.com`, not `app.mydomain.com`. Also, the `sameSite: 'None'` directive often becomes necessary, especially when running applications on different ports, and it should always be used in conjunction with `secure:true`. `HttpOnly: true` will prevent client-side js from accessing, this should be used in nearly every scenario.

**Step 3: Utilizing JWT for API Authentication**

For services on different subdomains (like `api.mydomain.com`), you will often use the `auth_token` cookie as a bearer token. The `auth_token` cookie contains the Json Web Token (JWT). You would then extract the JWT from the cookie header and utilize middleware to validate the JWT against your Auth0 tenant's public keys. This verification ensures that only authenticated requests are handled. Here's another example, showcasing this JWT verification process using Javascript (Node.js) and the `jsonwebtoken` library. Again this is not Auth0 specific but how an API will be able to parse the incoming JWT:

```javascript
// Example JWT Verification Middleware

import jwt from 'jsonwebtoken';

const verifyToken = async (req, res, next) => {
    const token = req.cookies.auth_token;
    if (!token) {
        return res.status(401).json({ message: 'No token, authorization denied' });
    }

    try {
        const decoded = jwt.decode(token, {complete: true});
        const kid = decoded.header.kid;

        // This next section would usually pull keys from Auth0 JWKS endpoint
        // For brevity, this part is simplified:
        const publicKeys = {
          'mockKey': "your_public_key_from_auth0_jwks"
        };
        const publicKey = publicKeys[kid];

        if (!publicKey) {
          return res.status(401).json({ message: 'Invalid public key'});
        }
        jwt.verify(token, publicKey, {algorithms: ['RS256']}, (err, user) => {
            if (err) {
                return res.status(403).json({ message: 'Token is not valid' });
            }
          req.user = user; // attach user information
          next();

        });
    } catch (e) {
        return res.status(400).json({ message: 'Invalid token' });
    }

};

// Example usage on an api endpoint

app.get('/secure-route', verifyToken, (req, res) => {
  res.json({ message: 'This is a protected route' });
});

```

This example is streamlined and does not include pulling keys from Auth0’s jwks endpoint. You would typically pull your public keys for verification from `https://your-domain.us.auth0.com/.well-known/jwks.json`.

**Step 4: Handling Refresh Tokens**

In longer-lived applications, you'll want to handle refresh tokens so users aren't forced to log in continuously. If you have configured Auth0 to provide refresh tokens, these usually need to be stored in secure httpOnly cookies, similar to the `auth_token`. Your application will need to utilize these refresh tokens to acquire new access tokens, thus maintaining a persistent authentication session. Here's a simplified Javascript example of how your front-end might request a refresh token and a new `auth_token` using the `auth0-spa-js` library:

```javascript

import { useAuth0 } from '@auth0/auth0-react';

const RefreshTokenHandler = () => {

  const { getAccessTokenSilently } = useAuth0();

    const getNewAccessToken = async () => {
      try {
        const newAccessToken = await getAccessTokenSilently({
          useRefreshTokens: true,
        });
          // this token is ready to send to server
        console.log('newAccessToken', newAccessToken);

        // you could now update an auth_token cookie via an api request
      } catch (e) {
        console.log(e);
      }
    };

    return <button onClick={getNewAccessToken}>Refresh Token</button>;
};
```

**Important Considerations**

*   **Security:** Always configure cookies with `httpOnly: true`, `secure: true` (especially in production), and the appropriate `sameSite` attribute. This helps prevent XSS and CSRF vulnerabilities. Note that setting `secure:true` mandates usage of https. `sameSite:None` requires that you are also setting `secure: true`.
*   **Auth0 Documentation:** Refer to the official Auth0 documentation regarding custom domains and cookie configurations. Auth0's setup can evolve, so always consult their latest guidance.
*   **CORS:** Ensure you have configured Cross-Origin Resource Sharing (CORS) appropriately if your subdomains communicate with each other through API calls.
*   **Token Storage:** Be careful with where and how you store tokens, especially refresh tokens. Client-side storage isn't always the safest route. Consider secure httpOnly cookies or server-side storage where appropriate, depending on your application.
*   **Library updates:** Be mindful of library updates. For example, Auth0's JS libraries may occasionally introduce breaking changes or require configuration updates. Always stay up-to-date with release notes.

**Recommended Resources**

For deeper dives, I recommend these sources:

*   **"RFC 6265: HTTP State Management Mechanism"**: This is the definitive specification for how cookies work in HTTP. It will give you an intimate understanding of cookie attributes and scoping rules.
*   **"OAuth 2.0 Security Best Current Practice":** This document provides security best practices for handling OAuth 2.0, especially around token management. It's crucial for understanding how to properly handle access and refresh tokens in your implementation.
*   **Auth0's official documentation:** Regularly check the official Auth0 documentation and community forums for the most up-to-date information and solutions.
*   **The OWASP guide on session management:** OWASP provides valuable resources for web security. Focus on their guidance regarding session management, especially cookie security practices.

In closing, achieving single sign-on across subdomains using Auth0 on a custom domain requires a solid understanding of cookie scoping, token management, and the security best practices associated with them. It isn't magic, just careful configuration and adherence to the standards. By understanding these concepts and referencing official documentation, you can successfully build a seamless user authentication flow across all your applications.
