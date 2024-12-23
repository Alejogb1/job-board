---
title: "Why am I getting a 401 Unauthorized error when accessing a Quarkus application in a Docker container secured by Keycloak?"
date: "2024-12-23"
id: "why-am-i-getting-a-401-unauthorized-error-when-accessing-a-quarkus-application-in-a-docker-container-secured-by-keycloak"
---

, let’s unpack this 401 error with your Quarkus application behind a Keycloak barrier, all snug in a Docker container. This is a surprisingly common hiccup, and I’ve personally navigated these authentication labyrinths a few times, especially when intricate docker-compose setups start layering on complexity. Instead of blaming gremlins, let’s systematically trace potential points of failure.

The 401 Unauthorized response, in its essence, means your application is trying to access a protected resource without presenting valid credentials or with credentials that the authentication server, Keycloak in this case, has rejected. It's less about the container and more about the authentication handshake. I’d bet a considerable amount that the culprit lies somewhere in the interplay of Keycloak’s configuration, your Quarkus application's security settings, and possibly the Docker network. I’ve spent many late nights troubleshooting very similar issues.

First off, let’s consider the authentication flow itself. Ideally, your Quarkus application will redirect unauthenticated requests to Keycloak, where the user logs in (or has an existing session). Keycloak then issues a token, and your application then verifies this token against Keycloak. A problem at any stage in this process can result in a 401. Let’s examine potential pitfalls:

*   **Keycloak Configuration:** A common source of errors is misconfiguration within Keycloak itself. Have you correctly set up a *realm* and *client* that your Quarkus application uses? Ensure the client has the correct *redirect URIs* configured – these are incredibly critical. It sounds simple, but mismatched URLs between Keycloak and your application's configuration will absolutely block authentication. The client *access type* also matters; for your application, it's likely going to be *confidential*, which requires a client secret. Double check these; typos are the bane of a sysadmin’s existence. Check the Keycloak logs to see if requests reach the server and if any obvious errors appear during token issuance. *Debugging security often requires you to trace exactly what Keycloak and your application are ‘seeing’.*

*   **Quarkus Application Security Settings:** Here is where many fall into a trap. Ensure you’ve added the correct Quarkus extensions (e.g., `quarkus-oidc`, possibly `quarkus-smallrye-jwt`) to handle the OIDC flow or JWT validation. The `application.properties` (or `application.yaml`) file of your Quarkus application is the gatekeeper here. Critical properties include `quarkus.oidc.auth-server-url`, `quarkus.oidc.client-id`, and if your client has *confidential* access type, `quarkus.oidc.credentials.secret`. Furthermore, if your Quarkus app is behind a reverse proxy, you will need to make sure `quarkus.oidc.enabled-for-all` is false in many cases, and rely on annotations in the code to protect only relevant endpoints. Don't forget to check any `mp-jwt` properties, since in some cases, that is used by default.

*   **Docker Networking:** I’ve seen instances where Docker's networking caused unexpected behavior. Ensure your Quarkus app within the container can actually resolve the Keycloak server. Docker, in its own internal network, might have a different hostname or port for Keycloak than you'd assume from your host machine. This is especially true when using docker-compose. Your application might be attempting to connect to `localhost:8080` when it's, in fact, on `keycloak:8080` within the Docker network. Explicitly declare the Keycloak server hostname within the docker network within your application’s configuration.

Let's illustrate with a few snippets that I’ve personally found beneficial, given these three common issues.

**Example 1: Keycloak Realm and Client Setup (pseudo-code)**

Let's imagine a simple Keycloak setup. I'm using JSON-like notation to outline the structure.

```json
{
  "realm": {
    "name": "my-quarkus-realm",
    "clients": [
      {
        "clientId": "my-quarkus-app",
        "accessType": "confidential",
        "secret": "my-secret-client-string",
        "redirectUris": [
          "http://localhost:8080/*" ,
          "http://quarkus-app:8080/*" // in docker network, if used
        ],
          "webOrigins": ["*"] // or specify origins, for development, be careful using *
      }
    ]
  }
}
```

This JSON block represents what you’d configure via the Keycloak admin console. Crucially, the `redirectUris` includes both `localhost` for local development and what might be the Docker hostname for network access. Make sure to add this `webOrigins` property as well, or you may run into a CORS error when testing through a front-end. It also shows `confidential` client type and that a client secret must be used.

**Example 2: Quarkus application.properties (or .yaml) Configuration**

Here's how that might translate into your Quarkus application’s configuration, focusing on the key bits using an example in `application.properties` file:

```properties
quarkus.oidc.auth-server-url=http://keycloak:8080/auth/realms/my-quarkus-realm
quarkus.oidc.client-id=my-quarkus-app
quarkus.oidc.credentials.secret=my-secret-client-string
quarkus.oidc.enabled=true
quarkus.oidc.token.validation.required=true
quarkus.oidc.tenant-enabled=true
```

Note the `auth-server-url` now explicitly uses `keycloak` which is how the Keycloak container would be named in a `docker-compose.yml` file, rather than localhost, which works fine locally. This explicitly tells your application where to connect to Keycloak within its Docker network context. These are critical settings, and mistakes here are common. You should have a similar section if using JWT, however OIDC is generally recommended for modern applications. The important bit is that `token.validation.required` will enable token verification.

**Example 3: Docker Compose Setup**

Here’s a simplified `docker-compose.yml` example, highlighting the networking aspect:

```yaml
version: "3.9"
services:
  keycloak:
    image: quay.io/keycloak/keycloak:latest
    ports:
      - "8080:8080"
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
    networks:
       - my_app_net

  quarkus-app:
    build: .
    ports:
      - "8081:8080"
    depends_on:
        - keycloak
    networks:
      - my_app_net

networks:
  my_app_net:
    driver: bridge
```

Notice how both containers are on the same `my_app_net` network, and therefore `keycloak` is resolvable by the `quarkus-app` container by its container name `keycloak`, allowing it to find Keycloak at `http://keycloak:8080`.

Troubleshooting this issue often means a process of systematic checking. First, verify that the user can log in at keycloak directly, for example, by opening the keycloak login page in a browser. Next, examine both Keycloak and application logs. Keycloak will log when it receives an authentication request, any failures during token generation, or invalid configurations. Your application's logs should show how it’s attempting to connect to Keycloak and the results. Set the logging level in `application.properties` to `DEBUG` for the `org.eclipse.microprofile.jwt` and `io.quarkus.oidc` packages during debugging to see more detail. If no errors are visible in the application logs, the problem likely is a failure in the initial redirect to keycloak (e.g. a url misconfiguration).

For authoritative resources, I strongly recommend checking the official Keycloak documentation, it’s truly invaluable. For deeper understanding of OIDC and JWT flows, look into *OpenID Connect 1.0* specification and the *JSON Web Token (JWT)* RFC 7519. Also, *Programming Identity Management in Java* by Scott Oaks and Pratik Patel is an excellent reference that dives into more implementation details related to this subject. I also found *OAuth 2 in Action* by Justin Richer helpful in understanding the underlying mechanisms. These resources can help you build a solid understanding beyond specific frameworks.

Debugging authentication issues can be tiresome. It can feel like you are spinning your wheels, but with systematic checks, reviewing the authentication flow, and making incremental changes, you can identify and resolve the cause of the 401 error. And believe me, it's quite satisfying when that login page finally appears without an error message.
