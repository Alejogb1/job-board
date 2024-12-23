---
title: "How can custom claims be added from a database?"
date: "2024-12-23"
id: "how-can-custom-claims-be-added-from-a-database"
---

Alright, let's tackle this one. Having worked on various identity and access management (iam) systems over the years, adding custom claims sourced from a database is a recurring need, and one that often warrants careful planning. The basic concept involves fetching data from your database—usually based on some user identifier—and packaging it into claims that are then included in the user's access token. This approach allows for fine-grained authorization and can avoid constant trips back to the database for authorization purposes.

The primary challenge often lies in how you integrate this database lookup into your authentication workflow. You have to consider performance, security, and maintainability. Ideally, you want the claim insertion process to be as seamless as possible without introducing significant latency or vulnerabilities. I’ve personally seen implementations where poorly handled database lookups crippled authentication services under heavy load, so this is not a trivial matter.

The key is to intercept the authentication flow after the user is authenticated but before the token is issued. This is where the concept of an "authentication hook" or a "claims provider" becomes crucial. Depending on the iam system you're using—whether it's something you've built yourself, a framework like spring security, or a dedicated solution such as auth0 or okta—the specifics will vary. However, the underlying principle remains the same: you inject a custom logic component that takes the authenticated user's identifier, queries the database, and adds the resulting information as claims to their token.

Let’s look at three scenarios and implementations, using different tech stacks for illustration.

**Scenario 1: A simple java application with spring security**

Assume we are building a java microservice and using spring security for authentication. We can create a custom `authenticationeventlistener` that fetches user roles from the database, which act as our custom claims.

```java
import org.springframework.context.ApplicationListener;
import org.springframework.security.authentication.event.AuthenticationSuccessEvent;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class CustomClaimsProvider implements ApplicationListener<AuthenticationSuccessEvent> {

    private final UserRepository userRepository; // Assume this exists

    public CustomClaimsProvider(UserRepository userRepository) {
        this.userRepository = userRepository;
    }


    @Override
    public void onApplicationEvent(AuthenticationSuccessEvent event) {
        if (event.getAuthentication().getPrincipal() instanceof User) {
             User userDetails = (User) event.getAuthentication().getPrincipal();
             String username = userDetails.getUsername();

             UserEntity user = userRepository.findByUsername(username);
             if (user != null){
                List<GrantedAuthority> roles =  user.getRoles().stream()
                        .map(role -> new SimpleGrantedAuthority("ROLE_" + role.getName()))
                        .collect(Collectors.toList());

                userDetails.getAuthorities().addAll(roles); //add roles as authorities which can be extracted as claims
            }
        }
    }
}

```

In this example, we're listening to `AuthenticationSuccessEvent`. When a user successfully authenticates, we retrieve the principal (which should be our user object), use their username to query the database via `userRepository.findByUsername`, then enrich the authentication object with roles. These roles, in spring security terms, are equivalent to 'claims'. The principal's authorities are easily converted to claims during token generation.

**Scenario 2: A Node.js application with passport.js**

Moving over to the javascript world, suppose you are using passport.js for authentication. The implementation looks different, but the logic remains the same.

```javascript
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const db = require('./db'); // Assume a db connection module

passport.use(new LocalStrategy(
    {usernameField: 'email'}, // if your username field is not 'username'

    async function(email, password, done) {
    try {
        const user = await db.getUserByEmail(email);

        if (!user) {
            return done(null, false, { message: 'Incorrect email.' });
        }
        if (user.password !== password) { //simple password check here
            return done(null, false, { message: 'Incorrect password.' });
        }

        //at this point user is authenticated
        const customClaims = await fetchUserClaims(user.id); //Fetch claims based on user ID

        const enrichedUser = { ...user, claims: customClaims }; // add user claims as a field

        return done(null, enrichedUser);


    } catch (error) {
        return done(error);
    }


    }

));


passport.serializeUser(function(user, done) {
    done(null, user.id);
});

passport.deserializeUser(async function(id, done) {
   try {
       const user = await db.getUserById(id);
       const customClaims = await fetchUserClaims(id) //Fetch claims based on user ID
       const enrichedUser = { ...user, claims: customClaims };
       done(null, enrichedUser);

   } catch (error){
       done(error);
   }

});

// this function is responsible for fetching custom claims.
async function fetchUserClaims(userId){

        const userClaims = await db.fetchCustomClaims(userId);
         return userClaims
    }

```

In this passport.js example, the `LocalStrategy` is modified. We're fetching user details, and importantly, calling `fetchUserClaims` to pull the custom claims from the database. Then the claims are added as part of a new user object that is returned after authentication. We also need to make sure to deserialize the claims after successful authentication. Passport manages session and these claims are also persisted in the session.
The `fetchCustomClaims` function is an abstraction that handles querying the database for claims based on the user id.

**Scenario 3: Using a dedicated iam solution such as auth0**

Dedicated iam solutions such as auth0 usually provide a more high-level interface that makes claim enrichment easier. Instead of directly modifying authentication callbacks, you generally use a rules or actions mechanism. Here is a conceptual example of how this can be done in auth0 using a rule:

```javascript
    async function (user, context, callback) {
      const namespace = 'https://example.com/claims/';
      const user_id = user.user_id;

      const axios = require('axios');
      try {

            const response = await axios({
                method: 'get',
                url: `https://your-api.com/users/${user_id}/claims` // Fetch claims from your database via an API
              });
           const userClaims = response.data;

           if (userClaims) {
                userClaims.forEach(claim => {
                     context.idToken[namespace + claim.key] = claim.value;
                     context.accessToken[namespace + claim.key] = claim.value;
                });

          }

          callback(null, user, context);
        }
         catch (error){
             callback(error);
         }


    }
```

In this example, we are using an auth0 rule to fetch custom claims from a custom api endpoint. After fetching the claims, we assign them to the `idToken` and `accessToken`.  Note that the rule is an async function. Here the custom claims are namespaced to prevent clashes with existing claims. The claims are obtained via an api and not directly from the database, which means you'll need an api that can query the database.

**Key Considerations and Recommendations**

When implementing database-sourced claims, remember these crucial aspects:

*   **Performance**: Database lookups during authentication can introduce latency, especially under high load. Consider caching mechanisms and optimize your database queries.
*   **Security**: Be meticulous about data sanitization when querying the database and make sure only the necessary claims are added to the tokens. Avoid leaking sensitive data.
*   **Error Handling**: Gracefully handle database errors and avoid exposing too much information in error responses.
*   **Claims Management**: Develop a clear strategy for managing and updating claims as your application evolves. You should also have a way to invalidate the token if claim related information changes.
*   **Token size:** Be aware of limitations imposed by token sizes, such as those found in json web tokens (jwt). Avoid overly large claim sets to prevent token size exceeding limits.
*   **Data Consistency**: Ensure that changes in user attributes that impact claims are properly propagated and updated.

For further reading, I would strongly recommend the "OAuth 2 in Action" book by Justin Richer and Antonio Sanso, for a comprehensive overview of OAuth 2.0, which is often the backbone for identity management systems and claims management. Furthermore, if you're using java, familiarize yourself with the Spring Security documentation, especially regarding custom authentication providers. For javascript, understanding the inner workings of passport.js, and, for specific implementations, the documentation for your IAM provider like Auth0 or Okta is essential. For general understanding of identity and access management, "Designing Identity Management Solutions" by Jay Heiser, John Kindervag and Daniel Lewin is also a great resource to look into.

This approach, while more complex than hardcoding claims, is essential for scaling applications that require dynamic, role-based access control. By carefully planning and implementing these steps, you can significantly enhance the security and flexibility of your authorization architecture.
