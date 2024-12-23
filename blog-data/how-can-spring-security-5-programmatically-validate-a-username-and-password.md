---
title: "How can Spring Security 5 programmatically validate a username and password?"
date: "2024-12-23"
id: "how-can-spring-security-5-programmatically-validate-a-username-and-password"
---

Alright, let's talk about programmatically validating usernames and passwords with Spring Security 5. It's something I’ve tackled quite a bit over the years, especially back when I was building a distributed authentication service for a microservices architecture. We needed fine-grained control, bypassing the more convention-based approaches, and it involved some interesting custom implementations.

Essentially, you're moving away from relying solely on Spring Security’s default mechanisms—like parsing properties files or using database-backed authentication—and taking direct control over the validation logic. This is incredibly useful when you need to integrate with bespoke authentication systems, apply unique validation rules beyond simple equality checks, or interact with external services before granting access.

The core concept revolves around crafting a custom `AuthenticationProvider`. In Spring Security, the `AuthenticationProvider` interface is where the magic happens. It's the component responsible for taking an `Authentication` object (which usually contains credentials like username and password) and deciding if the user is authenticated or not. Instead of letting Spring Security handle this implicitly, you can define your own provider that does exactly what you want.

Here’s how I typically approach it, broken down into the necessary steps, followed by some code examples:

**Step 1: Implement a Custom `AuthenticationProvider`**

You'll create a class that implements the `AuthenticationProvider` interface. This interface requires two methods: `authenticate(Authentication authentication)` and `supports(Class<?> authentication)`. The `supports` method determines if this provider can handle a specific type of `Authentication` object. You’ll usually be working with the `UsernamePasswordAuthenticationToken`. The `authenticate` method is where your validation logic resides. It receives the `Authentication` object containing the credentials and returns an authenticated version of it if validation succeeds or throws an `AuthenticationException` if validation fails.

**Step 2: Register your Custom Provider**

Once you've implemented your custom provider, you need to register it with Spring Security’s authentication manager. This is typically done within a `WebSecurityConfigurerAdapter` class (or its functional equivalents if using the newer security configurations) where you configure the authentication mechanism using `AuthenticationManagerBuilder`. You add your custom `AuthenticationProvider` to the list of available providers.

**Step 3: Handle Success and Failure**

Your `authenticate` method needs to produce a successful `Authentication` object or throw an `AuthenticationException`. In the case of success, you often return a fully populated `UsernamePasswordAuthenticationToken`, potentially including granted authorities. On failure, you'll throw a specific exception, which can be tailored to your needs (e.g., `BadCredentialsException`, `LockedException`, etc.)

Let's dive into some code to clarify.

**Example 1: Basic Validation Against a Hardcoded User**

This example is very simplistic for demonstration purposes, but it shows the basic structure.

```java
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;

import java.util.Collections;

@Component
public class SimpleHardcodedAuthenticationProvider implements AuthenticationProvider {

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        String username = authentication.getName();
        String password = authentication.getCredentials().toString();

        if ("testuser".equals(username) && "password123".equals(password)) {
            return new UsernamePasswordAuthenticationToken(
                username,
                password,
                Collections.singletonList(new SimpleGrantedAuthority("ROLE_USER"))
            );
        } else {
            throw new BadCredentialsException("Invalid username or password");
        }
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return UsernamePasswordAuthenticationToken.class.isAssignableFrom(authentication);
    }
}
```

And you would then register this provider in your security configuration class like this:

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@Configuration
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Autowired
    private SimpleHardcodedAuthenticationProvider simpleAuthenticationProvider;

    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.authenticationProvider(simpleAuthenticationProvider);
    }
}
```

**Example 2: Validation Using an In-Memory User Service**

This example uses a more robust approach utilizing a custom service. This hypothetical `UserService` would handle user lookups, and we'd inject it into the authentication provider.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;

interface UserService {
    User findByUsername(String username);
}

record User (String username, String password, String role){}

@Component
class InMemoryUserService implements UserService {
    private final Map<String, User> users;

    InMemoryUserService() {
         users = new HashMap<>();
         users.put("testuser", new User("testuser", "password123", "ROLE_USER"));
         users.put("admin", new User("admin", "admin123", "ROLE_ADMIN"));
    }

    public User findByUsername(String username) {
        return users.get(username);
    }
}


@Component
public class UserServiceAuthenticationProvider implements AuthenticationProvider {

    @Autowired
    private UserService userService;

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        String username = authentication.getName();
        String password = authentication.getCredentials().toString();

        User user = userService.findByUsername(username);

        if (user != null && user.password().equals(password)) {
             return new UsernamePasswordAuthenticationToken(
               username,
                password,
                Collections.singletonList(new SimpleGrantedAuthority(user.role()))
            );

        } else {
             throw new BadCredentialsException("Invalid username or password");
        }
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return UsernamePasswordAuthenticationToken.class.isAssignableFrom(authentication);
    }
}
```

Again, the provider needs to be wired into the `AuthenticationManagerBuilder` in your security configuration just like Example 1.

**Example 3: Validation Against an External Service (Placeholder)**

Here, imagine validating against an external authentication service. We’ll abstract the communication logic behind a `ExternalAuthenticator` interface, for the sake of readability.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;
import java.util.Collections;

interface ExternalAuthenticator {
   boolean authenticate(String username, String password);
   String getRole(String username);
}

@Component
class MockExternalAuthenticator implements ExternalAuthenticator {
    public boolean authenticate(String username, String password) {
        return ("testuser".equals(username) && "password123".equals(password)) || ("admin".equals(username) && "admin123".equals(password));
    }
    public String getRole(String username) {
        if ("testuser".equals(username)) return "ROLE_USER";
        if ("admin".equals(username)) return "ROLE_ADMIN";
         return null;
    }

}


@Component
public class ExternalServiceAuthenticationProvider implements AuthenticationProvider {

    @Autowired
    private ExternalAuthenticator externalAuthenticator;

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        String username = authentication.getName();
        String password = authentication.getCredentials().toString();

        if (externalAuthenticator.authenticate(username, password)) {
            String role = externalAuthenticator.getRole(username);
            return new UsernamePasswordAuthenticationToken(
                username,
                password,
                Collections.singletonList(new SimpleGrantedAuthority(role))
            );
        } else {
             throw new BadCredentialsException("Invalid credentials from external service");
        }
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return UsernamePasswordAuthenticationToken.class.isAssignableFrom(authentication);
    }
}
```

Again, this would need to be registered in the security config.

**Key takeaways:**

*   **Customization:** The primary advantage is total control. You aren't bound by convention.
*   **Integration:** Easily plug into legacy or external authentication systems.
*   **Complex Validation:** Implement intricate logic (e.g., multi-factor, time-based checks).
*   **Error Handling:** Tailor the specific authentication exception types you want to throw.

**Further Reading:**

For deeper understanding, I highly recommend checking out the following:

1.  **"Spring Security Reference"**: This is the official documentation and the definitive source. Look specifically into the sections on authentication, providers, and configuration.
2.  **"Pro Spring Security" by Carlo R. B. Pescio**: This book provides an in-depth analysis of various Spring Security aspects including custom authentication providers. It walks you through the internals of how security is implemented.
3.  **"OAuth 2 in Action" by Justin Richer and Antonio Sanso**: While this is specifically on OAuth, understanding the flows and concepts around authorization frameworks can broaden your knowledge and improve understanding of security architecture.

These resources, coupled with practical experimentation, will give you a firm grasp on programmatically validating credentials using Spring Security. It's a powerful technique that can address complex security scenarios with flexibility and precision.
