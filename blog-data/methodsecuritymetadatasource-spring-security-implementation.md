---
title: "methodsecuritymetadatasource spring security implementation?"
date: "2024-12-13"
id: "methodsecuritymetadatasource-spring-security-implementation"
---

Okay so you're asking about `MethodSecurityMetadataSource` in Spring Security huh? I've been down that rabbit hole a few times let me tell you it's not always sunshine and roses but it gets the job done once you wrangle it I'll give you the lowdown based on my experience

Basically `MethodSecurityMetadataSource` is the unsung hero the brains behind method level authorization in Spring Security It's the interface responsible for figuring out what security attributes apply to a given method invocation Think of it as a map that links methods to the access control rules that should be enforced

The framework uses it at runtime during method calls to decide whether the current user has the right permissions to execute a method It's a core component of Spring Security's method security model so getting your head around it is pretty crucial

Now there are a bunch of different implementations provided by Spring out of the box and they're each good for different use cases I'll talk through the ones I've battled with myself a bit

One of the most common implementations you'll see is `AnnotationMethodSecurityMetadataSource` This guy reads security annotations like `@PreAuthorize`, `@PostAuthorize`, `@Secured` etc directly from your methods that's pretty neat right? It scans your code for these annotations and turns them into security attributes that the access control infrastructure uses

I remember this one time I was working on a large legacy project It had a complete mess of security annotations scattered across the codebase It was like playing a game of where's waldo but with access control rules I ended up spending a week writing a bunch of unit tests to ensure the security configuration was working as intended and it was also very confusing as some methods had too much security logic baked into the methods themselves It taught me the importance of proper planning though

Here's an example of how you might use annotations

```java
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.stereotype.Service;

@Service
public class MyService {

    @PreAuthorize("hasRole('ADMIN')")
    public void adminOnlyOperation() {
        // Admin stuff goes here
    }

    @PreAuthorize("hasAnyRole('USER', 'ADMIN')")
    public void userOrAdminOperation() {
         // Stuff for users and admins
    }

    public void publicOperation() {
       // Open for everyone
    }

}
```

This example is pretty self explanatory each method is annotated with the required role

Another key implementation is `MapBasedMethodSecurityMetadataSource` This implementation does not use annotations It's more manual you have to manually configure the mapping between your methods and security attributes in configuration files or in code I used this when a client required a very specific and bespoke security setup for their web services where it would be difficult to use annotations.

I remember dealing with this in a project where we had a complex multi-tenancy setup The access rules were way too intricate for basic annotations plus they were managed centrally so we went this route. I remember I spent a lot of time debugging this manually when the configurations were wrong that's always a pain.

Here's a basic example of how you might set it up using a map

```java
import org.springframework.security.access.ConfigAttribute;
import org.springframework.security.access.SecurityConfig;
import org.springframework.security.access.method.MapBasedMethodSecurityMetadataSource;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MyCustomMetadataSource extends MapBasedMethodSecurityMetadataSource {

    public MyCustomMetadataSource() {
       Map<Method, List<ConfigAttribute>> map = new HashMap<>();

        try{

            Method adminMethod = MyService.class.getMethod("adminOnlyOperation");
            map.put(adminMethod, List.of(new SecurityConfig("ROLE_ADMIN")));

            Method userOrAdminMethod = MyService.class.getMethod("userOrAdminOperation");
            map.put(userOrAdminMethod, List.of(new SecurityConfig("ROLE_USER"), new SecurityConfig("ROLE_ADMIN")));

            setMethodMap(map);

        }catch(NoSuchMethodException ex){
            throw new RuntimeException("Could not find method", ex);
        }

    }

}
```

As you can see this is all manual it gets messy quickly if you have too many methods and that's a real issue

Now there's another one `DelegatingMethodSecurityMetadataSource` which is like a dispatcher it delegates calls to other `MethodSecurityMetadataSource` implementations I often used this if I need to use multiple sources of security rules for example maybe a combination of annotation based and programmatically defined rules This is probably the most advanced one of the three and requires more careful setup to make sure everything is properly configured. When this goes wrong it can be a real headache to solve.

Here is an example of how you would delegate two method security metadata sources

```java
import org.springframework.security.access.method.DelegatingMethodSecurityMetadataSource;
import org.springframework.security.access.method.MethodSecurityMetadataSource;
import org.springframework.security.access.annotation.AnnotationMethodSecurityMetadataSource;

import java.util.List;

public class MyDelegatingMetadataSource extends DelegatingMethodSecurityMetadataSource{

    public MyDelegatingMetadataSource() {
        super(List.of(new AnnotationMethodSecurityMetadataSource(), new MyCustomMetadataSource() ));
    }


}
```

You might have some scenarios where you would need to customize `MethodSecurityMetadataSource` if you're building some super complex application with very specific requirements

Here is where you need to put on your thinking cap and start thinking about implementing a specific one for your specific security needs. Usually if you reached this point you're in quite deep water. I've implemented my custom one for a few clients that required completely different approaches from the standard ones but there is also the possibility of using a default one if possible

The way to go is to implement the `MethodSecurityMetadataSource` interface directly and then you define exactly how method-to-security-attribute mappings are computed It's a lot of work so I would suggest to only go for this approach as a last resource. It will give you the maximum level of flexibility but at a high development cost.

I've seen some cases where people try to use `MethodSecurityMetadataSource` as an excuse to ignore best practices for example writing complex security logic directly in methods I mean you know who you are it's not pretty and it really impacts maintainability and testing its a shortcut but its not very good long term It is good to keep your code simple and to the point and do not abuse annotations too much. It is not all about shortcuts but about writing good software.

Resources? Well for a deep dive check out the Spring Security Reference Documentation obviously it has all the answers. A book I found quite insightful is "Spring Security in Action" by Laurentiu Spilca. If you want to really dig into the authorization aspect try "Role-Based Access Control: A Review and Evaluation" by Ravi Sandhu. I know its old but it is still very very relevant for understanding the basics.

In a nutshell `MethodSecurityMetadataSource` is a very powerful tool and its core for method based security in Spring Security but it requires a careful configuration and planning otherwise it can become a massive pain to manage.

Also a tip from my own experience is to keep your security rules simple and testable. It is better to avoid complexity whenever possible otherwise it becomes a debugging hell believe me I have been there.
