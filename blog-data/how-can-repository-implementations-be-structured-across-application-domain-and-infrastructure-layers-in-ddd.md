---
title: "How can repository implementations be structured across application, domain, and infrastructure layers in DDD?"
date: "2024-12-23"
id: "how-can-repository-implementations-be-structured-across-application-domain-and-infrastructure-layers-in-ddd"
---

Okay, let's tackle this. Structuring repositories in a DDD context – it's something I've spent considerable time refining over the years, particularly when I was knee-deep in a large-scale e-commerce platform. We hit a few snags early on, mostly due to the classic ‘database-centric’ approach leaking into our domain logic. So, let’s dissect this a bit, focusing on how to truly separate concerns and maintain a robust architecture.

The core principle we’re aiming for is a clear separation of responsibilities, adhering to the layered architecture inherent in Domain-Driven Design. The goal is that our application logic (the "what") shouldn't care about the specifics of how data is persisted (the "how"). This means our repositories, acting as intermediaries between the domain and the infrastructure, need to be carefully crafted. I'll explain how I’ve typically handled that separation.

Let's first talk about the different layers involved.

**Domain Layer:** This is where your core business logic resides. It contains your entities, value objects, and most importantly, the repository *interfaces*. The crucial point here is that the domain layer only knows about the *interface* of the repository, not the concrete implementation. This allows us to swap out data stores without modifying our core domain logic. For instance, if you’re fetching a user object, your domain only needs to know that there’s a `UserRepository` interface that can `getById(userId)` or `save(user)`.

**Application Layer:** This is often called the use-case layer. It sits between the domain and the presentation layer. Here you orchestrate the interactions between domain objects, and this is where you’ll use those repository interfaces defined in the domain layer. It doesn’t define how those interfaces are implemented – that’s key – but rather uses them to fulfill application-specific needs.

**Infrastructure Layer:** This is where the concrete repository implementations live. These are the classes that actually deal with the data store (be it a relational database, a NoSQL database, a file system, or even an external api). This is where the nitty-gritty of data access happens, translating domain concepts into persistence details.

Now, let’s look at the structure and what I've found effective:

1.  **Repository Interfaces in the Domain:** Within your domain layer, define an interface for each aggregate that requires persistence. For instance, `interface UserRepository { User getById(UserId id); void save(User user);}`. This establishes the contract that the application layer will use. These interfaces are purely abstract, focusing on domain operations like `getById` or `save`, and they don't know anything about databases, entities, SQL, or JSON.

2.  **Concrete Implementations in Infrastructure:** The actual database interactions are handled within the concrete classes located in the infrastructure layer. Let's say you're using JPA for a SQL database. You would have a class `JpaUserRepository implements UserRepository`, where you implement the methods defined by the `UserRepository` interface. Inside, you would have your JPA entities and map between your domain model `User` and the database representation. This is where you'd handle things like creating queries, executing them, mapping between database entities and domain entities.

3.  **Dependency Injection:** The application layer depends on the interfaces defined by the domain and the concrete implementations from the infrastructure, but it doesn't need to know about the infrastructure dependencies *directly*. We use dependency injection to inject the concrete implementation of the repository into the application service. This is typically done via a dependency injection container (like Spring or Guice), or a composition root.

Here's how it might look with code examples. Let's start with a very basic example of a `User` aggregate.

**Domain Layer (Java):**

```java
// src/main/java/com/example/domain/user/User.java
package com.example.domain.user;

public class User {
    private UserId id;
    private String username;
    private String email;

    public User(UserId id, String username, String email) {
        this.id = id;
        this.username = username;
        this.email = email;
    }

    public UserId getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getEmail() {
        return email;
    }

    // Other domain logic methods
}

```
```java
// src/main/java/com/example/domain/user/UserId.java
package com.example.domain.user;
import java.util.UUID;

public class UserId {
   private UUID id;

   public UserId(UUID id) {
        this.id = id;
    }
    
   public UUID getId(){
        return id;
   }
}

```

```java
// src/main/java/com/example/domain/user/UserRepository.java
package com.example.domain.user;

public interface UserRepository {
    User getById(UserId id);
    void save(User user);
}
```

**Application Layer (Java):**

```java
// src/main/java/com/example/application/UserService.java
package com.example.application;

import com.example.domain.user.User;
import com.example.domain.user.UserId;
import com.example.domain.user.UserRepository;

public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User getUserById(UserId id) {
        return userRepository.getById(id);
    }
     public void createUser(User user){
        userRepository.save(user);
     }

    //Other application logic methods using repository
}
```

**Infrastructure Layer (Java, using a hypothetical in-memory store for simplicity):**

```java
// src/main/java/com/example/infrastructure/InMemoryUserRepository.java
package com.example.infrastructure;

import com.example.domain.user.User;
import com.example.domain.user.UserId;
import com.example.domain.user.UserRepository;
import java.util.Map;
import java.util.HashMap;

public class InMemoryUserRepository implements UserRepository {

    private final Map<UserId, User> users = new HashMap<>();
    @Override
    public User getById(UserId id) {
        return users.get(id);
    }

    @Override
    public void save(User user) {
         users.put(user.getId(), user);
    }

    //Other persistence related logic
}
```

**Key Observations**

*   The `UserService` (in the application layer) is not coupled to the specific persistence mechanism (`InMemoryUserRepository`). It’s working with the `UserRepository` interface. This allows us to swap the in-memory repository with an actual database repository without modifying `UserService`.
*   The Domain Layer knows nothing about the infrastructure. This provides a clear separation of concerns.
*   The `InMemoryUserRepository` handles the specific logic of storing `User` objects in memory (in a real system it could be saving to a SQL database or NoSQL data store).

This example, albeit simplified, illustrates the basic principle of layering repository implementations. Let’s see another one, this time with a potential JPA implementation. I'm just including the repository class, keeping the rest of the code structure the same.

**Infrastructure Layer (Java, JPA implementation)**

```java
// src/main/java/com/example/infrastructure/JpaUserRepository.java
package com.example.infrastructure;

import com.example.domain.user.User;
import com.example.domain.user.UserId;
import com.example.domain.user.UserRepository;
import com.example.infrastructure.entity.UserEntity;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import java.util.UUID;

import java.util.Optional;

public class JpaUserRepository implements UserRepository {

    @PersistenceContext
    private EntityManager entityManager;

    @Override
    public User getById(UserId id) {
         UUID userId = id.getId();
         UserEntity userEntity = entityManager.find(UserEntity.class, userId);
         if (userEntity == null){
            return null;
         }
        return new User(new UserId(userEntity.getId()), userEntity.getUsername(), userEntity.getEmail()); // mapping the entity to the domain object
    }

    @Override
    public void save(User user) {
        UserEntity userEntity = new UserEntity();
        userEntity.setId(user.getId().getId());
        userEntity.setUsername(user.getUsername());
        userEntity.setEmail(user.getEmail());
         entityManager.persist(userEntity);
    }
}
```

**Infrastructure Layer (Java, User JPA Entity):**

```java
// src/main/java/com/example/infrastructure/entity/UserEntity.java
package com.example.infrastructure.entity;

import jakarta.persistence.*;
import java.util.UUID;


@Entity
@Table(name = "users")
public class UserEntity {
    @Id
    private UUID id;
    private String username;
    private String email;

     public UUID getId() {
        return id;
    }
    public void setId(UUID id){
        this.id = id;
    }

    public String getUsername(){
        return username;
    }

    public void setUsername(String username){
       this.username = username;
    }
    public String getEmail(){
        return email;
    }

    public void setEmail(String email){
        this.email = email;
    }
    
}
```

Here, we are using JPA to handle the persistence. The JpaUserRepository translates from the domain object `User` to the persistence entity object `UserEntity`. Notice that our domain logic is completely agnostic to JPA, or even that a database is being used at all.

As a third, quite contrasting example, imagine the repository had to interact with a third-party api, we could have something like this:

**Infrastructure Layer (Java, External API repository):**

```java
// src/main/java/com/example/infrastructure/ExternalApiUserRepository.java
package com.example.infrastructure;

import com.example.domain.user.User;
import com.example.domain.user.UserId;
import com.example.domain.user.UserRepository;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import java.util.UUID;

import java.util.Optional;

public class ExternalApiUserRepository implements UserRepository {

    private final RestTemplate restTemplate;
    private final String apiUrl = "https://api.external.example/users/"; //Example external API URL
     
    public ExternalApiUserRepository(RestTemplate restTemplate){
        this.restTemplate = restTemplate;
    }

    @Override
    public User getById(UserId id) {
        UUID userId = id.getId();
        ResponseEntity<UserApiDto> response = restTemplate.getForEntity(apiUrl+userId, UserApiDto.class);
         if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
           UserApiDto userDto = response.getBody();
           return new User(new UserId(UUID.fromString(userDto.getId())), userDto.getUsername(), userDto.getEmail());
        }else {
           return null;
        }

    }

    @Override
    public void save(User user) {
        UserApiDto userDto = new UserApiDto();
        userDto.setId(user.getId().getId().toString());
        userDto.setUsername(user.getUsername());
        userDto.setEmail(user.getEmail());
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<UserApiDto> request = new HttpEntity<>(userDto, headers);
        restTemplate.postForEntity(apiUrl,request,String.class);
    }

    //Helper class to serialize the request and map the response
    private static class UserApiDto{
        private String id;
        private String username;
        private String email;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }
    }
}
```

This final example shows how we can interact with an external API while maintaining the same structure. We are still implementing the `UserRepository` interface defined in the domain and returning our domain object as a result of using the external service. The rest of our application remains completely unaffected.

**Recommendations for Further Reading**

To dive deeper into this topic, I highly recommend the following:

*   **"Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans:** This is the seminal work on DDD. It provides the foundational principles for the approach.
*   **"Implementing Domain-Driven Design" by Vaughn Vernon:** This book offers a practical, hands-on guide to applying DDD principles, including excellent chapters on repository patterns.
*  **"Patterns of Enterprise Application Architecture" by Martin Fowler:** This is invaluable for understanding various architectural patterns, and it touches on the repository pattern in the context of larger application design.

Remember, the goal of this architecture is not just to “follow DDD blindly,” but to build software that is maintainable, testable, and resilient to changes. The separation of concerns provided by correctly structured repositories is a key element in achieving that. The examples provided above are illustrative, but I encourage you to adapt them to the specific needs of your projects.
