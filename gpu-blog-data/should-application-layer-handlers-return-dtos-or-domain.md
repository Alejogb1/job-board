---
title: "Should application layer handlers return DTOs or domain objects?"
date: "2025-01-30"
id: "should-application-layer-handlers-return-dtos-or-domain"
---
The decision of whether application layer handlers should return Data Transfer Objects (DTOs) or domain objects hinges on a crucial architectural principle: the separation of concerns.  My experience building and maintaining large-scale systems, particularly within the financial services sector, has repeatedly underscored the importance of this principle.  Returning domain objects directly from the application layer often leads to unintended coupling and compromises data integrity and security.  Therefore, I strongly advocate for returning DTOs.

**1. Clear Explanation:**

The application layer serves as a bridge between the presentation layer (UI, API) and the domain layer.  Its primary responsibility is to orchestrate business logic and translate requests into actions within the domain. Domain objects encapsulate business rules and data within the core of the application. They are often rich in functionality and may contain sensitive information or complex relationships.  Conversely, DTOs are simple data containers tailored specifically for data transfer. They are designed to minimize data exposure and are devoid of business logic.

Returning domain objects directly exposes the internal structure of the domain model to the presentation layer.  This creates tight coupling, making it difficult to evolve the domain model without impacting the presentation layer.  Changes in the domain, such as adding or removing fields, necessitate modifications throughout the application. Furthermore, exposing the entire domain object may inadvertently leak sensitive information to the presentation layer, potentially violating security best practices.

In contrast, returning DTOs provides a level of abstraction.  The application layer maps domain objects to DTOs, selectively including only the data required by the presentation layer.  This loose coupling allows for independent evolution of the domain and presentation layers.  Changes in the domain model will not necessarily ripple through the presentation layer, provided the DTO remains unchanged or adapts accordingly.  Furthermore, carefully constructed DTOs can effectively mask sensitive information, enhancing security.

The act of mapping from domain objects to DTOs also provides an opportunity for data transformation and validation.  The application layer can sanitize, format, or augment the data before presenting it to the client, ensuring data integrity and consistency.  This transformation process cannot be easily managed if domain objects are returned directly.


**2. Code Examples with Commentary:**

**Example 1:  Illustrating direct domain object return (undesirable):**

```java
// Domain object
public class User {
    private String id;
    private String passwordHash; // Sensitive data!
    private String email;
    // ... other fields and methods ...

    // Getters and setters
}

// Application layer handler (incorrect)
public User getUserById(String userId) {
    User user = userRepository.findById(userId);
    return user; // Directly returning the domain object exposes sensitive data.
}
```

This example highlights the security risk of returning a domain object directly. The `passwordHash` field, containing sensitive information, is exposed to the calling layer.


**Example 2:  Illustrating DTO return with simple mapping:**

```java
// DTO
public class UserDTO {
    private String id;
    private String email;

    // Getters and setters
}

// Application layer handler (correct)
public UserDTO getUserById(String userId) {
    User user = userRepository.findById(userId);
    UserDTO userDTO = new UserDTO();
    userDTO.setId(user.getId());
    userDTO.setEmail(user.getEmail());
    return userDTO;
}
```

Here, a `UserDTO` only contains non-sensitive information.  The application layer explicitly selects and maps the relevant fields, preventing accidental exposure of sensitive data.


**Example 3:  Illustrating DTO return with more complex mapping and data transformation:**

```java
// DTO
public class OrderSummaryDTO {
    private String orderId;
    private String customerName;
    private double totalAmount;
    private String status;
}


// Application layer handler (correct)
public OrderSummaryDTO getOrderSummary(String orderId) {
    Order order = orderRepository.findById(orderId);
    OrderSummaryDTO orderSummaryDTO = new OrderSummaryDTO();
    orderSummaryDTO.setOrderId(order.getId());
    orderSummaryDTO.setCustomerName(order.getCustomer().getName());
    orderSummaryDTO.setTotalAmount(order.calculateTotal()); // Business logic resides in the domain
    orderSummaryDTO.setStatus(order.getStatus().toString().toUpperCase()); //Data Transformation

    return orderSummaryDTO;
}
```

This example demonstrates more advanced mapping.  The `calculateTotal()` method resides within the `Order` domain object, keeping business logic contained. Data transformation, like converting the status to uppercase, is handled during DTO creation.


**3. Resource Recommendations:**

Several excellent books and articles delve into the principles of Domain-Driven Design and software architecture patterns. I would suggest seeking out works that explicitly cover application layer design and data transfer object usage.  In particular, material focusing on layered architecture and the separation of concerns will prove highly beneficial.  Exploring resources on object-relational mapping (ORM) frameworks will assist in understanding how to effectively translate between database models and domain objects, a process central to efficient DTO construction. Finally, studying security best practices for data handling and exposure will inform the development of secure and robust DTOs.
