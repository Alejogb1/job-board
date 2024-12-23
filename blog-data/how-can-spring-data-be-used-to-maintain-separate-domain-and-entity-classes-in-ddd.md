---
title: "How can Spring Data be used to maintain separate domain and entity classes in DDD?"
date: "2024-12-23"
id: "how-can-spring-data-be-used-to-maintain-separate-domain-and-entity-classes-in-ddd"
---

Okay, let's tackle this. I remember back when I was working on a large e-commerce platform, we faced a similar challenge: how to cleanly separate our domain logic from the persistence layer using Spring Data while adhering to Domain-Driven Design (DDD) principles. It’s a common stumbling block, and the solution requires a thoughtful approach. It's not always intuitive, especially when you're first exploring DDD with Spring.

The core issue lies in the fact that Spring Data, by default, tends to assume your entities are also your domain objects. This can quickly lead to an anemic domain model, where your entities end up with persistence-related concerns polluting your core business logic. DDD, on the other hand, advocates for a rich domain model, encapsulated and independent of infrastructural details like persistence. To bridge this gap, we need to consciously enforce a separation between domain models and JPA entities.

The key to maintaining this separation is to define separate classes for your domain models and JPA entities, and then map between them. Let's start with the domain model. This model should focus solely on the business logic and rules of your application, devoid of any JPA annotations. For example, in our old e-commerce system, we had a 'Product' domain model:

```java
// Domain Model: Product.java
package com.example.domain;

import java.math.BigDecimal;
import java.util.UUID;

public class Product {

    private UUID productId;
    private String name;
    private String description;
    private BigDecimal price;

    public Product(UUID productId, String name, String description, BigDecimal price) {
        this.productId = productId;
        this.name = name;
        this.description = description;
        this.price = price;
    }

    // Getters (and setters if needed, but try to minimize setters)

    public UUID getProductId() {
      return productId;
    }

    public String getName() {
      return name;
    }

    public String getDescription() {
      return description;
    }

    public BigDecimal getPrice() {
      return price;
    }

    public void applyDiscount(BigDecimal discountPercentage) {
        if (discountPercentage.compareTo(BigDecimal.ZERO) <= 0 || discountPercentage.compareTo(BigDecimal.ONE) > 0) {
             throw new IllegalArgumentException("Discount percentage must be between 0 and 1");
        }
        this.price = this.price.multiply(BigDecimal.ONE.subtract(discountPercentage));
    }
    // Business logic methods related to product
}
```

Notice how this `Product` class doesn't know anything about how it’s going to be persisted. It's just concerned with its own internal state and logic.

Now, let's look at the corresponding JPA entity, which we can store in a `persistence` package:

```java
// JPA Entity: ProductEntity.java
package com.example.persistence;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.math.BigDecimal;
import java.util.UUID;

@Entity
@Table(name = "products")
public class ProductEntity {
    @Id
    private UUID productId;
    private String name;
    private String description;
    private BigDecimal price;

    public ProductEntity() {
       // JPA needs a default constructor.
    }

    public ProductEntity(UUID productId, String name, String description, BigDecimal price) {
        this.productId = productId;
        this.name = name;
        this.description = description;
        this.price = price;
    }


    // Getters and setters
    public UUID getProductId() {
      return productId;
    }

    public void setProductId(UUID productId) {
      this.productId = productId;
    }

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public String getDescription() {
      return description;
    }

    public void setDescription(String description) {
      this.description = description;
    }

    public BigDecimal getPrice() {
      return price;
    }

    public void setPrice(BigDecimal price) {
      this.price = price;
    }
}
```

This `ProductEntity` class is a representation of your database table. It carries all of your JPA annotations and is responsible for how the data is persisted. The key thing is that it doesn’t contain any business logic.

The next crucial step is mapping between these two classes. We can achieve this using dedicated mapper classes or utility methods. Here's an example using a simple mapper class:

```java
// Mapper class: ProductMapper.java
package com.example.mapper;

import com.example.domain.Product;
import com.example.persistence.ProductEntity;

import java.util.UUID;

public class ProductMapper {

    public static Product toDomain(ProductEntity entity) {
       if (entity == null) {
            return null;
        }
       return new Product(entity.getProductId(), entity.getName(), entity.getDescription(), entity.getPrice());
    }

    public static ProductEntity toEntity(Product domain) {
      if(domain == null) {
        return null;
      }
      return new ProductEntity(domain.getProductId(), domain.getName(), domain.getDescription(), domain.getPrice());
    }
}
```

This `ProductMapper` provides methods to transform a `ProductEntity` into a `Product` and vice versa. It encapsulates all the mapping logic in one place.

Now, how does Spring Data fit into all of this? We use Spring Data repositories to manage our JPA entities, and the mapper is used in our service layer:

```java
// ProductRepository.java
package com.example.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.UUID;

public interface ProductRepository extends JpaRepository<ProductEntity, UUID> {
}
```

And, a sample service implementation that uses the mapper:

```java
// ProductService.java
package com.example.service;


import com.example.domain.Product;
import com.example.mapper.ProductMapper;
import com.example.persistence.ProductEntity;
import com.example.persistence.ProductRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;
import java.util.UUID;

@Service
public class ProductService {
  @Autowired
  private ProductRepository productRepository;

  public Optional<Product> findProductById(UUID productId) {
      Optional<ProductEntity> entityOptional = productRepository.findById(productId);
      return entityOptional.map(ProductMapper::toDomain);
  }

  public Product saveProduct(Product product) {
      ProductEntity entity = ProductMapper.toEntity(product);
      ProductEntity savedEntity = productRepository.save(entity);
      return ProductMapper.toDomain(savedEntity);
  }

  // Other service methods
}
```

In this service, the repository methods from Spring Data are used to interact with the `ProductEntity` objects. The `ProductMapper` class is used to convert between the returned entities and our domain objects.

By adopting this approach, our domain model remains pure, focused on business logic, and free from database concerns. This enhances the testability, maintainability, and overall robustness of our application. It also aligns with the core principles of DDD. This is important because changing persistence mechanisms in the future won't force changes in your core domain logic.

For further study, I would recommend reading "Implementing Domain-Driven Design" by Vaughn Vernon for a detailed understanding of DDD concepts and patterns. Also, the official Spring Data documentation provides valuable insights into using Spring Data with JPA and other data stores. Furthermore, consider exploring articles on the "hexagonal architecture" as an approach to enforcing separation of concerns, which complements DDD and how it works with Spring Data, as it does in this example, and provides another perspective on the topic. It is not enough to implement it; it is important to understand *why* you implement it, which DDD, architectural patterns, and great technical books such as these, will make clear.

By separating your entities from your domain models, and creating a solid mapping strategy, you can use Spring Data effectively in a DDD-driven application. This has been my approach for years, and it has consistently led to more maintainable and adaptable systems.
