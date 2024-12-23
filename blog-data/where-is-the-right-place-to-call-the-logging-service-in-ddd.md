---
title: "Where is the right place to call the logging service in DDD?"
date: "2024-12-23"
id: "where-is-the-right-place-to-call-the-logging-service-in-ddd"
---

,  I’ve certainly seen my share of logging debates, particularly when domain-driven design (ddd) enters the picture. It's not as straightforward as just dumping log statements everywhere, especially if you're aiming for a clean, maintainable architecture. The 'right' place, as always, is nuanced and depends on what exactly you're logging and why. From my experience working on a large e-commerce platform—let's call it "GlobalMart"—we really grappled with this and found a few effective strategies.

The core issue is separating concerns. We don’t want logging logic bleeding into our domain model. Domain entities should be focused solely on business logic, unaware of infrastructure concerns like logging. Similarly, application services should orchestrate business workflows and not be bogged down with logging details. Instead, logging should generally be treated as a cross-cutting concern, managed by infrastructure-level components, but invoked strategically.

The key is understanding the *purpose* of logging. Is it for auditing, debugging, or monitoring? Different needs imply different placements. Let's explore these through the lens of our fictional experience at GlobalMart.

**1. Logging within Infrastructure Layers (e.g., Repository Operations):**

For anything related to data persistence, repositories were a crucial point for logging. We needed visibility into how data was being retrieved or saved. Think of operations like fetching a customer from the database, or updating a product catalog record. This is where infrastructure components manage interactions with external systems, and errors here can indicate issues with the data layer, not the core business logic.

For instance, our `ProductRepository` had methods like `getProductById` and `updateProduct`. We wrapped these calls with logging that looked like this (using a hypothetical logging framework).

```python
# Python example
class ProductRepository:
    def __init__(self, logger, database_connection):
        self.logger = logger
        self.db = database_connection

    def get_product_by_id(self, product_id):
        self.logger.info(f"Fetching product with id: {product_id}")
        try:
            product = self.db.query("SELECT * FROM products WHERE id = %s", (product_id,))
            self.logger.debug(f"Product retrieved: {product}")
            return product
        except Exception as e:
            self.logger.error(f"Error fetching product with id {product_id}: {e}")
            raise

    def update_product(self, product):
        self.logger.info(f"Updating product: {product.id}")
        try:
             self.db.execute("UPDATE products SET ...", (product.name, product.price,...)) # Hypothetical update
             self.logger.debug(f"Product updated successfully: {product.id}")
        except Exception as e:
            self.logger.error(f"Error updating product {product.id}: {e}")
            raise

```

Here, we log before and after the database calls, including detailed error messages if anything fails. This helped us quickly diagnose issues with the database or data mapping. Notice that the repository itself doesn't care *how* the logging occurs; it's injected with a `logger` instance via dependency injection, making it easily swappable if we wanted a different logging implementation. This is very important for testing and maintainability.

**2. Logging within Application Services (For Business Workflow Information):**

Application services orchestrate domain logic. Logging here often means tracking the execution of workflows and the actions of users within the system. We didn't log individual method calls within entities but focused on the significant steps within use cases.

Consider a use case where a user adds an item to their shopping cart. The application service would manage the entire flow, from validating the item to persisting the changes. The logging would center on key steps within this operation, like adding the item to the cart, applying any promotions, and any errors.

```java
// Java Example

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ShoppingCartService {

    private static final Logger logger = LoggerFactory.getLogger(ShoppingCartService.class);

    private CartRepository cartRepository;
    private ProductRepository productRepository;


    public ShoppingCartService(CartRepository cartRepository, ProductRepository productRepository) {
        this.cartRepository = cartRepository;
        this.productRepository = productRepository;
    }

    public void addItemToCart(String userId, String productId, int quantity) {
        logger.info("User {} attempting to add item {} (qty: {}) to cart.", userId, productId, quantity);
        try {
            Product product = productRepository.getProductById(productId);
            if (product == null) {
              logger.warn("Product with ID {} not found.", productId);
              throw new IllegalArgumentException("Product not found");
            }

            Cart cart = cartRepository.findOrCreateCart(userId);
            cart.addItem(product, quantity); // Domain operation
            cartRepository.save(cart);
            logger.info("Item {} added successfully to cart for user {}.", productId, userId);

        } catch (IllegalArgumentException e) {
            logger.error("Error adding item to cart: Product not found.", e);
            throw e;

        } catch (Exception e) {
             logger.error("Unexpected error adding item to cart.", e);
             throw e; // Or handle appropriately
        }
    }
}

```

Here the logger (in this case slf4j) provides informative log lines before and after key actions, including detailed error messages. This helped us monitor the flow of user actions, track performance of particular workflows and debug if a business process was having problems. By logging at a higher level, within service actions, we are not adding overhead or noisy details to lower levels where they're not as needed. Note again we use an abstraction for the logger.

**3. Logging at the Edge (API Entry Points, Event Handlers):**

For requests entering our system (like http requests at our api endpoints) or asynchronous events, logging is crucial to understand what triggers actions within the application and what goes out. This is a very useful boundary to log information for both auditing and debugging. Our web api controllers, or our event handlers, would also include their logging.

```csharp
// C# example
using Microsoft.Extensions.Logging;

public class OrderController
{
    private readonly ILogger<OrderController> _logger;
    private readonly IOrderService _orderService;

    public OrderController(ILogger<OrderController> logger, IOrderService orderService)
    {
        _logger = logger;
        _orderService = orderService;
    }

    [HttpPost("orders")]
    public async Task<IActionResult> CreateOrder([FromBody] OrderDto orderDto)
    {
        _logger.LogInformation("Received request to create order.");
        try
        {
            var order = await _orderService.CreateOrder(orderDto);
             _logger.LogInformation($"Order created with id {order.Id}");
             return Ok(order);
        } catch (Exception ex) {
             _logger.LogError(ex, "Error creating order.");
             return BadRequest("Failed to create order"); // Or return a suitable error
        }

    }
}
```

In this c# example, we use a framework level logger injected as an abstraction. Log messages record the entry of the request and log if any exception is thrown so we can debug it. When handling events, we employed a similar approach, logging the start and end of event handling as well as any errors that arose.

**Important Considerations and Resources:**

*   **Structured Logging:** Avoid simply printing strings; log structured data (e.g., json) to enable easier querying and analysis. In production environments, it's immensely helpful to query logs based on fields, rather than having to parse text. Libraries like `logstash` or `fluentd` are valuable for this.
*   **Log Levels:** Use appropriate log levels (debug, info, warn, error, etc.) to control verbosity and ensure you're not flooded with unneeded messages, especially in production.
*   **Correlation IDs:** Use correlation ids to track the progress of a request as it moves through multiple services. This makes it much easier to understand the full request lifecycle during debugging or analysis. A unique identifier can be generated at the entry point of an application and then passed to other services.
*   **Contextual Data:** Include relevant contextual information in logs. For example, the user id, product id, or order id are valuable.

For deeper reading, I’d recommend *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans; it lays the foundation for why keeping domain concerns separate is paramount. Also, *Patterns of Enterprise Application Architecture* by Martin Fowler is excellent for understanding patterns around infrastructure and data access. Additionally, exploring books specific to your chosen logging framework (e.g., slf4j for Java, `Microsoft.Extensions.Logging` for c#, or Python logging module documentation) is highly valuable for understanding advanced configuration options.

In essence, logging within DDD shouldn't be an afterthought but a well-planned component of the architecture. It's about placing the right logging *strategically* at the boundaries of various layers and where key business events occur to gain valuable insights, without polluting domain logic with technical concerns. In our "GlobalMart" days, we certainly learned this the hard way before landing on these patterns.
