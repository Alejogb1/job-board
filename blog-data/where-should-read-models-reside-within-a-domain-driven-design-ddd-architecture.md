---
title: "Where should read models reside within a Domain-Driven Design (DDD) architecture?"
date: "2024-12-23"
id: "where-should-read-models-reside-within-a-domain-driven-design-ddd-architecture"
---

Let's tackle this one; it’s a recurring theme, and I’ve certainly seen it play out in various ways throughout my years working with domain-driven architectures. The location of read models within a DDD architecture isn't a one-size-fits-all scenario; it often hinges on the specific needs and constraints of the application you're building. Now, when we talk about DDD, we're thinking about a software design philosophy, rather than a prescriptive methodology. So, flexibility is key.

Specifically, the placement of read models usually involves a careful consideration of separation of concerns, performance requirements, and the overall architecture you've chosen. Let's start by defining what a read model is in a DDD context, just to ensure we're all on the same page. Essentially, a read model is a data structure optimized for querying and display purposes, distinct from your write model (the aggregate that encapsulates business logic). It's designed to cater to specific UI needs and optimize for read performance rather than adhering to the intricate rules of your domain.

Now, there isn’t a definitive 'correct' location, but I've observed three primary patterns emerging in practice, each with its advantages and trade-offs:

1. **Within the Application Layer, Close to the Query Handlers:** This approach positions read models as a direct concern of the application layer, sitting close to the query handlers that are responsible for retrieving data. In this scenario, the application service will fetch or build read models directly, based on data obtained from the persistence layer. This is often the simplest to implement, particularly in smaller applications, or where the read requirements aren't too complex. It also keeps the code involved in the read process within a singular location.

   Here’s a simple example in Python, illustrating how a query handler might interact directly with a data mapper and a read model:

   ```python
   from dataclasses import dataclass
   from typing import List, Dict

   @dataclass
   class ProductSummary:
       product_id: str
       name: str
       price: float

   class ProductDataMapper:
       def find_all(self) -> List[Dict]:
           #Simulating database interaction
           return [{"id": "1", "name": "Laptop", "price": 1200.0}, {"id": "2", "name": "Mouse", "price": 25.0}]


   class ProductQueryHandler:
       def __init__(self, data_mapper: ProductDataMapper):
           self.data_mapper = data_mapper

       def handle(self) -> List[ProductSummary]:
           raw_data = self.data_mapper.find_all()
           return [ProductSummary(product_id=item['id'], name=item['name'], price=item['price']) for item in raw_data]


   mapper = ProductDataMapper()
   handler = ProductQueryHandler(mapper)
   summaries = handler.handle()

   for summary in summaries:
      print(f"Product ID: {summary.product_id}, Name: {summary.name}, Price: {summary.price}")
   ```

   In this case, `ProductSummary` is our read model, and `ProductQueryHandler` acts as an application service that directly constructs the view from data it retrieves. It’s straightforward, and minimizes the introduction of additional layers.

2.  **As a Dedicated Service or Project, Separated from the Application Layer:** As applications grow in complexity, especially in microservices architectures, it becomes beneficial to segregate the read model infrastructure into its own service, or module. This separation provides several advantages. It decouples the read operations from your domain and application logic. This allows you to scale your read infrastructure independently, and it also allows for using different databases more suited for specific query requirements (e.g., a document database like MongoDB for complex aggregations). When we’ve adopted this model, we’ve often found it beneficial to use technologies better tailored for read concerns, such as full text search engines or key-value stores.

     Here's a Java example, illustrating a separation of concerns between a domain layer and a read model service utilizing a message queue for updates:

   ```java
   import java.util.UUID;

   //Domain Model (Simplified)
   class Order {
     private UUID orderId;
     private String customerId;
     private String status;

     public Order(UUID orderId, String customerId) {
        this.orderId = orderId;
        this.customerId = customerId;
        this.status = "PENDING";
     }
     public void setStatus(String status) {
        this.status = status;
     }
       public UUID getOrderId() { return orderId; }
       public String getCustomerId() { return customerId;}
       public String getStatus() { return status;}
   }

   // Event for Order status change
    class OrderStatusChanged {
        public UUID orderId;
        public String newStatus;
    public OrderStatusChanged(UUID orderId, String newStatus) {
            this.orderId = orderId;
            this.newStatus = newStatus;
        }
    }

   // Read Model
    class OrderView {
        private UUID orderId;
        private String customerId;
        private String status;
        public OrderView(UUID orderId, String customerId, String status) {
            this.orderId = orderId;
            this.customerId = customerId;
            this.status = status;
        }

        public void setStatus(String status) {
            this.status = status;
        }

         @Override
         public String toString() {
          return "OrderView{" +
              "orderId=" + orderId +
              ", customerId='" + customerId + '\'' +
              ", status='" + status + '\'' +
              '}';
        }
   }

   // Read model service
    class OrderReadService {
         private final HashMap<UUID, OrderView> orderViews = new HashMap<>();
        public void handleStatusChange(OrderStatusChanged event) {
             OrderView view = orderViews.get(event.orderId);
             if(view != null) {
                view.setStatus(event.newStatus);
             } else {
                 System.out.println("Order " + event.orderId + " not found in read model");
             }
        }

        public void createOrderView(Order order){
             OrderView newView = new OrderView(order.getOrderId(), order.getCustomerId(), order.getStatus());
             orderViews.put(order.getOrderId(), newView);
        }

        public OrderView getOrderView(UUID orderId) {
            return orderViews.get(orderId);
        }

    }
   // Mock Event Bus
    class EventBus {
        private final java.util.List<java.util.function.Consumer<Object>> subscribers = new java.util.ArrayList<>();
        public void publish(Object event) {
            for (var subscriber : subscribers) {
                subscriber.accept(event);
            }
        }

        public void subscribe(java.util.function.Consumer<Object> subscriber){
          subscribers.add(subscriber);
        }
    }


   public class Main {
        public static void main(String[] args) {

            EventBus eventBus = new EventBus();

            OrderReadService orderReadService = new OrderReadService();

            eventBus.subscribe(event -> {
                if (event instanceof OrderStatusChanged) {
                   orderReadService.handleStatusChange((OrderStatusChanged) event);
                }
            });

            UUID orderId = UUID.randomUUID();
            Order order = new Order(orderId, "customer1");

            orderReadService.createOrderView(order);

            eventBus.publish(new OrderStatusChanged(orderId, "SHIPPED"));

            System.out.println(orderReadService.getOrderView(orderId));
        }
   }

   ```
   Here, the `OrderView` represents a specific read model, and `OrderReadService` handles updates through the `EventBus`, decoupled from the application service responsible for business logic.  This is a CQRS (Command Query Responsibility Segregation) pattern that can provide high scalability for read operations.

3. **Within a Dedicated Data Store Optimized for Queries (Data Materialization):** This third approach is not just about a separate service but often a different *type* of data store. Here, we use specialized databases that are particularly efficient for querying, such as NoSQL databases designed for fast retrieval of data based on various dimensions (think document, graph, or key-value databases). The read model data is essentially 'materialized' into these data stores, allowing for fast, optimized retrieval. This strategy usually involves a process to transform data from your domain model into the read model format. Techniques such as eventual consistency and event sourcing typically come into play, as updates to the read store may lag behind domain updates.

   Here's a simplified example illustrating this concept using node.js and a simulated data storage:
   ```javascript
   // Domain object
   class User {
     constructor(id, name, email) {
       this.id = id;
       this.name = name;
       this.email = email;
     }
   }
   // Read Model projection, simulates data store
   class UserView {
       constructor(){
         this.data = {};
       }

      update(id, name, email){
         this.data[id] = { name, email };
       }

      find(id){
        return this.data[id] || null;
      }

     findAll() {
          return Object.values(this.data);
      }
   }
   // Domain event handler to update read model
   class UserReadModelHandler {
       constructor(userView){
         this.userView = userView;
       }
       handleUserCreated(event){
           this.userView.update(event.id, event.name, event.email)
       }
      handleUserUpdated(event){
        this.userView.update(event.id, event.name, event.email);
      }

   }

   // Example usage
   const userView = new UserView();
   const readModelHandler = new UserReadModelHandler(userView);

   const user1 = new User('123', 'Alice Smith', 'alice@example.com');
   readModelHandler.handleUserCreated(user1);

   const user2 = new User('456', 'Bob Jones', 'bob@example.com');
   readModelHandler.handleUserCreated(user2);
   const updatedUser2 = new User('456', 'Robert Jones', 'robert@example.com');
   readModelHandler.handleUserUpdated(updatedUser2);

   console.log('Find user 1: ', userView.find('123'))
   console.log('Find user 2:', userView.find('456'))
   console.log('All users: ', userView.findAll());
   ```
   Here,  `UserView` is the read model projection, updated by the `UserReadModelHandler`. It's a simplified example to highlight the concept of projection.

In my experience, these three patterns, or combinations thereof, cover most scenarios when determining where read models should reside. Ultimately, the 'best' place to put your read models is going to depend on the scale and complexity of your system. If you're just getting started or have relatively simple read requirements, going with pattern one—keeping them within the application layer near your query handlers—is often sufficient. However, as your application evolves, moving toward the separate service/data store approach can be crucial for scalability and maintainability.

If you're looking to dive deeper into the concepts I've described, I'd recommend reviewing "Implementing Domain-Driven Design" by Vaughn Vernon. Also, the materials surrounding the CQRS pattern, such as those discussed in Martin Fowler's writings and talks, provide an understanding of the benefits and practical implementation details around read models. Additionally, looking into specific database strategies outlined in books about specific databases, such as 'MongoDB: The Definitive Guide,' can be invaluable when considering a specialized data store for your read side. These resources provide the detailed understanding necessary to make informed decisions about the architecture of your system.
