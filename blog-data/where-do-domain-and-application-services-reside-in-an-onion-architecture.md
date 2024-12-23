---
title: "Where do domain and application services reside in an onion architecture?"
date: "2024-12-23"
id: "where-do-domain-and-application-services-reside-in-an-onion-architecture"
---

Let’s unpack the question of where domain and application services reside within an onion architecture. I’ve spent a fair bit of time implementing this pattern across various projects, and I’ve seen it work beautifully when done well and cause significant headaches when not. So, let's approach this from a practical, experienced perspective.

The onion architecture, at its core, is about decoupling. It’s a layered architecture where dependencies point inward, towards the core of the application, rather than outward. This principle is crucial. Think of it like the layers of an onion – the core is at the center, and each layer wraps around it, relying only on the layers below it. This arrangement is designed to isolate the core domain logic from the technical implementation details.

Now, to your specific question about domain and application services. Domain services reside at the very center of the onion, within what is often termed the ‘domain’ or ‘entities’ layer. These services encapsulate the core business logic and rules of the application. They operate on domain entities – the fundamental objects representing concepts from the business domain (like a `Customer`, an `Order`, or a `Product`). These domain services shouldn’t know anything about *how* data is stored, *how* user interfaces are rendered, or *how* external systems are invoked. They are purely focused on business logic. We aim for these services to be as close as possible to the actual business requirements, making them very stable.

Application services, on the other hand, sit in the layer just outside the domain layer, often referred to as the ‘application’ layer or ‘use cases’ layer. They orchestrate and coordinate the execution of domain logic for specific application requirements. These services do not contain any business rules themselves, but rather they act as an intermediary between the user interface (or another external entry point) and the domain. They are designed to take requests from the outside, validate input, invoke operations on domain services, and return data structured for the outside world. They can also encapsulate transactions or coordinate operations across multiple domain services.

It's important to highlight that the application layer has knowledge of the domain, but not the other way around. This dependency direction is key to maintaining the separation of concerns and ensuring that changes in the application layer do not propagate into the domain layer. We aim to keep the domain model pure and focused.

Let's illustrate this with a simple example using Python. Imagine we're building a basic e-commerce application. We have `Customer`, `Order`, and `Product` as our core domain entities.

**Snippet 1: Domain Service**

```python
# domain/customer_service.py
class Customer:
    def __init__(self, customer_id, name, email):
        self.customer_id = customer_id
        self.name = name
        self.email = email

class CustomerService:
    def __init__(self, customer_repository): #Dependency Inversion
        self.customer_repository = customer_repository

    def create_customer(self, name, email):
        if not name or not email:
            raise ValueError("Name and email are required")
        if self.customer_repository.get_customer_by_email(email):
            raise ValueError("Customer with this email already exists")
        
        new_customer = Customer(self.customer_repository.get_next_id(), name, email)
        self.customer_repository.save_customer(new_customer)
        return new_customer

    def get_customer(self, customer_id):
      customer = self.customer_repository.get_customer(customer_id)
      if not customer:
        raise ValueError(f"Customer with id {customer_id} not found")
      return customer


#domain/customer_repository.py (abstraction)
from abc import ABC, abstractmethod

class CustomerRepository(ABC):
  @abstractmethod
  def get_customer(self, customer_id):
    pass

  @abstractmethod
  def get_customer_by_email(self, email):
    pass

  @abstractmethod
  def save_customer(self, customer):
    pass
  
  @abstractmethod
  def get_next_id(self):
    pass


```

Here, `CustomerService` is a domain service. It handles the creation of new customers, ensuring business logic, like not allowing duplicate email addresses, is enforced. Crucially, note the injected `customer_repository` which is an interface. This is a critical application of the dependency inversion principle which allows us to change the persistence implementation without modifying the domain logic.

**Snippet 2: Application Service**

```python
# application/customer_app_service.py
from domain.customer_service import CustomerService
from domain.customer_repository import CustomerRepository # Import the interface
from typing import Optional

class CustomerAppService:
    def __init__(self, customer_repository: CustomerRepository): #dependency injection via the interface
        self.customer_service = CustomerService(customer_repository) #injecting the domain service

    def register_new_customer(self, name, email) -> Optional[dict]:
        try:
            customer = self.customer_service.create_customer(name, email)
            return {"customer_id": customer.customer_id, "name": customer.name, "email": customer.email}
        except ValueError as e:
            print (f"Error registering customer: {e}")
            return None

    def fetch_customer(self, customer_id: int) -> Optional[dict]:
      try:
        customer = self.customer_service.get_customer(customer_id)
        return {"customer_id": customer.customer_id, "name": customer.name, "email": customer.email}
      except ValueError as e:
        print(f"Error fetching customer: {e}")
        return None


```

The `CustomerAppService` is an application service. It receives requests from the outside, calls `CustomerService` to perform actions, and transforms the results as needed to be presented to an external client (in this case a simplified dictionary). This example also implements dependency injection, using the `CustomerRepository` interface and injecting the dependency into the constructor, further illustrating best practice of separation of concerns. It doesn't contain any of the business rules, delegating all business rule logic to the domain layer.

**Snippet 3: Concrete Implementation of Customer Repository and Usage**

```python
#infrastructure/customer_repository_impl.py
from domain.customer_repository import CustomerRepository
class InMemoryCustomerRepository(CustomerRepository): #concrete implementation of the abstraction
  def __init__(self):
    self.customers = {}
    self.next_id = 1
  
  def get_customer(self, customer_id):
    return self.customers.get(customer_id)
  
  def get_customer_by_email(self, email):
    for customer in self.customers.values():
        if customer.email == email:
            return customer
    return None
  
  def save_customer(self, customer):
      self.customers[customer.customer_id] = customer
      
  def get_next_id(self):
      next_id = self.next_id
      self.next_id += 1
      return next_id
    
# entrypoint.py
from application.customer_app_service import CustomerAppService
from infrastructure.customer_repository_impl import InMemoryCustomerRepository
#setup of dependencies 
customer_repository = InMemoryCustomerRepository()
customer_app_service = CustomerAppService(customer_repository)


#example of usage
new_customer = customer_app_service.register_new_customer("John Doe", "john.doe@example.com")
if new_customer:
  print(f"Created customer: {new_customer}")
else:
  print("Failed to register customer")

fetched_customer = customer_app_service.fetch_customer(1)
if fetched_customer:
  print (f"Fetched customer: {fetched_customer}")
else:
    print("Failed to fetch customer")
```

Here we see the implementation of a concrete repository, and the overall example in action. Notice how the `InMemoryCustomerRepository` implements the `CustomerRepository` abstraction and is passed to the application service which in turn passes this to the domain service, fulfilling the dependency injection. This promotes a very modular and testable design.

In real-world applications, repositories would connect to a database, use an API, or some other form of persistent storage. This example demonstrates the separation of concerns at the architecture level.

For further study, I’d recommend the book *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans. This is a foundational text for understanding domain modeling. For a deep dive into architectural patterns, *Patterns of Enterprise Application Architecture* by Martin Fowler is invaluable. Furthermore, *Clean Architecture* by Robert C. Martin explains the core tenets of the onion architecture (although he uses a slightly different name) and provides clear examples. Also, *Implementing Domain-Driven Design* by Vaughn Vernon is helpful for practical implementation. These resources should solidify your understanding and provide a robust background for building well-structured applications using the onion architecture.
