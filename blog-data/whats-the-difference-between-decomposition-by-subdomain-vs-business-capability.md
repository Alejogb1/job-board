---
title: "What's the difference between Decomposition by subdomain vs business capability?"
date: "2024-12-16"
id: "whats-the-difference-between-decomposition-by-subdomain-vs-business-capability"
---

Okay, let's tackle this. I've spent a fair amount of time knee-deep in architectural debates, and the distinction between decomposing systems by subdomain versus business capability is something that often comes up, particularly when you're scaling and trying to build manageable, maintainable software. It’s a critical decision point with serious implications for team structure and overall architecture health.

The core difference, at a high level, is the focus: subdomain decomposition aligns itself with the *problem space*, while business capability decomposition aligns more with the *solution space*. This distinction, though seemingly subtle, leads to vastly different architectural outcomes.

Let's start with subdomain decomposition. This approach, rooted in domain-driven design (ddd), focuses on identifying the core areas of expertise or ‘subdomains’ within a larger business domain. A domain, in this context, represents the overall sphere of activity or knowledge. Subdomains, therefore, are specific areas within that domain. For example, in an e-commerce system, you might have subdomains like 'product catalog,' 'order management,' 'payment processing,' and 'customer profiles.' The key is that these are areas of *understanding*, representing different specialized parts of the business problem. Each subdomain aims to be self-contained, with its own language (ubiquitous language in ddd parlance), models, and, ideally, its own dedicated team. This separation, based on the business *problem*, makes it easier for a team to focus its knowledge and expertise. Think of it as breaking down a complex problem into smaller, more manageable chunks, each with a clear area of responsibility and a language specific to the context.

On the other hand, business capability decomposition focuses on identifying what a business *does*. Capabilities are the things an organization performs to achieve its goals. They're higher-level, often crossing over multiple subdomains. In the e-commerce example, a business capability could be "manage customer orders," which would likely touch across the 'customer profiles,' 'product catalog,' and 'order management' subdomains. Another business capability might be "perform financial transactions" which might interact with ‘payment processing’ as well as ‘order management’ and even ‘customer profiles’ for invoicing and past purchasing history. These capabilities are usually expressed in active voice – e.g., "schedule appointments," "process payments," "manage inventory," etc. Decomposing along these lines results in modules or services focused on the *actions* the business performs, not necessarily the areas of specialized knowledge. This often results in more vertical slices of functionality.

Now, the real world is rarely perfectly textbook. In my experience, starting with subdomain decomposition often proves advantageous when you are building a greenfield system or re-architecting a legacy monolith. It encourages a deep understanding of the business, fosters domain expertise in teams, and promotes better alignment between software and business processes. The benefits, however, tend to become more complicated when these subdomains require a high level of coordination and depend on tightly coupled, synchronous calls across the network or internal application layers. In situations like that, you might find that focusing purely on subdomains does not result in ideal architectural outcomes. Instead you may end up with a distributed monolith where services are highly dependent upon each other.

Let's look at some code examples. Imagine we're building a very simplified system for booking appointments:

**Example 1: Subdomain Focused Approach (using Python)**

```python
# appointments/models.py - Part of the 'appointments' subdomain

class Appointment(object):
    def __init__(self, client_id, date, time, service):
        self.client_id = client_id
        self.date = date
        self.time = time
        self.service = service

    def __repr__(self):
        return f"Appointment for {self.client_id} on {self.date} at {self.time}"

# clients/models.py - Part of the 'clients' subdomain

class Client(object):
    def __init__(self, client_id, name, contact_info):
        self.client_id = client_id
        self.name = name
        self.contact_info = contact_info

    def __repr__(self):
      return f"Client: {self.name}, {self.contact_info}"
```

In this extremely basic example, the 'appointments' and 'clients' subdomains are clearly separated, each with their own models and logic. The *problem space* and domain of concern is well understood in this example and each of the models, or data entities, has a clear purpose and area of concern.

**Example 2: Business Capability Focused Approach (using Python)**

```python
# booking_service.py - A service handling the 'book appointment' capability

class BookingService(object):
    def __init__(self):
        self.appointments = []
        self.clients = []

    def add_client(self, client_id, name, contact_info):
      self.clients.append({"client_id": client_id, "name": name, "contact_info": contact_info})

    def book_appointment(self, client_id, date, time, service):
        # Some validation or logic might go here
        if not any(client['client_id'] == client_id for client in self.clients):
           raise Exception("Client not found")
        self.appointments.append({"client_id": client_id, "date": date, "time": time, "service": service})
        return True

    def get_appointments(self):
        return self.appointments
```

Here, the `BookingService` is focused on a specific capability: 'book appointment'. It encapsulates logic that touches on data related to both appointments and clients. Note that it handles the concept of appointments and clients within a single service, therefore focusing on a particular use case across subdomains. The solution space, or business requirement, is prioritized over separation of concerns based on the problem domain.

**Example 3: Hybrid Approach (using Python)**

```python
# appointments/appointment_service.py
class AppointmentService(object):
    def __init__(self, client_service):
        self.client_service = client_service # dependency injected to avoid tight coupling
        self.appointments = []

    def book_appointment(self, client_id, date, time, service):
        # Validate client through client_service (assume external dependency)
        if not self.client_service.client_exists(client_id):
           raise Exception("Client not found.")
        self.appointments.append({"client_id": client_id, "date": date, "time": time, "service": service})
        return True
    def get_appointments(self):
        return self.appointments

# clients/client_service.py
class ClientService(object):
   def __init__(self):
      self.clients = []

   def add_client(self, client_id, name, contact_info):
      self.clients.append({"client_id": client_id, "name": name, "contact_info": contact_info})

   def client_exists(self, client_id):
      return any(client["client_id"] == client_id for client in self.clients)

```

This demonstrates a hybrid approach where we have services focusing on specific subdomains (ClientService and AppointmentService), but the AppointmentService relies on the ClientService to achieve specific business capabilities like booking an appointment. It promotes modularity and encapsulation, where a service is responsible for its own domain, while at the same time, achieving specific business objectives. This kind of dependency injection based approach using modular services and interfaces is how teams can achieve greater agility and reduced team cognitive load while delivering useful business value to the end user.

In practice, a purely subdomain- or capability-driven approach rarely works in isolation. What I've found most effective, is a *blended* approach. Starting with subdomain decomposition to establish a firm grasp of the problem space, and then refactoring or creating services that are capability-focused where it makes sense. It often becomes evident that specific capabilities require logic that cuts across several subdomains, which are best captured in a standalone capability-based service. It is critical to maintain cohesion (high intra-module coupling) and minimize coupling (low inter-module coupling) wherever possible for any approach used.

If you’re looking to deepen your understanding, I’d recommend starting with Eric Evans' "Domain-Driven Design: Tackling Complexity in the Heart of Software." Also, Martin Fowler’s "Patterns of Enterprise Application Architecture" is excellent for understanding different architectural patterns that relate to different types of decomposition. For a more modern perspective, check out Sam Newman’s “Building Microservices: Designing Fine-Grained Systems”, which covers patterns that often emerge when designing complex, distributed systems. You’ll find these resources extremely helpful in further exploring this critical area of software architecture.
