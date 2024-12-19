---
title: "How to do ddd: remove an entity (microservice) dependent on another (microservice)?"
date: "2024-12-15"
id: "how-to-do-ddd-remove-an-entity-microservice-dependent-on-another-microservice"
---

alright, so you're looking at the classic distributed systems headache, right? dealing with interdependent microservices, specifically removing one that's a dependency of another. i've been there, more times than i care to remember. it’s never as simple as just flipping a switch. you can't just yank out a component, especially not one that's acting as a dependency for another, without consequences. let’s talk about how to approach this using, what i understand as your preferred domain-driven design, ddd, methodology in mind.

from my own experience, i had a project, oh, maybe seven years ago, where we built a system for processing financial transactions. we had a service for user accounts (user-service) and a separate one for handling payments (payment-service). payment-service, obviously, was heavily reliant on user-service for, well, user details. then, the business went through a restructuring, and suddenly, we needed to integrate with an external payment provider which handled the user creation by itself and we needed to get user-service out of the payment-service picture. it was a nightmare, i tell you.

so, here's the thing. ddd encourages us to think in terms of bounded contexts. each microservice should ideally map to its own bounded context with a well-defined domain model. the problem arises when these contexts have tight coupling. in your case, it's the dependency of the microservice you want to remove on another.

you basically have a few options, none of which are magic bullets. i'm going to try and give you the practical advice on how to approach this. first, we analyze and understand.

*   **analysis and understanding:** you need to map out the exact nature of the dependency. what data does the dependent service need from the service you want to remove? what api calls are being made? start with a good analysis of your system to find those hard dependencies. get all the data needed to understand how the two services are talking to each other and what is the minimum information required for the payment-service to work. you want to achieve loose coupling. i bet you want to avoid a situation where you have to change the payment service again once the user service is removed. that also means you need to get a picture of where the data is needed and its requirements. do a proper discovery to really understand the data flow.

*   **identifying the domain:** i'd suggest looking at domain events that each service might be producing. understand what the aggregate roots are and what are their domain boundaries, and how the communication between them is being done. that is a good start for a good transition process. that way, you can start decoupling the components.

once you have that clear, you can move on.

**option 1: the 'api abstraction' approach (the most common case)**

this is my go-to when the dependency isn’t too deeply embedded. basically, you introduce an abstraction layer between the dependent service and the service you want to remove.

1.  **create an interface or abstraction in the dependent microservice:** this interface should define the data the dependent service requires from the now retiring dependency. so in our example, the `payment-service` would now have an interface that provides user information, no matter where it comes from.

2.  **implement the interface with the existing service:** initially, you implement this interface using the existing service. in our example, the `user-service`.

3.  **implement the interface with the new external service (or by the dependent service):** then, you switch the implementation over to get the data from the external payment provider. or, the `payment-service` can retrieve this information itself, if it's possible and less complex than calling an external service.

4.  **gradually move to the new abstraction and retire the old one:** gradually switch the calls to the old service (user-service in our case) over to the new implementation. once this is done, you can retire the old service completely.

here's some example code using python, as that's my current favorite:

```python
# initial interface in payment-service
class UserInfoProvider:
    def get_user_info(self, user_id):
        raise NotImplementedError

# initial implementation using user-service
class UserServiceUserInfoProvider(UserInfoProvider):
    def __init__(self, user_service_client):
        self.user_service_client = user_service_client
    def get_user_info(self, user_id):
        return self.user_service_client.get_user(user_id)

# later implementation, after user-service removal, using the external provider

class ExternalPaymentProviderUserInfoProvider(UserInfoProvider):
    def __init__(self, external_payment_provider_client):
        self.external_payment_provider_client = external_payment_provider_client
    def get_user_info(self, user_id):
       return self.external_payment_provider_client.get_user_info_from_payment(user_id)

# even later, direct access from payment-service
class DirectPaymentProviderUserInfoProvider(UserInfoProvider):
    def get_user_info(self, user_id):
        # ... fetch user info directly from a database or data store ...
        return {"user_id": user_id, "name": "from payment itself", "address": "address"}
```

**option 2: the 'data replication' approach (a little trickier, use with care)**

this is more radical and suitable when the data needed by the dependent service is relatively static, or you can tolerate some latency. in this approach, you replicate the needed data from the retiring service into the dependent one.

1.  **identify the data:** determine what data is needed from the service you want to remove.

2.  **implement data replication:** create a mechanism to replicate this data from the retiring service to the dependent one. this could be a batch process, an event-driven system, or a database replication process.

3.  **update the dependent service:** modify the dependent service to read data from its own local store instead of the retiring service.

4.  **retire the old dependency:** after that the retiring service can be turned off, it should not be needed by the dependent service anymore.

this method adds complexity with data synchronization but it gets rid of service dependency. here's an example of data replication with a simplified data model:

```python
# old approach
class PaymentService:
    def __init__(self, user_service_client):
        self.user_service_client = user_service_client

    def process_payment(self, user_id, amount):
        user_info = self.user_service_client.get_user(user_id)
        # ... process payment with user info ...

# new approach
class PaymentService:
    def __init__(self, user_local_data):
        self.user_local_data = user_local_data

    def process_payment(self, user_id, amount):
       user_info = self.user_local_data.get(user_id)
        # ... process payment using the replicated user info ...


# simplified data store
user_local_data = {
    "user_1": {"name": "john doe", "address": "address 1"},
    "user_2": {"name": "jane doe", "address": "address 2"}
}

```

**option 3: the 'event-driven' approach (when you want maximum decoupling)**

if you have the bandwidth, or if the previous options do not make sense for your business case, consider using events. in this approach you have the microservice sending the data through events to the dependent ones. the receiving dependent microservice stores the information in its own datastore.

1.  **publish events from the old microservice:** modify the microservice to publish domain events containing the necessary data.

2.  **subscribe to events:** the dependent microservice subscribes to the events and updates its own local datastore, as with the second option.

3.  **migrate to direct access if possible:** after all the dependent microservices are using the event data, you can retire the old microservice. also as in option 2, if it makes sense for the use-case you can move to get the data directly from the dependent services.

this one is a good option for situations where the dependent service doesn’t need the information immediately or frequently, thus avoiding creating more problems that the ones you are trying to solve. this method reduces coupling to a minimum, since it is fully asynchronous. it adds the complexity of the event system. here is a code example in python using a message broker:

```python
# old approach
class UserService:
    def get_user(user_id):
        # ... query the db and get the user ...
       return user_data

# new event approach for user creation
import pika
class UserService:
    def publish_user_created(user_id,user_data):
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        channel.exchange_declare(exchange='user_events', exchange_type='topic')
        channel.basic_publish(exchange='user_events', routing_key='user.created', body=str(user_data))
        connection.close()

class PaymentService:
    def __init__(self, local_user_data):
        self.local_user_data = local_user_data
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='user_events', exchange_type='topic')
        result = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        self.channel.queue_bind(exchange='user_events', queue=queue_name, routing_key='user.created')

    def event_callback(self, ch, method, properties, body):
        user = eval(body)
        self.local_user_data[user['user_id']] = user #storing the user data in payment service datastore
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_consuming(self):
        self.channel.basic_consume(queue=queue_name, on_message_callback=self.event_callback)
        self.channel.start_consuming()

```

**important considerations:**

*   **database changes:** if the retiring service has a database that's being used as a source of truth by others, you'll need to migrate data or update the dependent microservices to use a new data source. this is a big task on its own. plan this well.
*   **testing:** this is absolutely critical. you have to test each approach thoroughly. think about integration tests, contract tests, and performance tests. don't just go blindly changing things. test, test, test.
*   **monitoring:** keep a very close watch on your system during this change. you'll need to monitor for errors, performance problems, and any unexpected behavior. this type of change is not always easy to monitor.

**additional reading:**

for a more theoretical deep dive, i recommend 'domain-driven design: tackling complexity in the heart of software' by eric evans. it's a classic, and it really helps you get your head around ddd principles. also, 'building microservices' by sam newman is a great practical resource for microservices architecture. also read about bounded contexts. this will help you in the analysis phase. these books have helped me to understand these issues in depth.

it sounds like a lot of work, and honestly, it is. but when done carefully and methodically, you can remove an entity with dependencies successfully. in my experience, the key is having a clear roadmap, taking it step by step and having a good communication channel with your team. you may ask: hey i had a problem with the user service and the payment service and now i have to deal with an external service. Well, at least, now it will be their problem if they have some issues with the data, not yours... just kidding.

anyway, i hope that helps and remember to take it easy and step by step.
