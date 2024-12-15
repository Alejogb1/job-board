---
title: "Where 'global entities' should be in a DDD SaaS example?"
date: "2024-12-15"
id: "where-global-entities-should-be-in-a-ddd-saas-example"
---

ah, global entities in a ddd saas, that old chestnut. it's a question that's tripped up more than a few of us, myself included, back in the day. i remember when i was first getting into domain driven design, i thought everything had to fit neatly into a bounded context. turns out, that's not always the case, and sometimes you've got these entities that feel like they belong everywhere and nowhere at the same time. you know, the ones that every part of your application needs to know about.

the core problem, as i see it, isn’t *where* to put these global entities technically, but how to model them correctly in the first place. if you have a true global entity, it implies that you are probably misunderstanding some fundamental aspects of your domain. these 'global entities' are usually masquerading as core business concepts that should have their proper homes.

for instance, let's talk about something seemingly simple like "currency." it's tempting to say, "ah, currency, that's a global entity, everyone needs that." so we have a currency entity with fields like `code`, `name`, and `symbol`. sure, every context might need to know about currency, but does every context need to manipulate it? probably not. i'd argue that currency is most likely not an entity in the ddd sense but rather a value object. the actual entity would be something else perhaps an "account" or "product listing" or "transaction."

in fact, the first saas i worked on, many years ago, we had a 'user' entity that we treated like a global entity. it was everywhere! every service, every context, knew about the `user` entity. we were constantly chasing down bugs where one service would modify the `user` entity and cause issues in another service. it was a nightmare. later, after much head-scratching, we realized that what we thought of as a global `user` entity, was actually several different concepts all crammed together. there was the authentication and authorization user context, the user profile context, and even user preferences that belonged in an independent context. after breaking it apart into three separate contexts, each with its own user concept, life became much much easier.

that’s why we have to be careful. i think most of the time you are trying to create a shared entity that represents something which is actually a reference point, a read-only source of truth, or a value that is just used in the context but not owned by it.

if you really have an entity which is truly global, a couple of patterns tend to work well. the first one is a kind of shared data context or shared kernel. this is where the global entity exists and its context is just to serve other contexts. for example, a reference point for an entity like a country. all you need is the codes or information about those countries. this context is in charge of updating the data of the entity itself, and all the other contexts just use that data.

here’s a simple example in python for how you could structure this read-only shared context for `country` entities which is not an actual DDD entity:

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class CountryData:
    name: str
    code: str
    continent: str

class CountryDataProvider:
    def __init__(self, data: Dict[str, CountryData]):
        self._data = data

    def get_country(self, code: str) -> Optional[CountryData]:
        return self._data.get(code)

    def all_countries(self) -> list[CountryData]:
        return list(self._data.values())
    

# example setup
country_data = {
    "us": CountryData(name="United States", code="us", continent="north america"),
    "uk": CountryData(name="United Kingdom", code="uk", continent="europe"),
    "ca": CountryData(name="Canada", code="ca", continent="north america"),
    "fr": CountryData(name="France", code="fr", continent="europe"),
}

country_provider = CountryDataProvider(country_data)
```

in this example, our `countrydataprovider` acts as the single source of truth for country data. different bounded contexts can use this `countrydataprovider` to retrieve country information, but none of them are responsible for updating or managing this data. the important point is that the `CountryData` class is a value object (it is immutable) and the `CountryDataProvider` doesn't implement any sort of write functionality to the data, it only provides read access. this approach decouples each context from needing to know the specifics of the country's data and also prevents having conflicts of different versions of the same entity spread accross your application.

another approach you can use which sometimes can look like a global entity is when you use event sourcing. in event sourcing you can use an event bus to share events to different contexts, and the contexts that need to use a particular event, just listen to the bus and update its own data model.

here is a very simple example of how it could be implemented with python. imagine an event like a new user signup:

```python
import uuid
import datetime
from typing import Dict, List
from abc import ABC, abstractmethod

#event definition
class Event(ABC):
    def __init__(self, event_id: str, occurred_at: datetime.datetime):
       self.event_id = event_id
       self.occurred_at = occurred_at

@dataclass
class UserSignedUpEvent(Event):
   user_id: str
   email: str
   
#event bus definition
class EventBus:
  def __init__(self):
    self._handlers: Dict[str, List[callable]] = {}
  
  def publish(self, event: Event):
     event_type = type(event).__name__
     if event_type in self._handlers:
        for handler in self._handlers[event_type]:
           handler(event)
     
  def subscribe(self, event_type:str, handler: callable):
      if event_type not in self._handlers:
          self._handlers[event_type] = []
      self._handlers[event_type].append(handler)

#example implementation
event_bus = EventBus()

def email_handler(event:UserSignedUpEvent):
  print(f"sending email to: {event.email}")

def user_repository_handler(event:UserSignedUpEvent):
   print(f"persisting user {event.user_id} with email: {event.email}")
   #persisting in db...
   
event_bus.subscribe(UserSignedUpEvent.__name__, email_handler)
event_bus.subscribe(UserSignedUpEvent.__name__, user_repository_handler)

user_event = UserSignedUpEvent(event_id=str(uuid.uuid4()), occurred_at=datetime.datetime.now(), user_id=str(uuid.uuid4()), email="test@test.com")
event_bus.publish(user_event)
```

this simple example provides a basic implementation of an event bus. you can see how the email service and the user repository service just listen to the bus. when the user signed up event happens, both handlers are triggered.

in this pattern no context owns the user entity per se, but they just react when a new user is signed up by listening to an event. this decouples contexts from knowing each other. this allows each bounded context to use its own data models for its specific needs while reacting to the events that are relevant to it.

and of course, there's the good old data transfer object (dto) approach. sometimes you don't even need entities in different contexts, just data. in that case, your global entity might be better served as a dto that's passed between services and contexts. it's about sending the information not the responsibility of that information. the main idea is to transfer only the relevant data each context needs.

here's a python example of that:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class UserDto:
    user_id: str
    email: str
    first_name: str
    last_name: str

def create_user_dto(user_id: str, email: str, first_name: str, last_name: str) -> UserDto:
    return UserDto(user_id=user_id, email=email, first_name=first_name, last_name=last_name)

#example use case
user_dto = create_user_dto(user_id="123", email="test@test.com", first_name="john", last_name="doe")
print(user_dto)

```

the `userdto` is just a data container. no specific context owns this structure. it can be constructed by one context, transferred to another and used there.

the key takeaway here, in my experience, is that often, these supposed "global entities" aren't truly global entities, but rather a kind of data or event that different contexts need to be aware of. if you are constantly changing data in the entity from several contexts, and that is creating trouble, it’s a warning sign. think if the entity should be decomposed into multiple entities or if that entity is just not an entity, but a value object or a reference value.

as for resources, i'd suggest taking a look at eric evans' "domain-driven design: tackling complexity in the heart of software". it's a classic for a reason and it tackles these situations in depth. there are other books of martin fowler that can be useful for the event sourcing technique.

and sometimes, as a last resort, if all the modeling approaches fails, there is always the option to use a good old shared database between contexts. i'm just kidding, don't do that. i mean it… or else!
