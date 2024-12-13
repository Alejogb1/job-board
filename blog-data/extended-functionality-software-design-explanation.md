---
title: "extended functionality software design explanation?"
date: "2024-12-13"
id: "extended-functionality-software-design-explanation"
---

Okay so extended functionality software design huh Been there done that got the t-shirt Well several t-shirts actually dealing with this kind of stuff I’ve seen it all from monolithic disasters that crashed if you looked at them wrong to microservices that talked so much they made your network cry

Let me give you a rundown from a trenches perspective because theoretical textbook answers are fun and all but we are here for real-world scars

First off when you say extended functionality we are usually talking about a couple of things A requirement came in for something brand new that the original design didn't quite anticipate or we need to bolt-on some capability that we weren't sure was even possible during the first pass Right and both of these have their own set of design headaches We can talk about both since you are open to this

I remember one project we had it was some kinda legacy CRM system built 10 years ago I was brought in as a consultant to "modernize" it haha Good luck with that the original design was something out of a nightmare of spaghetti code every method was a mile long and god knows what was happening inside those loops Every new feature felt like adding another precarious level on a Jenga tower one wrong move and the whole thing was crashing We needed to add an integration with a new external payment gateway and the whole thing felt like trying to fit a square peg into a round hole I spent days untangling logic just to understand where to put the payment API calls I had nightmares where I saw lines and lines of code chasing me

For those cases my preferred strategy revolves around isolating the old system and building the new functionality as a separate module or microservice This can be a little bit more work initially but it saves pain in the long run Trust me here we are trying to avoid the scenario when touching one line makes 100 other things stop working We did exactly that we built a small microservice completely independent from the main CRM We call it "payment gateway service" And that thing handles the authentication payments error handling everything and it worked beautifully for years

Think about it like this instead of trying to carve a new room into an old cramped house you build a separate structure and maybe add a nice bridge later This allows you to work on the new feature without the fear of destabilizing the legacy system it also buys you time to slowly maybe possibly migrate the old bits to the new system

Here's a basic code example using python illustrating that separation of concerns:

```python
# old_crm_system.py - Simplified example of the legacy system

def process_customer_order(customer_id, order_details):
    # spaghetti logic
    print(f"Legacy CRM processing order for customer {customer_id}")
    return True # or False who knows in real life

# new_payment_service.py - The microservice doing its thing
import requests

class PaymentService:
    def __init__(self, api_url):
        self.api_url = api_url

    def process_payment(self, amount, payment_details):
        try:
          response = requests.post(f"{self.api_url}/payments", json={"amount": amount, "payment_details": payment_details})
          response.raise_for_status() # raise an exception for bad status codes
          return True
        except requests.exceptions.RequestException as e:
          print(f"Payment error {e}")
          return False

# main.py - How they interact
from old_crm_system import process_customer_order
from new_payment_service import PaymentService

payment_api_url = "https://fake-payment-gateway.com"
payment_service = PaymentService(payment_api_url)

customer_id = 123
order_details = {"items": ["item1", "item2"], "total": 100}

if process_customer_order(customer_id, order_details):
    payment_success = payment_service.process_payment(order_details["total"], {"card_number": "1234-5678-9012-3456", "expiry": "12/24"})
    if payment_success:
        print("Order processed successfully")
    else:
        print("Order payment failed")

```

See the clear separation there The old CRM system does its basic order processing then calls the new Payment Service to handle the money bits This way if you break the payment bit (or even if the third party gateway changes its api) the CRM is mostly unaffected

Sometimes though it is not about complete replacements or microservices sometimes you just need to add a specific feature in an existing codebase For that we should talk about the use of design patterns like Strategy or Decorator This is when your app isn't a complete mess but you know you should not mess with the existing logic too much

For example let’s say you have a basic logging functionality right now it writes to a file But you want to add the option to write logs to a database or maybe some cloud logging service without touching the main log function all over the app The decorator design pattern is your best friend here:

```python
# logger.py - The old base logger
class BaseLogger:
  def log(self, message):
    print(f"Base Logger {message}")
    with open("app.log", "a") as f:
      f.write(f"{message}\n")

# db_logger.py - Decorator logging to DB
class DatabaseLogger:
    def __init__(self, logger):
      self.logger = logger
    def log(self, message):
      print("db logic here")
      self.logger.log(message)

# cloud_logger.py - Decorator logging to cloud
class CloudLogger:
    def __init__(self, logger):
      self.logger = logger
    def log(self, message):
      print("cloud logic here")
      self.logger.log(message)

# main.py
from logger import BaseLogger
from db_logger import DatabaseLogger
from cloud_logger import CloudLogger

base_logger = BaseLogger()

db_logger = DatabaseLogger(base_logger)
cloud_logger = CloudLogger(base_logger)

base_logger.log("This is base log")
db_logger.log("This is db log")
cloud_logger.log("This is cloud log")
```

Here the `BaseLogger` is our old logging system and instead of changing the `log` method we wrap it with `DatabaseLogger` or `CloudLogger` this allows us to extend the functionality without changing a single line of existing logic And yes we can compose several decorators one after the other like this to get the desired behavior

Now lets say we have a complex validation process in place for user data right we have validations for username password email all that stuff But now we want to add a new validation for address data that should happen only when the user selects an option to provide it instead of touching all the user validation methods again we can create a new validation class that is called only under that condition using the Strategy pattern:

```python
# validator.py - The strategy interfaces
class ValidationStrategy:
    def validate(self, data):
        raise NotImplementedError

class UserNameValidator(ValidationStrategy):
    def validate(self, data):
        print(f"Validating username {data}")
        return len(data) > 5

class PasswordValidator(ValidationStrategy):
    def validate(self, data):
        print(f"Validating password {data}")
        return len(data) > 8

class AddressValidator(ValidationStrategy):
  def validate(self, data):
    print(f"Validating address {data}")
    return len(data) > 10

# data.py - data processing class using strategy
class UserDataProcessor:
    def __init__(self):
      self.validators = []
    def add_validator(self, validator):
      self.validators.append(validator)
    def validate_data(self, data):
        for validator in self.validators:
          if not validator.validate(data):
            return False
        return True

# main.py
from validator import UserNameValidator, PasswordValidator, AddressValidator
from data import UserDataProcessor
processor = UserDataProcessor()
processor.add_validator(UserNameValidator())
processor.add_validator(PasswordValidator())

# example 1 normal use
user_data = {"username": "myuser", "password": "mypassword"}
if processor.validate_data(user_data):
    print("User data is valid")
else:
    print("User data is not valid")

# example 2 only if has address data use address validation
user_data_with_address = {"username": "myuser", "password": "mypassword", "address": "123 Main St"}
processor.add_validator(AddressValidator())
if processor.validate_data(user_data_with_address):
    print("User data with address is valid")
else:
    print("User data with address is not valid")

```

Here `UserNameValidator` `PasswordValidator` and `AddressValidator` are the strategies each doing their part and the `UserDataProcessor` is using them based on the context this helps a lot with code reusability and separation of concerns

A little side note i saw a guy at work once trying to use a hammer to drive a screw into the wood you may laugh but it’s as funny as trying to extend an application without considering your toolset right tool for the job is a thing my old mentor always told me you should remember that

Now when you are dealing with the complexity of software architecture keep in mind that what’s ideal in theory isn’t always practical in reality We need to make compromises we need to balance flexibility maintainability and performance it is a real game of trade offs

And finally don’t reinvent the wheel there are excellent resources available instead of looking for blog posts you should read books like "Design Patterns: Elements of Reusable Object-Oriented Software" by the Gang of Four it’s a classic for a reason And for microservices architecture you can look into “Building Microservices” by Sam Newman it will help you get a grasp of distributed architecture

Software design is not about the best solution its about the best solution for this particular context And I hope I gave you a bit of real world info that you can apply to your own problem. And remember keep your code clean write meaningful tests and may your deploys be successful because no one likes a late night debugging marathon.
