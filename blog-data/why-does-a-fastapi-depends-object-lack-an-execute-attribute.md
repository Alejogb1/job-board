---
title: "Why does a FastAPI Depends object lack an execute attribute?"
date: "2024-12-23"
id: "why-does-a-fastapi-depends-object-lack-an-execute-attribute"
---

Alright, let's delve into why a FastAPI `Depends` object doesn't come with an `execute` attribute. It's a question I've encountered myself a few times, particularly during the early days of adopting FastAPI for our microservices. The confusion often stems from a misunderstanding of how dependency injection works within FastAPI’s framework. The short of it is: it's not meant to be executed directly by the user. The `Depends` object, at its core, is a *declaration* of a dependency, not a direct command to execute something. Think of it as a blueprint, rather than the building itself.

Now, let's unpack this a bit. In traditional procedural programming, you might call functions directly. With FastAPI, especially when using `Depends`, we shift towards a declarative approach. You declare what needs to be available (the dependency), and FastAPI’s machinery figures out how and when to provide it. When you use `Depends` within a route definition, you're telling FastAPI, "Before executing this route, make sure this function (your dependency) is called, and its result is passed into the route function as an argument." This dependency resolution happens behind the scenes during the route invocation process.

The absence of an `execute` attribute isn't an oversight; it’s a design choice reflecting this core principle of declarative dependency injection. If `Depends` had an `execute` method, it would suggest direct, manual execution, disrupting the intended flow and undermining the system. That would break the core contract FastAPI has of handling that for you.

To understand it better, let’s look at how FastAPI processes these dependencies. When a request hits an endpoint, FastAPI's internals examine the endpoint’s signature, identifies the arguments defined using `Depends`, and then fetches the results of those dependencies in the correct order. The result is that this function is called by FastAPI, not you. The result of each dependent function is then passed into your route’s function as an argument. This is what allows you to avoid writing repetitive code. You effectively encapsulate logic and reuse it across multiple routes.

Let's illustrate this with a few examples. I remember struggling with this at one point as we were moving our authentication system from something a bit homebrew to FastAPI and oauth2, so it makes a good, illustrative scenario:

**Example 1: Simple Dependency**

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated
from datetime import datetime

app = FastAPI()

def get_current_time():
    return datetime.now()

@app.get("/time")
def read_time(time: Annotated[datetime, Depends(get_current_time)]):
    return {"current_time": time}

```

In this example, `get_current_time` is the dependency. We don't and should not try to `get_current_time.execute()`. Instead, `Depends(get_current_time)` is a declaration that tells FastAPI that the result of `get_current_time()` needs to be passed into the `read_time` function as the `time` argument, before the route itself is invoked. FastAPI takes care of executing `get_current_time()` when a request hits `/time`.

**Example 2: Dependency with Parameters**

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class User(BaseModel):
    name: str
    id: int

async def get_user_by_id(user_id: int) -> User:
    # imagine a database lookup here
    if user_id == 1:
       return User(name="Alice", id=1)
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.get("/users/{user_id}")
async def read_user(
    user: Annotated[User, Depends(get_user_by_id)]
):
    return {"user": user}

```

Here, `get_user_by_id` is a dependency that itself takes an input. Again, we don't manually execute it. `Depends(get_user_by_id)` indicates that this function must be called to satisfy the argument for `user`, and FastAPI knows it needs to extract the `user_id` from the URL and pass that to `get_user_by_id`. The result is then passed into `read_user` function, where we can then use the resultant `user` object. You can even make the dependencies asynchronous, as shown here, which opens the door for I/O bound tasks like calls to databases or external apis.

**Example 3: Security Dependency**

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from typing import Annotated
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
security = HTTPBearer()

class User(BaseModel):
   name: str
   id: int

async def get_current_user(
   token: Annotated[str, Depends(security)]
) -> Optional[User]:
    if token == "validtoken":
        return User(name = "Bob", id=2)
    else:
      return None


async def is_active_user(user:Annotated[Optional[User], Depends(get_current_user)]):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


@app.get("/secure")
async def secure_endpoint(
   user: Annotated[User, Depends(is_active_user)]
):
    return {"message": f"Hello, {user.name}!"}

```

This example incorporates a security dependency to fetch and validate a user with bearer tokens in the authorization header. `get_current_user` and `is_active_user` are both dependencies. The user should not attempt to run those dependencies themselves, instead, we instruct FastAPI that we need a token to fulfill our security dependency, and it will look to see if that token exists within the authorization header. If valid, it proceeds onward and returns a result which is then passed into our `secure_endpoint` route function. This example further reinforces how the entire system works; you declare your dependencies and FastAPI’s dependency injection system handles the rest.

The key takeaway here is that you are not responsible for executing these functions directly, which is why they lack an `execute` attribute. The `Depends` object is a signal to FastAPI’s system to handle the execution of your specified dependencies as part of the request lifecycle.

For a deeper understanding of dependency injection patterns and their use within python frameworks, I'd recommend looking at *Dependency Injection in Python* by Mark Seemann. For a more specific view into how FastAPI implements this I'd recommend checking out the official FastAPI documentation, which offers a pretty good technical guide to the internals. Also, reviewing the source code of FastAPI on GitHub is a valuable exercise to see how things really work, especially if you're up for a more involved investigation. Additionally, explore articles discussing patterns in dependency injection more generally, and you'll realize the common theme that it's the framework's responsibility to handle invocation.

To reiterate, the absence of an execute attribute is intentional, reflecting the declarative nature of dependency injection. It streamlines the development process and promotes reusable logic. By letting FastAPI manage the dependency execution, you, as the developer, can focus more on the core application logic and less on the plumbing involved in fetching and setting up your dependencies.
