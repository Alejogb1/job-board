---
title: "Which is the proper way to declare this kind of object for swagger?"
date: "2024-12-15"
id: "which-is-the-proper-way-to-declare-this-kind-of-object-for-swagger"
---

alright, so you're asking about how to properly define an object schema in swagger (or openapi, same thing mostly). i’ve been around the block a few times with this, trust me. it might seem straightforward, but there are a few ways to skin this cat, and some are definitely cleaner than others, especially when your api starts growing.

let's start with what i usually see people getting confused with – the difference between inline objects and using `$ref`. i remember back in my early days, i’d just throw everything inline into the `properties` section of my swagger definition. oh boy, what a mess that became! my swagger files were sprawling, duplicates were everywhere, and when i needed to make a tiny change, i had to hunt down each instance of the object. nightmare fuel, honestly. i even ended up, that one time, copy pasting the schema of the user object (first name, last name, id) into different parts of the schema definition just for the `getUser` endpoint and the `postUser` endpoint, and well... changing the id field type from int to string in one of them was enough to create a real mess for the frontend, good times... good times... if we can call that "good".

so, the key is to think in terms of reusability and maintainability. you don’t want to be repeating yourself; that's just asking for trouble. that's why `$ref` is your best friend. it allows you to define your object schemas once and then reference them wherever they're needed. think of it like defining a function; you write it once and call it multiple times. cleaner code, easier to debug and more consistent!

i'll throw some examples for clarity. let’s say you have a simple user object.

**inline example (the messy way, just for illustration):**

```yaml
paths:
  /users:
    get:
      summary: Get a list of users
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: array
                items:
                  type: object
                  properties:
                    id:
                      type: integer
                      format: int64
                    firstName:
                      type: string
                    lastName:
                      type: string
    post:
      summary: Create a new user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                id:
                  type: integer
                  format: int64
                firstName:
                  type: string
                lastName:
                  type: string
```

see how we've got the user object defined directly in both the `get` and the `post` responses? that’s not good! in practice, i've seen this mess even for more complex objects than this, for some reason, some people like to torture themselves with duplication. it gets worse as your api grows.

now, let’s do it the better way, with `$ref` to the rescue:

**using `$ref` (the cleaner way):**

```yaml
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
          format: int64
        firstName:
          type: string
        lastName:
          type: string
paths:
  /users:
    get:
      summary: Get a list of users
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/User'
    post:
      summary: Create a new user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/User'

```

here, we define the user object under `components/schemas/User` and then use `$ref` to reference it in the `get` and `post` responses. see how much cleaner that is? makes life easier, reduces copy-paste and keeps things consistent. i swear, copy-paste programming should be a crime; it's a gateway to more errors and bugs!

now, about specific types, because i've also seen some confusion there. for example, handling dates, timestamps, and enums can trip people up. the `format` property is your friend here, just try to stay consistent, if you're using dates, use `date` instead of a string, otherwise it would require the frontend to do some extra parsing. i once had a frontend dev literally tell me "my life is hard enough already, please, use proper types on the api response". i've never forgotten that.

here's an example with a date and an enum:

**example with date and enum:**

```yaml
components:
  schemas:
    UserProfile:
      type: object
      properties:
        userId:
          type: integer
          format: int64
        birthDate:
          type: string
          format: date
        status:
          type: string
          enum: [active, inactive, pending]
paths:
  /user/profile/{userId}:
    get:
      summary: Get a user's profile
      parameters:
        - in: path
          name: userId
          required: true
          schema:
            type: integer
            format: int64
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserProfile'

```

so, we’ve got `birthDate` defined as `type: string` with `format: date`, which tells swagger and the tools that consume it that it's a date string, not just any string. and the status is defined as an enum using the `enum` keyword. now that's some good stuff right there, much better than receiving a random string in a field that should be an enum, trust me, i've seen things... horrible things.

one last thing i'll add, be consistent about naming conventions. don't be calling one object `user` and another one `UserProfile` for the same entity, that's just confusing. and for god’s sake, do not name your schemas the same as your http verbs! i once received a yaml file with the following schemas: "get", "post", "delete". oh, the humanity!

for resources, instead of a link i’d strongly suggest looking at the official openapi specification documentation, it's the most reliable source for everything swagger. also, i’ve always found that "designing apis" by audrey tang (which, by the way, its not the same audrey tang, it's a very common name) and “rest api design rulebook” by mark massé both to be useful in structuring apis.

also, a tip, if you really want to get into the nitty-gritty, check out any of the many online swagger editors. they are really great at validating your swagger definitions and catching errors early. they're like having a linter for your api definitions. they can also generate code for your apis in most languages, so it's like 2 tools in one. that's more than what you can get with your average potato.

i hope this helps. it's a process, but once you get the hang of it, defining schemas in swagger becomes much more straightforward, and you avoid a lot of headaches down the road.
