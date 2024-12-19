---
title: "post on session robot framework implementation?"
date: "2024-12-13"
id: "post-on-session-robot-framework-implementation"
---

Alright so you're wrestling with session management in Robot Framework right been there done that got the t-shirt and several battle scars to prove it This isn't your typical hello world scenario we're talking about persistent state communication between different parts of your test suite and that's where things can get hairy faster than a cat stuck in a yarn ball Let's break it down like we're debugging a particularly stubborn piece of code

I've been using Robot Framework since it was practically a newborn baby and trust me session management especially when you are working with different APIs or trying to simulate a multi-user scenario it can become a real head scratcher especially if you're used to simple linear scripts Remember back in the day when we were using just keywords and global variables everything felt like a single threaded application running sequentially a simple script a simple life then we needed to keep state between different parts of our tests boom sessions enter stage left we're talking about a shift from linear to a more dynamic approach it's a bit like trying to manage a network of independent actors all needing to know a bit of the current world's state

So the core of the problem here is that Robot Framework by default doesn’t have built in session management it's designed to be a test automation framework not a web server or a stateful application this is a feature not a bug but it means we have to take a step outside of the box to simulate sessions we need to establish a shared space a central hub where we can store session data that every part of the test can access

My first approach and I'm not proud of it was global variables don't do this trust me it's like using duct tape to fix a leak in a spaceship it works for a hot minute but then it explodes I remember one project where I used global variables to keep the user information after a successful login it was a disaster tests would randomly fail because the global variables would get overwritten by different test cases and we spent days debugging ghost failures

Then I discovered dictionaries they’re like mini databases right there in the test suite a map or a record where we can store data keyed by some meaningful identifier and that my friend is the solution we're looking for It could be a user ID or a session token or even a device ID Anything that can uniquely identify the context of the session

Here is an example of a simple session management using dictionaries

```python
*** Settings ***
Library    Collections
*** Variables ***
${SESSIONS}  Create Dictionary

*** Keywords ***
Create Session
    [Arguments]    ${session_id}
    Set To Dictionary    ${SESSIONS}    ${session_id}    Create Dictionary

Set Session Data
    [Arguments]    ${session_id}    ${key}    ${value}
    ${session}    Get From Dictionary   ${SESSIONS}   ${session_id}
    Set To Dictionary   ${session}   ${key}    ${value}
    Set To Dictionary    ${SESSIONS}    ${session_id}    ${session}

Get Session Data
    [Arguments]    ${session_id}   ${key}
    ${session}    Get From Dictionary    ${SESSIONS}    ${session_id}
    ${value}      Get From Dictionary   ${session}    ${key}
    [Return]    ${value}

Delete Session
    [Arguments]    ${session_id}
    Remove From Dictionary    ${SESSIONS}    ${session_id}

*** Test Cases ***
Example Session Management
    Create Session    user123
    Set Session Data    user123    username    john.doe
    Set Session Data    user123    auth_token  some_token
    ${username}  Get Session Data    user123   username
    Log    Username is ${username}
    ${token}  Get Session Data    user123    auth_token
    Log    Auth token is ${token}
    Delete Session  user123

```

This example is a basic implementation but it shows the core idea we have a dictionary called `$SESSIONS` which contains other dictionaries Each session is uniquely identified by a string like user123 and that dictionary holds all the data for that session

Now the key to this solution is making sure that you are referencing the session with a unique ID every time and of course managing the session's lifecycle I like to think of sessions like a temporary tattoo pretty cool but it doesn’t last forever You have to create the session you have to use it and you have to destroy it once it’s no longer needed otherwise your code will start smelling bad with unused data

Sometimes you need a session for each browser or each user so the session id could be something like browser1 or user_a_b_c_123 etc it all depends on your application's requirements I've seen scenarios where teams are using a combination of the device ID and the user ID to create more complex sessions it's all about the level of granularity that you need in your tests

Here's a more concrete example using a REST API you can adapt this if you're using a different protocol

```python
*** Settings ***
Library    RequestsLibrary
Library    Collections
*** Variables ***
${SESSIONS}  Create Dictionary
${BASE_URL}   https://api.example.com

*** Keywords ***
Create APISession
    [Arguments]    ${session_id}
    ${session}  Create Dictionary    base_url  ${BASE_URL}
    Set To Dictionary    ${SESSIONS}    ${session_id}    ${session}
    Create Session   ${session_id}   ${BASE_URL}

Set APISessionHeader
    [Arguments]    ${session_id}    ${header}    ${value}
    ${session}    Get From Dictionary   ${SESSIONS}   ${session_id}
    ${headers}   Get From Dictionary   ${session}   headers    default
    Run Keyword If    "${headers}" == "default"  Create Dictionary    headers
    Set To Dictionary   ${headers}   ${header}    ${value}
    Set To Dictionary   ${session}  headers     ${headers}
    Set To Dictionary    ${SESSIONS}    ${session_id}    ${session}

Get APISessionData
   [Arguments]    ${session_id}   ${key}
    ${session}    Get From Dictionary    ${SESSIONS}    ${session_id}
    ${value}      Get From Dictionary   ${session}    ${key}
    [Return]    ${value}

Make APIRequest
    [Arguments]   ${session_id}    ${method}    ${endpoint}    ${data}=None
    ${session}    Get From Dictionary   ${SESSIONS}    ${session_id}
    ${headers}    Get From Dictionary    ${session}     headers    default
    ${base_url}    Get From Dictionary   ${session}    base_url
    ${url}  Catenate  ${base_url}  ${endpoint}
    ${response}   Send Request   ${method}   ${url}   headers=${headers}  json=${data}
    [Return]  ${response}

Delete APISession
    [Arguments]    ${session_id}
    Remove From Dictionary    ${SESSIONS}    ${session_id}

*** Test Cases ***
Test API Session Management
    Create APISession   user_session_1
    Set APISessionHeader  user_session_1    Content-Type   application/json
    Set APISessionHeader  user_session_1    Authorization  Bearer some_auth_token
    ${response} Make APIRequest  user_session_1  GET /users
    Log  Response: ${response.status_code}
    Log  Response content ${response.json()}
    Delete APISession user_session_1

```

In this code I created a set of keywords that wrap the RequestsLibrary I create a session with a base url I keep the headers there and then I use those information on the API requests with a unique session id So no more mixing things up

And finally if you have a database you can use that as your session management but you need to have proper locking mechanisms so you don’t get race conditions If you don't know what is a race condition trust me you don't want to know it now ( it's a joke get it? it's a race )

Here is a very basic example of a session management using redis

```python
*** Settings ***
Library    redis
Library   Collections
*** Variables ***
${REDIS_HOST}    localhost
${REDIS_PORT}    6379
${REDIS_DB}     0

*** Keywords ***
Redis Connect
    Connect To Redis   ${REDIS_HOST}    ${REDIS_PORT}    ${REDIS_DB}

Create RedisSession
    [Arguments]    ${session_id}
    Redis Connect
    Set Redis   ${session_id}    {}

Set RedisSessionData
    [Arguments]    ${session_id}    ${key}    ${value}
    Redis Connect
    ${session_data}  Get Redis   ${session_id}
    ${session_data_dict}   Evaluate   json.loads( """${session_data}""" )    json
    Set To Dictionary    ${session_data_dict}  ${key}   ${value}
    ${session_data_json}  Evaluate  json.dumps( """${session_data_dict}""" )   json
    Set Redis    ${session_id}   ${session_data_json}

Get RedisSessionData
    [Arguments]    ${session_id}   ${key}
    Redis Connect
    ${session_data}   Get Redis   ${session_id}
    ${session_data_dict}   Evaluate    json.loads( """${session_data}""" )   json
    ${value}     Get From Dictionary   ${session_data_dict}    ${key}
    [Return]     ${value}

Delete RedisSession
    [Arguments]    ${session_id}
    Redis Connect
    Del Redis   ${session_id}

*** Test Cases ***
Test Redis Session
    Create RedisSession session123
    Set RedisSessionData  session123   username   john.doe
    Set RedisSessionData  session123   auth_token   some_secret_token
    ${username}  Get RedisSessionData   session123   username
    Log  Username ${username}
    ${token}  Get RedisSessionData   session123   auth_token
    Log  Auth token ${token}
    Delete RedisSession session123

```
These examples will give you a solid base to work with I strongly recommend you not to implement your own session management from scratch as in real life scenarios you will face a lot of challenges and race conditions and you will end up with a lot of technical debt I suggest you do more research about the subject like the paper "Distributed Transactions" from the 1980s and read about different implementation of sessions in the web technologies world or look up books like "Designing Data-Intensive Applications" by Martin Kleppmann to dive deep into the subject to see which way suits better your project I hope this helps you tackle your session management woes Happy testing and remember there’s always a better way to do it keep learning and exploring and don’t be afraid to refactor when needed
