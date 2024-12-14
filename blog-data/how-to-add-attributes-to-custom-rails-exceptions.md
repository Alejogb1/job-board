---
title: "How to add attributes to custom rails exceptions?"
date: "2024-12-14"
id: "how-to-add-attributes-to-custom-rails-exceptions"
---

alright, so you’re hitting that spot where rails' built-in exceptions just aren't cutting it, and you need to stuff more info into your custom exceptions. i’ve been there, many times. let me tell you a bit about how i've tackled this. it's less about some magical fix and more about understanding how ruby classes and exception handling work, and then bending them to your will.

first, let’s get the basics down. when you create a custom exception in rails, it's just a ruby class that inherits from `standarderror` or one of its subclasses. this means you can treat it like any other class and add instance variables, methods, and whatever else you fancy.

in my early days, i remember a particularly nasty bug involving a third-party api. we were getting back weird responses, but the standard `rescuable` just wasn't giving us enough information to pinpoint the issue. i was spending hours tracing logs, but having more context would have saved my sanity. that's when i started going down the rabbit hole of custom exceptions.

here's the first thing you should do: define your custom exception with an initializer that accepts the extra information. this is basic ruby stuff but crucial for your problem.

```ruby
class apiunresponsiveerror < standarderror
  attr_reader :response_code, :body

  def initialize(msg = "api returned an error", response_code: nil, body: nil)
    @response_code = response_code
    @body = body
    super(msg) # calls the standarderror initialize, keeps the exception message
  end
end
```

now, when your code catches an error from the api, you can wrap it in your custom exception and include that juicy response data:

```ruby
  begin
    response = httparty.get("https://api.example.com/data")
    response.raise_for_status # this will raise httparty::responseerror
  rescue httparty::responseerror => e
    raise apiunresponsiveerror.new("api call failed", response_code: e.response.code, body: e.response.body)
  end
```

this example is fairly straightforward. we're catching the `httparty::responseerror` which might occur on http status codes other than 200. instead of just letting that bubble up, we're creating our `apiunresponsiveerror` and passing in relevant details like the http response code and response body. we are storing the information with the instance variables `@response_code` and `@body` that are accessed via `attr_reader`. you can then rescue this custom exception later and have access to that info. note how the `standarderror` is also called within the `initialize` method and it passes the exception message `msg` from this method to its own initializer.

remember, `attr_reader` defines getter methods for your instance variables. you’ll want to use `attr_writer` or `attr_accessor` if you need to set or modify those attributes after initialization, but in my experience, exceptions should be immutable once created, so readers are usually good enough.

the really useful stuff comes into play when you start to think about how you use these attributes. are you just logging them? are you displaying them to users (careful with sensitive info!)? are you using them in specific rescues? let’s consider the logging case:

```ruby
# in a global exception handler, e.g application_controller
  rescue_from apiunresponsiveerror do |exception|
    logger.error "api error: #{exception.message} (code: #{exception.response_code}, body: #{exception.body})"
    render json: {error: "an external api call failed"}, status: :internal_server_error
  end
```

this is a common pattern for handling exceptions in a controller. we use `rescue_from` to catch specific error types. then, using `exception.message`, `exception.response_code`, and `exception.body` we're able to generate a nice log message with the details of the exception. we can also perform other actions such as returning an error response to the end user. this will catch all instances of `apiunresponsiveerror` and output the data to the rails log as well as rendering a json response.

now let's tackle a more intricate situation. say, your application makes calls to a bunch of different apis and each of them might respond in their own ways, returning different body formats. a simple http response code won’t cut it. so, maybe you want to log the specifics about the error itself from the api. let’s take a look at that:

```ruby
class specificapiunresponsiveerror < apiunresponsiveerror
  attr_reader :api_specific_error_details

  def initialize(msg = "api returned an error", response_code: nil, body: nil, api_specific_error_details: {})
    @api_specific_error_details = api_specific_error_details
    super(msg, response_code: response_code, body: body)
  end
end
```

this example defines a subclass of `apiunresponsiveerror` called `specificapiunresponsiveerror`. we're adding a `api_specific_error_details` attribute which will store details of the specific error returned by that api. the constructor calls `super` to reuse our previous `apiunresponsiveerror` constructor with the necessary parameters.
now, we can modify the api call logic to generate the more specific exception when needed:

```ruby
begin
    response = httparty.get("https://api.example.com/special-data")
    response.raise_for_status
  rescue httparty::responseerror => e
    parsed_body = json.parse(e.response.body)
     raise specificapiunresponsiveerror.new("api call failed",
      response_code: e.response.code,
      body: e.response.body,
      api_specific_error_details: parsed_body["error_details"]
    )
  end
```
we're parsing the json body and extracting specific error details into the new attribute in the custom exception. this allows us to capture the extra information about the api call.

one last piece of advice i can give is to be careful about what you stuff into your exceptions. avoid adding sensitive info like passwords or session tokens, especially if they are logged or displayed in any way. if you start adding too many attributes, it might indicate that your exception isn't well-defined. think about the core problem you're trying to represent with the exception and whether all that information is truly necessary. or, you might want to make different, more specific, exceptions.

this is where it's important to get a good grasp of solid principles, like “single responsibility”. your exceptions should be focused and have a specific meaning. if your exception starts storing everything about your database, your environment, and the phase of the moon, you should probably split it into multiple ones. also remember that exceptions, despite being very powerful tools, are not a way to pass data around your application, but a sign that something unexpected happened.

as for resources, i'd recommend looking into “effective ruby: 48 specific ways to write better ruby” by gregory brown, this will help you understand ruby classes and inheritance in much more depth. the ruby documentation on `standarderror` is also helpful. for exception handling itself, i suggest studying the design patterns found in martin fowler's "patterns of enterprise application architecture," especially the section on exception management. those are books i found really useful during my career. avoid the temptation to look for simple one-liners because that rarely addresses the underlying issues and often introduces more complexity later on.

also, one time i spent all night trying to debug an exception only to find that the problem was actually just a typo in one of the log messages. that was a great night.

so, that's about it. think about the data you need, create attributes for them, use initializers, and think about your logging and rescuing logic. it’s not rocket science, just solid coding principles put to use. good luck out there.
