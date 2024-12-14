---
title: "Why is there extra session generating with rails 5.2, jruby 9.3.4.0 (2.6.8)?"
date: "2024-12-14"
id: "why-is-there-extra-session-generating-with-rails-52-jruby-9340-268"
---

alright, so you're seeing extra sessions popping up with rails 5.2 and jruby 9.3.4.0 (that’s basically ruby 2.6.8 compat), huh? yeah, i've been there, battled with that beast myself a while back. it’s not exactly a straightforward problem, and it took me a fair amount of head scratching to get to the bottom of it. let me lay out what i learned and maybe it will save you some time.

first off, it’s not typically a rails core issue, especially if you're not seeing this behavior with standard ruby. the problem usually stems from how jruby interacts with rack and the underlying java servlet container, things that are often different than normal ruby. you end up with subtle differences in session handling that manifest as these ghost sessions.

in my experience, these extra session ids are usually created when the application server thinks a new request is being made, even though it's actually reusing the same underlying http connection. this can happen because of how jruby manages threads, or how the jvm and the web server interact when they keep connections open for performance.

for example, i remember a particularly frustrating project where we were seeing exactly this. it was a rails app talking to an api. we had a persistent http connection pool enabled, and we were blasting the api with several requests that used the same persistent http connection. for rails that looks like a user making requests to an api, however, internally it was the same connection being kept alive. the jruby/jvm combo would get in a tizzy and a new session id would be issued by the server for each request. basically, it was like having multiple users on the same session and causing problems.

here is a first example of a problem we saw, i’ll show it in an abstracted way so you get the idea:

```ruby
# bad.rb
def my_api_call(connection)
  response = connection.post("/some/endpoint", { data: 'stuff' })
  if response.code == 200
     puts "api response ok!"
  else
      puts "api error!"
  end
end

# somewhere else in your code
connection =  create_persistent_connection()
10.times { my_api_call(connection) }
```

in standard ruby this code would most likely run, and the session would be properly managed in a single thread for the user, but in jruby, this would trigger multiple session ids.

now, this is a simplified case, but imagine this happening in a complex rails application with lots of ajax calls or internal api requests, the application server could get confused about the actual user session.

one of the main culprits, and this is where my own troubleshooting time went, is the way that session cookies are handled with jruby and the app server, specifically regarding http headers. we noticed differences in how session cookie handling occurred under jruby. especially how the cookie expiration dates were being handled. so if your server or loadbalancer changes http headers along the way, it may cause a session expiration event in rails, even if the request seems to be coming from the same "user", thus causing extra sessions.

you see, the servlet container may be stricter about cookie handling than standard ruby’s webrick or puma, this can be due to the underlying jvm configuration, or the specific webserver.

the solution i found, after many hours staring at logs, was to dive deep into the session handling configurations of both the webserver and rails itself.

here's a snippet, showing how i customized the rails session store, using active record (but you could use memcache or redis, if you want the extra performance):

```ruby
# config/initializers/session_store.rb
Rails.application.config.session_store :active_record_store,
                                       key: '_my_app_session',
                                       expire_after: 24.hours, # be explicit
                                       same_site: :lax, # be explicit
                                       secure: true, # if https is used
                                       httponly: true # prevent javascript access
```
this snippet is important. being explicit with `expire_after`, `same_site`, `secure`, `httponly` can help with some inconsistencies. also, make sure to have a `session_domain` set correctly. if you are using subdomains for example, this will save you some headache. i forgot to set the `session_domain` once, and spent 2 days on a wild goose chase until i figured it out. i wasn't a happy bunny, and my hairline is still recovering from it.

now, about those http headers. often the webserver you’re running in jruby may have its own settings that override rails. for example if you are running inside a tomcat server, there are configuration files (like `context.xml` or the `server.xml`) that control session management. i strongly suggest you to review that too. look for anything related to session timeout, session cookie attributes and how the server handles persistent http connections and make sure they are aligned with what rails expects. you should also look at your loadbalancer and reverse proxy configuration too, sometimes they mess up the headers without you even knowing it. it has happened to me.

i spent some time troubleshooting that in particular. i was using haproxy as a loadbalancer, and i had a rule to change the headers `x-forwarded-proto`, but since i wasn’t also setting `x-forwarded-ssl` i had an edge case where the secure flag wasn’t correctly being set. thus creating a different session id, even for the same user.

another area to look into is how jruby is actually running your rails app. are you using a jruby-specific servlet container like torquebox or using a vanilla tomcat? each of these platforms may handle sessions in a slightly different way, and they require different configurations.

finally, and this is something i’ve seen in the wild, check your gem versions. sometimes, a specific version of a gem can cause odd interactions with jruby. especially session related gems. i spent a week debugging an issue that was all because a single gem was incompatible with jruby’s version of rack.

here is one more snippet that will show you a simple rack middleware to help you with debugging the session handling problems, add this as a debugging mechanism and remove it when not needed:

```ruby
# lib/middleware/session_debugger.rb
class SessionDebugger
  def initialize(app)
    @app = app
  end

  def call(env)
    req = Rack::Request.new(env)
    session_id = req.session.id rescue 'no session'
    puts "request path: #{req.path}, session id: #{session_id}"

    status, headers, body = @app.call(env)

    puts "response headers: #{headers}"

    [status, headers, body]
  end
end
# config/application.rb
config.middleware.use SessionDebugger
```
this middleware simply outputs the session id and headers for every request, so it’s easier to see what’s going on, and find those subtle edge cases that are creating those extra sessions. remember to add the middeware in the rails `application.rb` config.

so to summarize, when you’re seeing phantom sessions with rails, jruby, and a java webserver, focus on the following:
    1. ensure your rails session configuration is explicit and complete, specifically the expiration, domain, `secure`, and `same_site` flags.
    2. examine web server configuration for session management settings and http header manipulation.
    3. be sure to check your loadbalancer headers
    4. check your application middleware stack
    5. double-check your gems to be compatible with jruby version.
    6. test, test and test, with specific scenarios that are causing the session problems.

as for more in-depth reading about rack and session management, i would recommend looking into “rack: a ruby web server interface” by christian neukirchen and “understanding the session middleware” by rack. those are more on the low-level side, but they helped me understand the process behind the scenes. also i recommend reading the jruby documentation about rack and session handling. sometimes, it is just a matter of looking at the documentation again (and again, and again), i know it is boring, but that's how you find these things.

i hope this rambling helps. it is a tricky thing to debug, but with methodical approach you should get there. debugging these kind of situations is never easy, but always rewarding when you finally figure out what is going on.
