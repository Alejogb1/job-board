---
title: "Why is Ahoy gem filtering visit_token/visitor_token, and why is Geolocation NOT storing data in development/production database?"
date: "2024-12-15"
id: "why-is-ahoy-gem-filtering-visittokenvisitortoken-and-why-is-geolocation-not-storing-data-in-developmentproduction-database"
---

alright, let's unpack this ahoy and geolocation situation, seems like you've hit a couple of common snags. i've been down these roads before, probably more times than i care to remember, and i've got a few thoughts on what's likely going on, along with how to get things humming.

first up, the ahoy gem filtering of `visit_token`/`visitor_token`. this is something i actually struggled with a few years back when building an early user analytics dashboard. i was pulling my hair out trying to figure out why certain visits weren't showing up. turns out, it's usually down to how ahoy is configured, or rather, how it's *not* configured to handle those tokens.

ahoy, by default, uses cookies to track visits and visitors. it saves these tokens, `visit_token` for each browsing session and `visitor_token` for each unique browser/device. however, it also filters out certain requests to avoid polluting your analytics with bot traffic, API calls, or things that generally don't represent genuine user activity. things that ahoy discards in development environment. that's pretty handy in production, but can be frustrating in development.

now, there are a couple of places where this can go wrong, the first and most common issue is the `config.filter_parameters` option. If you see in your application.rb, or any initializer you might have `config.filter_parameters` you should take a look there. It might have patterns that match `visit_token` or `visitor_token` preventing them from being stored in the db.

another place where this can happen is within the `ahoy.rb` initializer, usually located in the `config/initializers` directory. this file allows us to set certain options in ahoy, and there might be some options there blocking the information from being recorded. Usually what we want to see is something like this:

```ruby
#config/initializers/ahoy.rb

class Ahoy::Store < Ahoy::Stores::ActiveRecordStore
end

Ahoy.configure do |config|
    # your other configurations like api_only and cookie_domain

  # Disables filtering for these tokens to ensure storage
  config.filter_parameters = []
end
```

this snippet ensures all `params` are being stored. which is great in development. In production we will want to be a bit more selective, especially with sensitive data.

you also mentioned the geolocation not storing data in development/production. this, again, is usually a configuration problem but it also comes with a few gotchas.

first, let's talk about how geolocation generally works with ahoy. ahoy does not implement geolocation functionality in itself, but it includes methods for storing a few geolocation attributes. You would be using another service or gem in order to gather geolocation data from an IP and then, send it to ahoy for storage. things like `geocoder` are often used. So, we can start from there.

the first thing i would check is that `geocoder` is correctly configured. that means you have an api key to the provider you have chosen and it is correctly stored in your application.yaml or whatever setup you are using for managing env variables.

once we have that sorted, let's dive into the potential reasons why we don't see the data saved in the database. one of the frequent reasons why we see no data at all in our database is related to how are the gem hooks working. if the `ahoy` gem is being initialized after the `geocoder` gem is trying to access the data, it won't be available. if the `ahoy` gem is trying to do that before a request has even occurred, also we will have problems. it is an initialization race. one of the ways we could address this is by using an `after_action` filter inside the `application_controller`. something like this:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  after_action :store_geolocation

  private

  def store_geolocation
    if current_visit
      ip = request.remote_ip
      begin
        result = Geocoder.search(ip).first
        if result
           current_visit.update(
            country: result.country,
            region: result.state,
            city: result.city
            )
        end
      rescue => e
         Rails.logger.error "Geolocation error: #{e.message}"
      end
    end
  end
end
```
this will guarantee the data will be updated once the request has occurred, and once the visit has been created.

finally, another reason why this might not work, and i know this one because it bit me in the butt multiple times, is that your development and production environment might be configured to use different databases. or might be using different credentials to communicate to the same database. when i started, i used to use `rails db:migrate` and forgot to do the same in production or viceversa, leading to a situation where my development environment would have the latest schema and production would have a different one. you would get errors when updating your database and it's not something we always notice in time. make sure that the migrations have been applied in production also.

you can check if the data is being stored in your database by running a simple `rails console` in your development environment and query the `Ahoy::Visit` table. `Ahoy::Visit.last` should retrieve the most recent visit and you will be able to see if there is geolocation information associated to that record.

to summarize, these are the main things to verify:

1.  double-check your `config.filter_parameters` inside `application.rb`, and the ahoy initializer. make sure you are not filtering any `params` containing the tokens in development.
2.  make sure you have correctly configured `geocoder` with its API key and that your `application.yaml` or `.env` files are correctly configured.
3.  ensure you are running all the migrations in production or staging and that your databases are aligned.
4.  try a filter like the above, that way you are guaranteed to execute the update to the geolocation attributes when all the information is available.

a final note, if you want to expand your knowledge on these topics, i would strongly recommend checking out “building a real-time web analytics application” by lucas da costa and “professional ruby on rails” by noel rappin. these books are a good solid base in rails that would expand your understanding of the underlying mechanisms of these types of problems.

one last thing, there's a fine line between an early adopter and an early debugger, sometimes i wonder if i should just become a professional debugger, it pays well.
