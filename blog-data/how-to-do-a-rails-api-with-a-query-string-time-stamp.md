---
title: "How to do a Rails API with a query string time stamp?"
date: "2024-12-14"
id: "how-to-do-a-rails-api-with-a-query-string-time-stamp"
---

alright, so you're looking at building a rails api that needs to handle a timestamp passed in as a query string. i've definitely been down this road, and trust me, it's pretty common. i'll walk you through how i usually approach this, keeping it straightforward and practical.

first off, let’s break down what’s actually happening. a query string timestamp is essentially a parameter passed within the url. for example, something like `/api/data?timestamp=1678886400`. the `1678886400` bit is the unix timestamp, basically, the number of seconds that have elapsed since january 1, 1970, at 00:00:00 coordinated universal time (utc).  it’s a universal way of representing a specific point in time, which is super useful when your api needs to work across different time zones.

so, how do we actually use this in rails? well, the first part is to make sure our controller can grab this parameter. it’s a very simple thing rails makes easy for us. in your controller, you would access this parameter using the `params` hash. for instance, the `timestamp` would be at `params[:timestamp]`.

now, here's the gotcha: those params are always strings. you’ll get `"1678886400"` not the number `1678886400`. we need to convert it to a date or time object for further processing. this is where rails' built-in methods come in handy. you will need to convert the string into an integer using `to_i` and then use `Time.at()` or `DateTime.strptime()` or similar to convert it into a time object, depending on what the timestamp type is.

let's take a look at a simple example. imagine you have an `api::v1::reports_controller`, and you want to return some data based on the given timestamp. i will use `time.at` here, but the other methods are equally valid and you can experiment with those options.

```ruby
# app/controllers/api/v1/reports_controller.rb
module Api
  module V1
    class ReportsController < ApplicationController
      def index
        timestamp = params[:timestamp]

        if timestamp.present?
          begin
            time = Time.at(timestamp.to_i)
            # do something with the time object, like filter data
            @reports = Report.where("created_at > ?", time)
            render json: { reports: @reports }
          rescue ArgumentError
              render json: { error: "invalid timestamp format" }, status: 400
            end

        else
          render json: { error: "timestamp parameter is required" }, status: 400
        end
      end
    end
  end
end
```

this snippet is pretty straightforward. we grab the `timestamp` from `params`, and check if it’s there. if so, we convert it to an integer and create a `time` object using `time.at`. then, we can use this time object to filter reports, or perform any kind of processing, in our case the filter. finally, we render the data as json or respond with the error.

i have a memory of building an api for historical data analysis. the database had millions of records and a query without a timestamp filter would just kill performance. it was a great reminder to always validate and sanitize parameters, and that a bad query is like a bad joke – it just doesn’t land well, you know?

now, let’s talk about error handling. in the above snippet, we’re using a `begin...rescue` block to catch potential `argumenterror` exceptions thrown by `time.at`. this is important because if someone sends a non-numeric value for the timestamp, like “yesterday”, the `to_i` method will return zero, and `time.at` will choke and throw that exception.  we don’t want our api to crash, so we catch it and respond with a helpful error message, along with a `400` status code.

but, what if you need to handle date time in different time zones, not only utc? that's a valid question, and you need to be mindful of this detail. the `time.at` is generally based on utc, so if you get a timestamp that represents a time in another zone, the conversion will be slightly off. the main idea here is to make sure your time representations are standardized across all of your application. usually this can be done by setting the timezone for your rails app, which can be done in `config/application.rb` file using `config.time_zone`. i highly recommend using utc and converting all of your times into utc for storage and processing internally, and only use the user specific timezone when the information is presented to the user.

here's an example that tackles the timezone situation, but it's a slightly different angle. lets say the timestamp represents the local time of some country.

```ruby
# app/controllers/api/v1/reports_controller.rb
module Api
  module V1
    class ReportsController < ApplicationController
      def index
        timestamp = params[:timestamp]
        timezone = params[:timezone] || 'utc' # default timezone

        if timestamp.present?
           begin
            time = Time.at(timestamp.to_i).in_time_zone(timezone)
            # do something with the time object, like filter data
            @reports = Report.where("created_at > ?", time)
             render json: { reports: @reports }
            rescue ArgumentError
              render json: { error: "invalid timestamp format" }, status: 400
            rescue ActiveSupport::TimeZone::UnknownTimeZone
              render json: { error: "invalid timezone" }, status: 400
            end
        else
          render json: { error: "timestamp parameter is required" }, status: 400
        end
      end
    end
  end
end
```

in this improved version, we are also accepting a `timezone` parameter and defaulting to utc. we are using `time.at` as before and then using `in_time_zone` to handle the conversion. if the timezone is not recognized an `activesupport::timezone::unknowntimezone` is raised and a proper error message is displayed. again, error handling is a must.

now lets dive into another use case, what if you need the timestamp to be a iso 8601 format date time instead of a unix timestamp. this is also a common need, for example, when interacting with systems that don't use unix timestamps.

```ruby
# app/controllers/api/v1/reports_controller.rb
module Api
  module V1
    class ReportsController < ApplicationController
      def index
        time_iso = params[:time_iso]

        if time_iso.present?
           begin
            time = DateTime.iso8601(time_iso)
            # do something with the time object, like filter data
            @reports = Report.where("created_at > ?", time)
             render json: { reports: @reports }
            rescue ArgumentError
              render json: { error: "invalid iso 8601 time format" }, status: 400
            end
        else
          render json: { error: "time_iso parameter is required" }, status: 400
        end
      end
    end
  end
end
```

in this code snippet, we changed the parameter name to `time_iso` to make it clear what we are passing. we also use `datetime.iso8601` instead of `time.at`. the parameter must be in iso 8601 standard format like `2024-03-17T12:30:00-05:00`. if the format is invalid the `argumenterror` will be caught and a error will be returned to the caller.

now, a couple of practical tips: always validate and sanitize user inputs. you never know what kind of data you'll get, so it’s always a good idea to check if the timestamp is within a reasonable range to avoid potential problems. also, i highly recommend checking the rails api documentation for `datetime` and `time` objects. this is the starting point of your research and will answer a lot of your questions.

another suggestion is to check the martin fowler’s “patterns of enterprise application architecture” book. while not directly related to handling timestamps, the book has a lot of knowledge about overall api design which you should be aware of when building a production app. also, read some articles about rest api design, how to properly return error messages, and the correct status codes and so on. there are many articles out there. but do not just randomly read, focus on good sources and check their content. you might also like the book "api design patterns" by jj geewax.

finally, remember, simple is better. when you add a lot of complexity you also add a lot of future maintenance. try to keep your code as simple and as clear as possible.  i hope this was helpful, good luck with your api!
