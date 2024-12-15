---
title: "How to integrate calendy in ruby on rails application?"
date: "2024-12-15"
id: "how-to-integrate-calendy-in-ruby-on-rails-application"
---

alright, let's talk about calendly integration in rails. i've been down this road a few times, and it's usually not too bad once you get the basic flow.

so, at its core, calendly doesn't provide a straightforward api that allows you to, let's say, directly create appointments *within* your app in a fully customized way. instead, what we are dealing with is embedding their functionality through their api via iframes or redirects and some webhooks for status updates. i’ve seen folks trying all sorts of complex hacks to get around this but ultimately, it's usually better to just leverage their system, with an understanding of its limitations. trust me, i’ve been there trying to force a square peg into a round hole.

in my experience, the typical integration involves a few key aspects: generating user-specific calendly links, embedding calendly schedulers into your app, and handling webhook notifications for events created through calendly.

let's break it down, starting with generating personalized links. calendly allows you to pre-populate some information into your booking flow by using query parameters. this is what we want to achieve, generating unique links. i had a project, a few years back, a small medical app where each doctor had their own calendar, and we needed to redirect patients to a calendly link. initially we tried using a generic link with calendly, and it was a mess, all appointments ended up in a single calendar. we then started to use personalized links, things became easier.

here’s a basic ruby example of how you could construct such a link:

```ruby
def calendly_link(user, event_type_uuid)
  base_url = "https://calendly.com/your_organization/#{event_type_uuid}"
  query_params = {
    'name' => user.full_name,
    'email' => user.email,
    'a1' => user.id # or some internal identifier
  }
  "#{base_url}?#{query_params.to_query}"
end

# usage
user = User.find(1)
event_uuid = 'your-event-uuid' # find it in calendly interface
link = calendly_link(user, event_uuid)
puts link #output something like: https://calendly.com/your_organization/your-event-uuid?name=john%20doe&email=john.doe@example.com&a1=1
```

this example assumes you have a `user` model with `full_name` and `email` attributes. replace `your_organization` and `your-event-uuid` with your actual values from your calendly account. the `a1` param can be used for passing any unique id to your system, but it has to be used when dealing with webhooks too.

now, let's say we want to embed calendly within our rails app, avoiding direct redirects. this is where the iframe approach comes in. we can use a simple view helper like this:

```ruby
def calendly_iframe(calendly_link, options = {})
  iframe_options = {
    'src' => calendly_link,
    'width' => '100%',
    'height' => '600px', #adjust as needed
    'frameborder' => 0
  }.merge(options)
  content_tag(:iframe, '', iframe_options)
end

# in your view:
# <%= calendly_iframe(calendly_link(@user, @event_uuid) %>
```

this helper creates an iframe pointing to the generated calendly link. you can adjust the `width` and `height` as needed to suit your layout. during this medical app development process that i mentioned, we had to tweak this several times to make it look good within our app's design. the 'frameborder' attribute is set to 0 by default to avoid iframe borders.

finally, webhooks, these are important for keeping your system in sync with events happening in calendly. calendly will send http post request to a url that we must configure in calendly. in rails, you can set up a route and controller action to receive these. one issue that i have seen a lot is that people forget to handle all types of webhooks events, which results in missing information. so, it is important to handle all the webhooks types you need to properly. if you just handle one or two then your application will be out of sync.

here's an example controller action for handling calendly webhooks:

```ruby
class CalendlyWebhooksController < ApplicationController
  skip_before_action :verify_authenticity_token # important for webhook endpoints

  def create
    payload = request.body.read
    event_type = params['event']
    puts "webhook type: #{event_type}" # good for debugging
    case event_type
    when 'invitee.created'
      handle_invitee_created(payload)
    when 'invitee.canceled'
      handle_invitee_canceled(payload)
      #add others here 'event.created' and 'event.canceled'
    else
     # handle or log other events
     puts 'unknown event type'
    end
    head :ok # respond to the webhook with 200 ok, other wise calendly will keep trying
  rescue StandardError => e
    Rails.logger.error("Error processing calendly webhook: #{e.message}")
    head :bad_request
  end

  private

  def handle_invitee_created(payload)
    data = JSON.parse(payload)
    # extract the information you need and update your database.
    # if using 'a1' param use that to identify the user.
    # for example, data['payload']['invitee']['a1'] will be the user id you passed.
    # or data['payload']['event']['uri'] can be used to get more details from calendl api
    puts "invitee created for user_id: #{data['payload']['invitee']['a1']}"
    puts data.inspect #good for debugging and see what is inside the request

    #update your user appointment table or whatever table you need
    # Appointment.create(user_id: data['payload']['invitee']['a1'])
  end

  def handle_invitee_canceled(payload)
     data = JSON.parse(payload)
     # do something else, cancel the appointment or whatever
    puts "invitee canceled: #{data.inspect}"

  end

end
```

this controller action receives the webhook payload, checks the event type, and then calls specific handler methods. in the `handle_invitee_created` method, you would parse the json payload and extract the necessary information. the `payload` object contains all the data you passed to the link, like the `a1` parameter and also the information about the appointment made. from here you can create an appointment or do whatever you need in your application.

note the `skip_before_action :verify_authenticity_token` line, that's important because webhooks are coming from outside our application. you also need to make sure you handle errors properly and return the correct http status, otherwise calendly might retry the webhooks causing duplicate appointment creation.

one thing i learned the hard way was always adding a logging system to help track webhooks responses.

now, in terms of resources for further reading, i would recommend checking out calendly's own developer documentation. it's pretty good and contains a lot of information on specific api calls. i also found that some books on api integration helped me to understand more general concepts. like "building microservices" by sam newman, even though this is not a direct calendly guide, it explains a lot about event-driven architecture that is necessary for working with webhooks.

remember this: calendly integration is not a one-size-fits-all solution. the approach will vary depending on your specific application needs. you might have a more complex workflow than the examples here, especially if you are trying to have a fine-grained control. for instance, if you need to modify the appointment on calendly side, or cancel it from your application you need to use the calendly api and make sure your application is in sync with calendly. or if you are trying to change dynamically the calendly event types you need to make a couple more of api calls. this approach is quite useful.

one funny thing happened to me. a client asked if we could make it so that after the user schedule a meeting, that all the data from the appointment will magically appear in the application database. i thought “sure, i'll just wave my magic wand”. *sigh*. webhooks is the answer here, but i think the client had different expectations.

i know this was a bit long, but i hope it covers some ground and gives you a better understanding of how to integrate calendly in a rails application. just remember to test each component thoroughly before going to production. let me know if you have any other questions.
