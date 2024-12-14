---
title: "With Ruby On Rails, how to pass params through a custom before_validation function?"
date: "2024-12-14"
id: "with-ruby-on-rails-how-to-pass-params-through-a-custom-beforevalidation-function"
---

alright, so you're hitting that classic rails params puzzle with `before_validation`, huh? i've been there, staring at the screen, wondering why my data isn't flowing the way i expected. seems like a simple enough thing, but it can get tangled up pretty quickly.

let me break it down based on how i've tackled this in the past. first off, the `before_validation` callback in rails runs *before* the model tries to validate its attributes. that's key. at this stage, you're working with the attribute data already set on the model instance, not the raw params hash from the request. the request's params get used to populate the model's attributes *before* `before_validation` is triggered. so, directly manipulating the params hash inside a `before_validation` isn't really how it's designed to work.

what you *can* do though is modify the model attributes inside the `before_validation` callback, and rails validation will act accordingly after these changes. you're essentially transforming the model's data before its validated.

let's imagine a scenario. say you have a user model, and you want to ensure that the email address is always lowercased before it's saved. you could try to hack away at the raw params which would be the wrong way. let's do it the proper rails way:

```ruby
class User < ApplicationRecord
  before_validation :downcase_email

  validates :email, presence: true, uniqueness: true, format: { with: URI::MailTo::EMAIL_REGEXP }

  private

  def downcase_email
    self.email = email.downcase if email.present?
  end
end
```

here, the `downcase_email` method is called before validation. it grabs the current value of the `email` attribute, lowercases it if it exists, and sets the attribute back on the model instance. this adjusted email is then used for validation and database storage. notice, we didn't touch the original params hash at all.

now, lets assume we've got a different problem, perhaps you have a json payload as one of the attributes on your model, and it comes in a certain format and you need to extract some info from it to populate another field before validation.

lets assume this scenario: you have `settings_json` attribute on your model, and we want to grab some config value from there and fill the `config_item` field before validation. here's how i'd approach that.

```ruby
class Configuration < ApplicationRecord
  before_validation :extract_config_item

  validates :config_item, presence: true

  private

  def extract_config_item
    return unless settings_json.present?
    parsed_settings = JSON.parse(settings_json)
    self.config_item = parsed_settings['some_key'] if parsed_settings.is_a?(Hash) && parsed_settings.key?('some_key')
  rescue JSON::ParserError
   # i've seen some weird json in my days, let's log and swallow.
    Rails.logger.warn("Invalid json for settings_json: #{settings_json}")
  end
end
```

in this example, if the `settings_json` attribute is populated we try to parse it, and if the parsing is successful we look for the `'some_key'` key. and grab it. and use the value to set the `config_item` value. this all happens before the model validation kicks in. this way, the validation has all values updated. i've seen this type of scenario happen more than once, when dealing with api data coming into the system. json parsing, handling potential errors... a classic, it feels like every other project requires it. it's important to check if a key exists in a hash, specially if the input comes from a third-party system, you never know what might end up there. so it's crucial to write the code defensively.

and let's throw one more scenario to see other options. lets assume we have a `Product` model and you want to create some `sku` automatically before validation based on the name and some custom logic.

```ruby
class Product < ApplicationRecord
  before_validation :generate_sku

  validates :sku, presence: true, uniqueness: true
  validates :name, presence: true

  private

  def generate_sku
    return if name.blank? || sku.present?
    sanitized_name = name.downcase.gsub(/[^a-z0-9]/, '-')
    self.sku = "#{sanitized_name}-#{SecureRandom.hex(4)}"
  end
end
```

in this example we generate the sku by sanitizing the name by removing special chars and appending a random hex string, this is a very simplified example for creating an sku but i've used similar methods in many of my projects. the random hex part is important to reduce the chances of collision. this method only runs if the `name` is present and the sku is blank. so, if someone is trying to set an sku manually on purpose, this method will not change it.

now, about those params. while you shouldn't be directly altering params within `before_validation`, sometimes you *do* need to access the request parameters, say, to apply logic conditionally. you have a few routes for that. you could inject these params into the model before saving/validation by using a custom setter on the model, like this:

```ruby
class SomeModel < ApplicationRecord
  attr_writer :some_param

  before_validation :do_something_with_param

  private

  def do_something_with_param
      if @some_param.present?
        # do some custom logic based on the param.
        # for example setting an extra attribute
        self.some_calculated_attribute = @some_param + 'some static value'
      end
  end
end
```
now, you can in the controller set the `some_param` by calling the setter method on the model instance like `model.some_param = params[:some_param]` right before saving or validating the model. and then your model can use it before the validation phase, this is something i've had to use before with some older rails api's. there are other ways, but this tends to be the less troublesome.

it all boils down to understanding the rails request lifecycle: params come in, get used to populate the model's attributes, and *then* the `before_validation` callbacks fire. you can't change the params directly in those callbacks. your focus should be on manipulating the model's attributes within the `before_validation` callbacks or passing parameters by using a custom setter before saving/validating.

if you are looking for more in-depth stuff about the lifecycle of rails requests and their callbacks, i'd point you towards "agile web development with rails" by sam ruby. its a classic, and it still provides the best insights for the framework. also, if you ever find yourself needing to debug some weird behavior in the lifecycle of models on rails, i'd suggest reading the "rails internals" section of the "rails 5 way" by obie fernandez, sometimes you will need to get into the nitty-gritty of the framework to figure out what's happening.

one little piece of wisdom i've gained over years of doing this is that sometimes the simplest solution is the one you should go for, rails provides a lot of ways to do things, and sometimes there's a temptation to use the most complex way, when really, the simpler approach is the more maintainable one. and hey, remember, a poorly configured database is like a bad relationship: it's never gonna validate your happiness. (i had to, sorry).

hope this helps! let me know if there are other questions
