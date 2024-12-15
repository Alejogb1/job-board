---
title: "How to do a Rails Serializer Return Previous Data If Statement Exist?"
date: "2024-12-15"
id: "how-to-do-a-rails-serializer-return-previous-data-if-statement-exist"
---

Alright, so you're looking at how to handle a situation in your rails serializers where you want to include previous data if a certain attribute exists, right? I've been down this road myself, more times than i care to remember. It's a pretty common scenario, especially when dealing with audit trails, versions, or basically anything that has a history. let's break it down with some code examples and how i've tackled similar problems in the past.

Basically, your serializer needs to know if the current object has a particular associated object, or an attribute, or some kind of conditional existence, and based on that, fetch data from the *previous* instance, or just return nothing, or return the default data, or something else. It all depends on the specific use case.

I will try to give you a few solutions based on my past personal experience, focusing on what I think works.

First off, let's assume we have a model called `widget` and we are keeping track of changes using something like `paper_trail` gem (other similar gems would work as well) this is pretty common. paper_trail is a good library and i recommend using it. We have a `widget_serializer` we are using in our api and want to show the previous `name` if this `name` exists on a previous version.

The simplest case might be a plain attribute on the model, let’s use `name` like i mentioned. You want to include the *previous* `name` value if it exists. This assumes we're using `paper_trail` (or something equivalent).

Here's a basic approach with paper_trail:

```ruby
# app/serializers/widget_serializer.rb
class WidgetSerializer < ActiveModel::Serializer
  attributes :id, :name, :previous_name

  def previous_name
    previous_version = object.previous_version
    previous_version&.name
  end
end
```

in this first snippet, you have an attribute named `previous_name` which executes the method `previous_name`. the method retrieves the previous version of the current object with `object.previous_version` and then if this version is valid (`&.`) it returns the `name` attribute of the previous version. if there is no previous version the whole `previous_version&.name` statement will return nil. this avoids `NoMethodError` exceptions. The `&.` operator is pretty cool. It's called the safe navigation operator. It’s a relatively modern addition to ruby, but has made my life so much easier. you should study it if you don't know it.

This is the easiest case. you have a `previous_version` and you want one attribute, which is usually a good way to start when you want to return previous versions, but you have to understand that `object.previous_version` actually performs a query to your database, so calling this many times can become an issue. it is not performatic. Let's see another case.

Now, let's imagine things are a little more complex and you don't want to repeat queries every time. Suppose our widget also has an association, say `widget_configuration` which also has versioning. and let's say you are displaying a lot of widget attributes at once. it would be inefficient to query `previous_version` many times for each attribute. I had exactly this problem a while ago, some code was running really slow, and i had to optimize it. the easiest solution is to fetch the previous version one single time. Here's how i did it:

```ruby
# app/serializers/widget_serializer.rb
class WidgetSerializer < ActiveModel::Serializer
  attributes :id, :name, :previous_name, :widget_configuration_name, :previous_widget_configuration_name

  def previous_version
    @previous_version ||= object.previous_version
  end

  def previous_name
    previous_version&.name
  end

  def widget_configuration_name
    object.widget_configuration&.name
  end

  def previous_widget_configuration_name
     previous_version&.widget_configuration&.name
  end

end
```

Here, we're caching the `previous_version` in an instance variable `@previous_version` so that subsequent calls reuse the already fetched previous version. Also, see that we can apply the `&.` safe navigation operator multiple times in the chain. so if the `previous_version` is nil, `previous_version&.widget_configuration&.name` will return nil, and if the `widget_configuration` is nil in the `previous_version`, it will also return nil. And it works without exceptions.

One other detail here is that `widget_configuration_name` now uses `object.widget_configuration&.name`. this is because if the widget doesn't have a `widget_configuration` it should return `nil` and not throw an error. This is a good practice if you do not want your api to crash because it has missing associations. the `&.` operator handles the `nil` association gracefully. I really appreciate the ruby language for this kind of feature.

But sometimes we need something even more complex.

Let's say that we only want to display the previous configuration if the current widget actually *has* a configuration. You are now dealing with conditional existence of an association and the conditional display of previous version attribute.

```ruby
# app/serializers/widget_serializer.rb
class WidgetSerializer < ActiveModel::Serializer
  attributes :id, :name, :previous_name, :widget_configuration_name, :previous_widget_configuration_name

  def previous_version
    @previous_version ||= object.previous_version
  end

    def previous_name
    previous_version&.name
  end

  def widget_configuration_name
    object.widget_configuration&.name
  end

   def previous_widget_configuration_name
    if object.widget_configuration.present?
       previous_version&.widget_configuration&.name
    else
      nil
    end
   end
end
```

In the last example we added a `if` condition before returning the previous value. here `object.widget_configuration.present?` checks if there is a `widget_configuration`. if it exists, we return the previous version, otherwise we return nil. This is important in our case because we are trying to return previous data if the current data exists. if there is nothing there, we return nil, but you can return any default value if you want.

In my experience, the trickiest part of this is often not the code itself, but understanding exactly *what* you want to return when that previous data doesn't exist or when a condition is not met. sometimes, you actually want to return an empty string, or a default value instead of `nil`, you have to be very precise about the specifications and business needs.

Also, when you're working with more complicated nested associations, things can quickly become quite messy and difficult to follow, so it's always a good idea to break down complex logic into smaller methods, which makes it easier to test and understand. it's easier to debug and easier to extend in the future. Also, it makes the serializers themselves look way cleaner, and not a huge chunk of unorganized code.

One thing i learned the hard way is always think about performance, when dealing with history or previous data. you should avoid executing the same queries multiple times. i remember one time i had to query a previous version for each attribute inside a serializer, and the api was really, really slow. so always fetch the data only once and use caching for the result. it's always a good idea.

Also, if you want to go deep into the performance issues of active record i recommend the book "rails performance" from nathan hopkins, very good book, really improved my understanding of how active record works, its lazy loading, caching, and many other important details.

Another thing you should know is that sometimes dealing with this kind of problem can be complex, but also quite funny, one time I was so confused i added a `pry` inside a serializer, and i completely forgot, and then the api started hanging because it was waiting for me to debug, i spent like 15 minutes wondering why it was not responding, so always check your code for unexpected `pry` calls. it's a silly but common mistake i do from time to time.

To wrap up, that's my usual way to deal with this situation. you start with simple code, try caching and then add conditional logic if needed. Also i always try to follow single responsibility principle and write small isolated methods to keep the code organized and testable. i hope this helps.
