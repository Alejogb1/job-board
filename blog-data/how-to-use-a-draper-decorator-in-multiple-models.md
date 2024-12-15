---
title: "How to use a Draper decorator in multiple models?"
date: "2024-12-15"
id: "how-to-use-a-draper-decorator-in-multiple-models"
---

hey, i've seen this kind of thing a few times, and it's a pretty common spot where people get stuck when they're starting to use decorators, especially with something like draper. the core issue, from what i gather, is that you have multiple models and you're looking for a way to apply the same decoration logic without copy-pasting or making a tangled mess of includes and inheritances. it's totally relatable, been there done that.

let's get this clear first, a decorator’s job, in the draper sense, is to add presentation-specific logic to a model object. it keeps things organized by separating concerns – the model handles data, and the decorator handles how that data appears. having that in mind, directly applying a draper decorator to multiple models is not really how draper is designed to work. each model typically has its own specific representation needs, meaning each model will have one associated decorator. if you try to go around this you usually end up with a single decorator that knows too much about specific models and conditional logic all over the place, making it difficult to maintain and reason about in the long run.

what you’re probably actually after is reusing some of the decoration *logic* across multiple decorators. that's a different story and it's where things start to get interesting. there are a few different ways to approach this. the best one depends on the specific pattern of reuse that you have in mind.

one common pattern is to use what i would call, i'd say, "decorator mixins" or shared modules. you can extract the common methods into a module and then include that module into your specific decorators. this has worked really well for me back in the day when i had to format timestamps in several views of different models. the core method was the same but they had different model names. let’s say you have a `User` model and a `BlogPost` model, and both need formatted dates but are models that have completely different attributes. here's an example of how you might do this:

```ruby
# app/decorators/concerns/date_formatter.rb
module DateFormatter
  def formatted_created_at
    created_at.strftime("%Y-%m-%d %H:%M:%S")
  end

  def formatted_updated_at
    updated_at.strftime("%Y-%m-%d %H:%M:%S")
  end
end

# app/decorators/user_decorator.rb
class UserDecorator < Draper::Decorator
  delegate_all

  include DateFormatter

  def full_name
    "#{object.first_name} #{object.last_name}"
  end
end

# app/decorators/blog_post_decorator.rb
class BlogPostDecorator < Draper::Decorator
  delegate_all

  include DateFormatter

  def short_excerpt
    object.content.truncate(100)
  end
end
```

in this example, both `UserDecorator` and `BlogPostDecorator` gain access to `formatted_created_at` and `formatted_updated_at` methods. this pattern is clean, allows you to keep your decorators focused, and is easy to test. i used to do this a lot and it allowed me to reduce duplicate code dramatically across different models decorators. remember that the shared logic has to be agnostic to specific model attributes, relying on the delegated ones (if any) and attributes that are consistently present across your models, such as `created_at` and `updated_at` in the example above.

now let's say you have methods that rely on specific attributes of a model or have slightly different behaviors per model. this is where inheritance can be useful. you can create a base decorator that contains the general logic and then create specific decorators that inherit from it, extending or overriding methods as needed. let's see an example of this where the formatting may differ:

```ruby
# app/decorators/base_date_decorator.rb
class BaseDateDecorator < Draper::Decorator
    def formatted_date(date_attribute)
        object.send(date_attribute).strftime("%Y-%m-%d") if object.respond_to?(date_attribute)
    end
end


# app/decorators/user_decorator.rb
class UserDecorator < BaseDateDecorator
    delegate_all
    def formatted_created_at
        formatted_date(:created_at)
    end
    def formatted_updated_at
        formatted_date(:updated_at)
    end

  def full_name
    "#{object.first_name} #{object.last_name}"
  end
end


# app/decorators/blog_post_decorator.rb
class BlogPostDecorator < BaseDateDecorator
    delegate_all
     def formatted_created_at
       formatted_date(:published_at)
     end
     def formatted_updated_at
       formatted_date(:modified_at)
     end
  def short_excerpt
    object.content.truncate(100)
  end
end
```
in this case we have a base class and specific models inheriting. i find this useful when you have some shared logic, like in our example the formatting date logic, but models may have different attributes. in this example the `formatted_date` method takes a date attribute as argument and can be used on different models without caring too much about the attribute name. this avoids having to write each time a slightly different date formatting method.

another, slightly more advanced approach, is to use a “decorator factory” pattern. this allows you to build decorators dynamically based on some configuration or model-specific parameters. i personally have not used this too much and find it a bit too much for most use cases, but it can be very handy in cases where the decoration logic differs dramatically across several models and a lot of logic or parameters are shared. let's say that for some obscure reason each model had it's own date formatting style. i have never found this case, but let's pretend.

```ruby
# app/decorators/decorator_factory.rb
class DecoratorFactory
    def self.build(model_class)
        klass = Class.new(Draper::Decorator)
        klass.delegate_all
        case model_class.name
        when "User"
            klass.define_method(:formatted_date) do |date_attribute|
              object.send(date_attribute).strftime("%m/%d/%Y") if object.respond_to?(date_attribute)
            end
            klass.define_method(:formatted_created_at) { formatted_date(:created_at)}
            klass.define_method(:formatted_updated_at) { formatted_date(:updated_at)}
            klass.define_method(:full_name) { "#{object.first_name} #{object.last_name}" }
        when "BlogPost"
             klass.define_method(:formatted_date) do |date_attribute|
              object.send(date_attribute).strftime("%d-%m-%Y %H:%M") if object.respond_to?(date_attribute)
            end
             klass.define_method(:formatted_created_at) { formatted_date(:published_at) }
             klass.define_method(:formatted_updated_at) { formatted_date(:modified_at) }
             klass.define_method(:short_excerpt) { object.content.truncate(100) }
        else
            raise "no decorator found for #{model_class.name}"
        end
        klass
    end
end

# app/controllers/users_controller.rb
def show
    @user = User.find(params[:id])
    decorator_class = DecoratorFactory.build(User)
    @decorated_user = decorator_class.new(@user)
end

# app/controllers/blog_posts_controller.rb
def show
    @blog_post = BlogPost.find(params[:id])
    decorator_class = DecoratorFactory.build(BlogPost)
    @decorated_blog_post = decorator_class.new(@blog_post)
end
```

in this case you just have a factory that generates a decorator for each model by adding dynamically the logic. this should be used when the logic is complex or involves multiple parameters that may not be easily passed via simple `include` or inheritance patterns. this is pretty much what ruby does under the hood when generating classes. and it can make things easier when trying to achieve more custom things.

the factory pattern is pretty complex and may be overkill for most cases, so it’s usually better to start with simple includes. just remember to keep your decorators focused on presentation logic. don't mix model logic with presentation logic and keep your decorators thin and testable.

in all of these scenarios, you’re not applying *the same decorator* to multiple models but rather reusing decoration *logic* across multiple decorators, which is the way to go if you ask me, as i've learned by trial and error in the past.

as for resources, i would highly recommend the “practical object-oriented design in ruby” by sandy metz, a classic in the ruby community, it tackles this kind of topic in general and provides a good overview of how classes should behave in general. the “rails anti-patterns” book can also give you insight into what things to avoid, although it might be a bit too much for what you're asking right now. another book, “refactoring: improving the design of existing code” by martin fowler also offers good insights into code smells and good ways to structure your code. most of the things that are mentioned there are applicable to the case at hand.

finally and just because, if a programmer has a bad day he might feel depressed but if he has a terrible day he may feel re-pressed, that's my silly joke of the day.

i hope this helps you and feel free to ask any more specific questions if you need more help on this topic.
