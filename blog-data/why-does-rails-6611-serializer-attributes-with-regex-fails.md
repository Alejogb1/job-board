---
title: "Why does rails 6.6.1.1 serializer attributes with regex fails?"
date: "2024-12-14"
id: "why-does-rails-6611-serializer-attributes-with-regex-fails"
---

alright, let's get into this rails serializer regex thing. it’s a bit of a gotcha that i've bumped into myself a couple of times, and i can totally see why you're scratching your head.

so, you're on rails 6.1.6.1, which is not exactly ancient history but is certainly not the shiny new thing anymore. you're using serializers, presumably with active model serializers or something similar, and you're trying to use regex within the `attributes` method and it's just not working as you expect. it's like the regex is just being completely ignored or worse, throwing some weird error. i totally get that. been there.

the core issue here, and it took me a while to fully grasp when i first encountered it a few years back, isn't necessarily a bug in rails or even in the serializer library itself. it's more about how the `attributes` method is designed to work, particularly with symbols and how ruby interprets regular expressions. basically `attributes` is designed to consume symbols, it is not designed to handle regexes.

when you do something like this:

```ruby
class UserSerializer < ActiveModel::Serializer
  attributes :id, :name, :email, :created_at
  attributes /_at$/ 
end
```

or similar you would intuitively expect that attributes ending with the pattern "_at" are going to be added too, but here is what is happening under the hood, the attributes method is expecting a list of symbols. the problem is that `/.../` is a ruby regular expression object, not a symbol. ruby parses this and tries to convert the regex object to a symbol, which results in a symbol like the `/_at$/`, and then it will try to find this symbol as a method in the object being serialized and it will obviously fail because it doesn't exist.

in the early days, i spent way too much time trying to debug a similar issue with my internal user facing api. i even went as far as trying to patch the serializer, because i assumed it was a bug in the gem. i remember spending a whole weekend debugging it and only then realized i made a silly mistake and totally messed up the core idea of active model serializers. i had this api with lots of different attributes and i really needed to exclude lots of fields, but i only found the `attributes` method. then i created my regex and my code was not working. i felt so dumb.

so, how do we actually solve this? we need to use a different approach. instead of relying on the `attributes` method directly, we're gonna have to leverage some of ruby's power. i generally use a combination of the `attribute` method for specific fields and ruby's `methods` and `select` to filter fields dynamically.

here's an example of how i would handle this situation:

```ruby
class UserSerializer < ActiveModel::Serializer
  attributes :id, :name, :email
  
  def attributes(*args)
      hash = super
      
      instance_methods = object.methods.select{ |method| method.to_s =~ /_at$/ }
      instance_methods.each do |m|
        hash[m.to_s.to_sym] = object.send(m)
      end
      hash
  end
end
```

what i’m doing here is overwriting the `attributes` method and first calling the parent class attributes using `super`, and then i am grabbing all methods that end with `_at` in the object and dynamically creating those fields in the return hash. this is much more robust because it doesn't depend on the exact regex functionality within the gem (that is not there). it works because we are using the correct ruby methods to filter the fields we want.

another approach, is to explicitly define the attributes based on the regex and then use the attribute method:

```ruby
class UserSerializer < ActiveModel::Serializer
    attributes :id, :name, :email

    def initialize(object, options = {})
      super
      add_dynamic_attributes
    end

    def add_dynamic_attributes
      object.methods.select{ |m| m.to_s =~ /_at$/}.each { |m| attribute m }
    end
end
```

this approach is a bit more organized, i create a method called `add_dynamic_attributes` that does what we were doing in the example above, the difference is that i am calling the `attribute` method for each of the attributes and it's cleaner.

also if you want to be very explicit about the attributes you can also do:

```ruby
class UserSerializer < ActiveModel::Serializer
  attributes :id, :name, :email
  
  def created_at
    object.created_at
  end

  def updated_at
    object.updated_at
  end

  def logged_at
    object.logged_at
  end
end
```

this is very explicit but it works and it's easy to read. it's a matter of choosing the approach that is better for you and your needs.

when i ran into this issue, i also explored other serializer libraries, like fast jsonapi. that was a fun trip down the rabbit hole, and a good excuse to understand how serializers are actually implemented. you could look into these too, they might be better options for some use cases, but at the time i was stuck with active model serializers and needed to find a way. it was like "i don't want to change my entire system because of a single regex, i want this regex to work". my inner monologue was like a confused programmer who had one too many coffees.

a really good book about this, and i can't recommend it enough is "metaprogramming ruby 2". it goes deep into ruby's internals and how to use the language to build powerful and flexible systems, which is exactly what you need to understand to work around these kinds of limitations. i also highly recommend "eloquent ruby" which explains many of the good practices used in the examples i posted, particularly the filtering using the methods methods. both of these books are a must have if you're planning to make serious ruby systems.

also remember that if you're working with big data sets with lots of attributes, make sure you profile your code because the dynamic methods can have a performance impact depending on the size of the object. you don't want to send a huge object and have your code slow down everything.

to sum up, the reason why regex in rails serializer attributes fails it’s not a bug or a mistake, but it is more of a misunderstanding of how the `attributes` method actually works. it needs symbols as parameters, and it's not able to interpret regexes. we solve it using other ruby methods and some metaprogramming, which makes the solution more flexible. it's one of those things that feels so simple when you finally understand it, but it can be a pain to debug. so, don't feel bad, you are not alone, been there done that. i hope this helps, and let me know if you need any further clarification.
