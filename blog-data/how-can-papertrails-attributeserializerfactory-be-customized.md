---
title: "How can Papertrail's AttributeSerializerFactory be customized?"
date: "2024-12-23"
id: "how-can-papertrails-attributeserializerfactory-be-customized"
---

Let's tackle this. Customizing Papertrail's `AttributeSerializerFactory` is something I've delved into a few times, particularly when dealing with complex data types in our audit logs. The default serializers are, of course, handy, but inevitably, you encounter situations where you need more control over how specific attributes are represented when changes are tracked. The goal, naturally, is maintaining readable and useful audit trails without losing critical information. Let’s explore the avenues for customization.

The `AttributeSerializerFactory` in Papertrail, as you likely know, is responsible for selecting the appropriate serializer based on the type of attribute being tracked. By default, it handles common types like strings, numbers, booleans, and dates pretty well. But, when you venture beyond that – think custom enumerations, complex objects, or even serialized data – things get interesting. It’s at this point that extending or overriding its functionality becomes essential.

My own experience with this dates back to a project where we were using a custom `Point` object to represent geographical coordinates. Simply serializing these as `[x, y]` wasn't very user-friendly in the audit log, especially when we needed to quickly understand the context of changes. This led us to implement our own custom serializer, and it's a journey I’m glad to share.

There are two primary methods for customization: creating entirely new serializers or customizing existing ones through options. We’ll cover both.

Firstly, creating a custom serializer involves implementing the `PaperTrail::Serializers::Serializer` interface. This typically involves defining two main methods: `serialize(object)` and `deserialize(value)`. The `serialize` method is responsible for transforming the object into a format suitable for storage in the audit log, and the `deserialize` method reverses this process when retrieving historical versions.

Here's a straightforward example of a custom serializer for our `Point` object, written in ruby as it would apply to the Papertrail gem:

```ruby
# Custom Point Serializer
class PointSerializer < PaperTrail::Serializers::Serializer
  def serialize(point)
    return nil unless point.respond_to?(:x) && point.respond_to?(:y)
    "#{point.x},#{point.y}"
  end

  def deserialize(value)
    return nil if value.nil?
    x, y = value.split(',').map(&:to_f)
    Point.new(x, y)
  rescue
    nil
  end
end

class Point
  attr_accessor :x, :y
  def initialize(x,y)
    @x=x
    @y=y
  end
end
```

In this snippet, the `serialize` method transforms a `Point` object into a comma-separated string of its x and y coordinates. The `deserialize` method takes this string and reconstructs a `Point` instance. Error handling is crucial here, as malformed data from older audit entries can occur due to schema changes over time. We use the rescue to catch any issues when deserializing the value so that we can deal with them gracefully.

Once you have this custom serializer, you need to register it with the `AttributeSerializerFactory`. This is done using the `register_serializer` method:

```ruby
PaperTrail.config.attribute_serializers.register_serializer(Point, PointSerializer.new)
```

Now, whenever Papertrail encounters a `Point` object as an attribute being tracked, it will use the `PointSerializer` to handle the serialization and deserialization.

However, creating completely custom serializers isn't always necessary. Sometimes, you might just need to tweak the existing serializers. Take dates, for instance. By default, Papertrail stores dates in a standard iso format, but suppose you need to store the timestamps in a more compact unix timestamp representation. You can customize the existing `DateSerializer` using options within `attribute_serializers`:

```ruby
 PaperTrail.config.attribute_serializers.register_serializer(Date,
      PaperTrail::Serializers::DateSerializer.new(format: :unix_timestamp))
 PaperTrail.config.attribute_serializers.register_serializer(DateTime,
      PaperTrail::Serializers::DateTimeSerializer.new(format: :unix_timestamp))
```

Here we've used the standard `DateSerializer` and `DateTimeSerializer` but initialized them with an option, `:format` set to `:unix_timestamp`. This would then serialize our dates as a numerical representation of the date. This approach is particularly useful for common types, where you don’t need to write a complete serializer. Remember that most built-in serializers expose several options via their constructor to adapt their behavior without having to override the entire class.

Another instance where I found customization useful was with serialized json columns using ActiveRecord. We were storing configurations as json blobs, and when a configuration changed, the entire json string was recorded as a single diff. This was often unwieldy to understand, so I opted to customize the serialization process to produce a more granular record of change. Here, it wasn't a specific custom object but the data format. To achieve a granular diff, we first needed to parse the json string into an object, then diff that object, and finally serialize that diff.

```ruby
class JsonDiffSerializer < PaperTrail::Serializers::Serializer

  def serialize(json_string)
    previous_value = @last_value.nil? ? {} : JSON.parse(@last_value)
    current_value = json_string.nil? ? {} : JSON.parse(json_string)
    @last_value = json_string

    diff = Hashdiff.diff(previous_value, current_value)

    diff.present? ? diff : nil
  rescue JSON::ParserError
    json_string # If not valid json, just record the string
  end

  def deserialize(value)
    value
  end

  def prepare_for_update(model, attribute_name, before, after)
    @last_value = before
    after
  end
end
```

In this example, `JsonDiffSerializer` utilizes the `hashdiff` gem, and stores the previous value in the `@last_value` instance variable, it provides granular changes, rather than diffing the full json blob. The `prepare_for_update` is used by papertrail to set up the context needed to perform the diff. The `deserialize` here is a passthrough as we're not looking to rehydrate this diff into a usable value, but we are only interested in the changes.

You would then register it similar to the point example.

```ruby
 PaperTrail.config.attribute_serializers.register_serializer(String, JsonDiffSerializer.new, if: lambda { |attribute_name|
      attribute_name == 'configuration' # assuming the attribute is named 'configuration'
    })
```

This setup registers `JsonDiffSerializer` only when the attribute name is configuration. This is a valuable way to apply custom serialization to specific attributes without creating a custom class.

For diving deeper into the theoretical side of serialization and versioning, I recommend a few resources. Start with Martin Fowler's "Patterns of Enterprise Application Architecture," as it provides a solid foundation for understanding object-relational mapping and data persistence concepts that underpin systems like Papertrail. Also, "Designing Data-Intensive Applications" by Martin Kleppmann offers insights into data modeling and versioning strategies at scale, which is highly relevant when building audit logging solutions. Finally, the ActiveRecord documentation itself, particularly sections on serialization, can provide a deeper understanding on the inner working of the database interactions.

Ultimately, customization of Papertrail's `AttributeSerializerFactory` should be guided by the specific needs of your project. Starting with the provided serializers and then extending as necessary is a solid approach. The key, as always, is to balance the need for detailed audit data with usability and performance considerations.
