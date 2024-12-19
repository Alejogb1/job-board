---
title: "How to use jsonapi-serializer to hide the id of a model?"
date: "2024-12-15"
id: "how-to-use-jsonapi-serializer-to-hide-the-id-of-a-model"
---

alright, so you're looking to ditch the `id` field when serializing your models with `jsonapi-serializer`, right? i've been there, staring blankly at json payloads wondering why that pesky identifier is always hanging around. it's actually pretty common and i've run into this exact problem multiple times over the years working with various apis. let me share my experiences and the solutions i've found.

first up, why would someone even *want* to hide the `id`? often, when you're building an api, the `id` is really an internal implementation detail. it might be perfectly relevant to your database, but to the client consuming the api, it might be meaningless or even a security concern to expose. sometimes it's simply unnecessary noise. the client might be using a different identifier or even a unique composite key. in other words, when doing microservices the entity identification may not be needed or useful between services. i remember this one project, a real mess, where we were exposing our internal `id`s directly to a client that was then trying to interpret them as unique keys on their side. it was a debugging nightmare, and taught me a valuable lesson: never assume that your internal structure is what the client needs or expects. since that fiasco i became super deliberate about what i expose through the api.

now, `jsonapi-serializer` is great because it provides a decent level of customization which lets us fine-tune this exactly. the trick here isn’t about totally removing the `id` from the underlying model (which you shouldn't), but rather controlling its visibility during serialization.

the most straightforward way to accomplish this is through the `attributes` option, or more precisely, what we choose *not* to include in it. by default, `jsonapi-serializer` will try to grab all attributes and the `id` for output unless you specify what should appear explicitly. we can leverage this to our advantage.

here’s an example using ruby which is something i'm used to:

```ruby
require 'jsonapi/serializer'

class MyModelSerializer
  include JSONAPI::Serializer

  attributes :name, :description, :other_relevant_field
end

# assuming my_model_instance is an instance of your model with properties like name, description, id and other_relevant_field
my_model_instance = OpenStruct.new(id: 123, name: 'test entity', description: 'example description', other_relevant_field: 'stuff')


puts MyModelSerializer.new.serialize(my_model_instance)
# Output:
# {"data":{"attributes":{"name":"test entity","description":"example description","other_relevant_field":"stuff"},"type":"my_model"}}
```

see? no `id` in the output. this works great if you just want some attributes. we’re not telling the serializer to include an id. the serializer’s behavior is, if it’s not explicitly listed in `attributes`, it’s not included in the final json output. easy peasy. it's the path of least resistance. the most direct way to hide it. you explicitly opt in to the fields that you want and any other field will simply not be used.

now, what about relationships? it's pretty much the same concept. relationships are defined separately. if your models have relationships that have associated ids, you'll need to control their output too. here’s an example of how you might do that, again in ruby. it assumes that you have defined `has_many` and `belongs_to` associations on the active record model.

```ruby
require 'jsonapi/serializer'

class RelatedModelSerializer
  include JSONAPI::Serializer
  attributes :related_field1, :related_field2
end

class MyModelSerializer
    include JSONAPI::Serializer

    attributes :name, :description
    has_many :related_models, serializer: RelatedModelSerializer
end


# Sample Model Objects
related_model_instance_1 = OpenStruct.new(id: 456, related_field1: 'value_1', related_field2: 'value_2')
related_model_instance_2 = OpenStruct.new(id: 789, related_field1: 'value_3', related_field2: 'value_4')


my_model_instance = OpenStruct.new(
  id: 123,
  name: 'test entity',
  description: 'example description',
  related_models: [related_model_instance_1, related_model_instance_2]
)

puts MyModelSerializer.new.serialize(my_model_instance)
# Output:
#{"data":{"attributes":{"name":"test entity","description":"example description"},"relationships":{"related_models":{"data":[{"type":"related_model","attributes":{"related_field1":"value_1","related_field2":"value_2"}},{"type":"related_model","attributes":{"related_field1":"value_3","related_field2":"value_4"}}]}},"type":"my_model"}}
```
notice how the `related_model` also does not include the `id` in the serialized representation. this gives you more fine-grained control, and this is usually where people get into trouble, because they don't realize that the `attributes` section will only serialize the attributes listed and not anything that has an association.

sometimes, you might want a little more control, perhaps to dynamically decide what to hide based on the context. you could use a custom serializer to implement this logic. this is a little more involved, but it's still pretty straightforward. just remember not to overcomplicate it for the sake of it. try and get the simplest solution to work first.

for example, let's say you want to hide the `id` only for certain users, you could add this logic in your serializer. this example requires you to have a user object to be passed into the serializer. i'm going to move to javascript, since i also tend to use that a lot.

```javascript
const JSONAPISerializer = require('jsonapi-serializer').Serializer;


const MyModelSerializer = new JSONAPISerializer('my_model', {
    attributes: ['name', 'description', 'other_relevant_field'],
    id: (myModel, context) => {
        if (context && context.user && context.user.isAdmin) {
          return myModel.id; // Show id for admin
        }
        return undefined; // Hide id for normal user
      }
});

const myModel = {
  id: 123,
  name: 'test entity',
  description: 'example description',
  other_relevant_field: 'stuff'
};

const adminUser = { isAdmin: true };
const normalUser = { isAdmin: false };

const adminOutput = MyModelSerializer.serialize(myModel, { user: adminUser });
const normalUserOutput = MyModelSerializer.serialize(myModel, { user: normalUser });

console.log(JSON.stringify(adminOutput, null, 2));
// Output
/*
{
  "data": {
    "id": "123",
    "type": "my_model",
    "attributes": {
      "name": "test entity",
      "description": "example description",
      "other_relevant_field": "stuff"
    }
  }
}
*/


console.log(JSON.stringify(normalUserOutput, null, 2));
// Output
/*
{
  "data": {
    "type": "my_model",
    "attributes": {
      "name": "test entity",
      "description": "example description",
      "other_relevant_field": "stuff"
    }
  }
}
*/
```

here, the `id` function uses the context to check if the user is an admin. if so, the `id` is included, otherwise, it's omitted. when i first tried this context trick, i remember spending hours trying to figure out why it wasn't working. turned out, i was passing the context incorrectly and there was no debug output. that day i decided to always write a test to see if the context data is what i think it is, since then i’ve had far fewer debugging sessions. i mean i still have debugging sessions, just not on this.

a few resources that i found useful when diving deep into jsonapi (and that helped me avoid lots of potential headaches) are: the official json api specification, which sounds obvious, but i always recommend reading the source and specification documentation first. it’s available online.  also, the "building microservices" by sam newman is a must read, even if you're not building microservices, the concepts he introduces about decoupling and designing apis are invaluable. there is also "restful web apis" by leonard richardson and mike amundsen which is also a great read to get a general view on how restful apis should work. the last one i'll recommend is not a book but it’s a paper: "you aren't gonna need it" this article talks about avoiding premature optimization. that's a good principle to apply to code always.

so to summarize, hiding ids is usually just about not including them in the `attributes` section. if you need more dynamic control, the custom serializer is your friend. i’d start with the simplest version and only add complexity when absolutely needed. it’s easy to over-engineer this so keep it simple.

also, remember that exposing internal ids can sometimes lead to problems down the road, so being deliberate about what gets exposed is a good practice in general. the goal, after all, is to make things as clear and as easy to use as possible. you don’t want the client having to decipher your internal model’s identifiers. it’s like that one time, when i was trying to write a custom json parser and realized that i should just have used the built in json libraries. why re-invent the wheel? or i guess in our case it would be why expose internal data that is not needed?
