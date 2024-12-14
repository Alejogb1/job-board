---
title: "With Rspec, how to expect to receive with only a subset of key value pairs?"
date: "2024-12-14"
id: "with-rspec-how-to-expect-to-receive-with-only-a-subset-of-key-value-pairs"
---

alright, so you're hitting that classic rspec testing scenario, where you've got a complex object, maybe a hash or a response from some api, and you only care about asserting specific parts of it. been there, totally done that, probably more times than i can count.

i've seen this pop up a lot when dealing with integration tests where you're getting back a json payload, and the response has a bunch of fields that are frankly, noise for your current test case. you don’t want to assert on everything in case the backend team decides to add a new field (it's going to happen anyway, no doubt). you just need to make sure that the bits you care about are there and are what you expect.

early in my career, i remember a project where we were pulling data from a legacy system. the responses were a mess – imagine a huge hash with deeply nested data structures, unpredictable keys, and varying levels of importance. we tried asserting against the whole thing. it was a maintainability nightmare. a tiny backend change and half our test suite would fall apart. i learned quickly we needed a better way.

so, rspec doesn't give you a direct method called “expect_a_subset”, but we can absolutely achieve what you need with a few techniques.

the first and probably most straightforward approach is to use `include`:

```ruby
    it "checks for a subset of key-value pairs" do
      actual = {
        id: 123,
        name: "test user",
        email: "test@example.com",
        created_at: Time.now,
        updated_at: Time.now
      }

      expected_subset = {
        id: 123,
        name: "test user",
        email: "test@example.com"
      }

      expect(actual).to include(expected_subset)
    end
```

here, the `include` matcher does exactly what you need. it checks if the `actual` hash contains all the key-value pairs present in the `expected_subset` hash. it doesn't care about any other keys in the actual hash. this is very effective when you're mostly dealing with hash objects or things that behave like hash objects.

i have a funny story about using this, once we used this on some code that produced complex nested structures and at first we thought that it would check the whole nested structure but it did not, and we ended up with tests that were completely worthless, but after we found out the solution things were easy, this approach helped us in many occasions.

if you need to do more complex checking of the values themselves, where just checking that the value exists at the key is not enough, you can combine `include` with other rspec matchers. for example, you might want to use `match` on specific keys:

```ruby
    it "checks for a subset with matching values" do
      actual = {
        id: "some-unique-id",
        status: "active",
        count: 100,
        metadata: {
          source: "api",
          version: "1.2.3"
        }
      }

      expect(actual).to include(
        id: match(/some-/),
        status: "active",
        metadata: include(version: "1.2.3")
      )
    end
```

in this example, we're making sure that:

*   `id` matches a pattern.
*   `status` has to have a value equals to "active"
*   `metadata` itself includes a nested `version` field with a specific value.

this provides a lot more flexibility and allows you to do things like regex matches, type checks, etc.

now, sometimes you're not dealing with hash-like structures. let's say you've got an object that behaves a little bit differently, but you still need to check for certain attributes and their values.

you can define a custom matcher, which allows you to use a block with your own checking logic. this gives you complete control over the matching process.

```ruby
    RSpec::Matchers.define :include_attributes do |expected_attributes|
      match do |actual|
        expected_attributes.all? do |key, expected_value|
          if actual.respond_to?(key)
            actual.send(key) == expected_value
          else
            false
          end
        end
      end
    end

    it "checks for a subset of attributes" do
      class User
        attr_reader :id, :name, :email

        def initialize(id, name, email)
          @id = id
          @name = name
          @email = email
        end
      end

      user = User.new(456, "test user 2", "test2@example.com")

      expected_attributes = {
        id: 456,
        email: "test2@example.com"
      }

      expect(user).to include_attributes(expected_attributes)
    end
```

in this last example, i created a custom matcher `include_attributes`, this lets us check if the object `actual` responds to the given attributes and that their values equal the given one.

this custom matcher uses `respond_to?` method to check if the object has the required method, and then we use send to invoke it, and compares the value, providing a different flexibility and making the test code a little bit more verbose.

when choosing which of these to use, i tend to go for simplicity. the first approach with `include` is perfect for straightforward hash checking, and its my go to when working with apis responses. the `include` with matchers provides more complex checking on values, and i tend to use that when the test requires a little bit more of validation. the custom matcher is useful when dealing with object that dont behave like hashes, but if you can avoid that, it is best to avoid it because of the verbosity that it adds.

if you want to deep dive into rspec, i'd recommend “the rspec book” by david chelimsky and david astels. it's been a staple for years and will cover this and much more. also check the official rspec documentation that is easy to understand and very complete.

hope this helps you.
