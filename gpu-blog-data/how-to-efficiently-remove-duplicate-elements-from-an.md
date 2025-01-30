---
title: "How to efficiently remove duplicate elements from an array in Ruby using `uniq` and `each`?"
date: "2025-01-30"
id: "how-to-efficiently-remove-duplicate-elements-from-an"
---
Ruby's `Array#uniq` method offers a succinct solution for removing duplicate elements, however its effectiveness and suitability hinge on the specific use case. While `uniq` appears straightforward, understanding its behavior in conjunction with iteration methods like `each` is crucial for efficient processing, especially when dealing with complex objects within arrays. Over the years, I’ve optimized data processing pipelines involving user records, product catalogs, and network event logs, routinely encountering the need for de-duplication, making the practical implications of choosing between `uniq` and other methods abundantly clear. I'll focus on clarifying the appropriate application of `uniq` and highlight scenarios where alternative approaches may be preferred.

The primary function of `Array#uniq` is to return a new array, containing only the unique elements from the original. By default, it uses the `==` operator for equality checks. This implies that for simple data types such as integers, strings, and symbols, duplicate removal is straightforward. However, when the array contains objects, the `==` method's behavior can vary significantly based on class implementation, potentially leading to unexpected results if not correctly considered. Additionally, the performance of `uniq` might be a concern when dealing with extremely large arrays, especially compared to techniques that utilize a hash for faster lookups. Further, note that `uniq` does *not* modify the original array in place. If in-place modification is needed, `uniq!` must be used. I will focus on understanding the mechanics of `uniq` and how it differs from explicit iteration with `each`.

Let's consider a basic example of deduplicating an array of integers.

```ruby
numbers = [1, 2, 2, 3, 4, 4, 5]
unique_numbers = numbers.uniq
puts "Original array: #{numbers}"
puts "Array with duplicates removed: #{unique_numbers}"
```

In this case, `uniq` efficiently identifies and removes the duplicate values `2` and `4`, returning a new array `[1, 2, 3, 4, 5]`. The original array remains unchanged. This example highlights the standard usage. The key takeaway here is the creation of a *new* array. If the objective were to modify the original array, `uniq!` should be used. However, in my experience, retaining the original dataset often proves beneficial for auditing or debugging purposes, so a new array, as `uniq` produces, often is a safer approach.

Now, consider an example where the array contains objects. Let's assume we have a simple `User` class.

```ruby
class User
    attr_reader :id, :name

    def initialize(id, name)
        @id = id
        @name = name
    end

    def ==(other)
        return false unless other.is_a?(User)
        @id == other.id
    end

    def eql?(other)
      self == other
    end

    def hash
      @id.hash
    end
end

users = [
    User.new(1, "Alice"),
    User.new(2, "Bob"),
    User.new(1, "Alice"),
    User.new(3, "Charlie"),
    User.new(2, "Bob")
]

unique_users = users.uniq
puts "Original users: #{users.map{|u| u.id}}"
puts "Unique users by ID: #{unique_users.map{|u| u.id}}"
```

In this scenario, the `User` class has an overridden `==` method. We are defining `==` to compare users based solely on their `id` attribute, thereby treating users with the same ID as duplicates, irrespective of their names. Furthermore, we define `eql?` and `hash` to ensure that objects with equal ids also return the same hash value, since they are now being considered equal and the hash is used by `uniq`. Because of this modified `==` method and the accompanying `eql?` and `hash`, `uniq` correctly removes the duplicate users based on ID, resulting in `unique_users` containing only users with unique IDs. This is often critical in real world scenarios, where comparisons between objects often depend on specific attributes rather than the default object identity check. If the `==` method were not overridden, the `uniq` method would simply return the original array because each `User` object would be considered unique, despite having identical `id` values, demonstrating the importance of custom equality definitions when utilizing `uniq` on custom objects.

Finally, let's examine a scenario where we might use `each` alongside a more manual approach, which could sometimes provide better control, particularly in scenarios requiring custom logic beyond basic equality. This approach would generally be used when processing large arrays where hash table lookups are significantly faster than `uniq`'s more direct comparisons.

```ruby
users = [
    User.new(1, "Alice"),
    User.new(2, "Bob"),
    User.new(1, "Alice"),
    User.new(3, "Charlie"),
    User.new(2, "Bob")
]

unique_users = {}
users.each do |user|
  unique_users[user.id] = user
end
puts "Unique users using hash and each: #{unique_users.values.map{|u| u.id}}"
```

In this example, we avoid `uniq` entirely and instead employ `each` to iterate through the array of users. We use a hash, `unique_users`, where the keys are the user IDs. In the `each` block, we assign the current `User` object to the hash using the ID as the key. If a user with the same ID is encountered again, the older entry in the hash is replaced (it is idempotent). Finally, `.values` extracts the unique `User` objects. This approach is effective and may be slightly more performant, especially when the number of unique elements is low compared to the overall array size, because of the constant-time lookup capability provided by the hash table. In situations where more intricate handling of duplicates is needed, such as selecting the first or last occurrence, this method is often preferred over using `uniq`.  It’s more verbose, but the gains can be considerable when efficiency is paramount. It also allows for more customization of the deduplication criteria.

In conclusion, while `uniq` is a concise and often perfectly adequate method for deduplication, it's not universally ideal. The behavior depends on the nature of the elements in the array and the implementation of the `==` method when objects are involved. In situations where custom logic or performance considerations become paramount, alternatives using `each` with a hash can provide significant benefits. Choosing between them involves considering factors like the size of the array, the nature of elements within the array, whether a new array is needed versus in-place mutation, and whether custom deduplication rules are necessary. In my own practice, `uniq` is a go-to for simple cases with basic data types, but I frequently employ techniques based on hash tables for larger datasets with custom objects, or when the deduplication needs to adhere to some specific rule besides equality.

For further study on array manipulation and efficiency in Ruby, I would recommend researching the Ruby documentation on `Array` methods, especially `uniq`, `uniq!`, and methods pertaining to iteration like `each`, `map`, and `reduce`. In-depth books on algorithms and data structures would also offer valuable insights into the performance implications of various deduplication strategies, alongside explanations of concepts like hash tables, comparison operators, and object equality. Online resources detailing Ruby performance optimization can also offer hands-on techniques for improving code efficiency. Lastly, reviewing other developers' approaches on platforms like Stack Overflow can illustrate the diverse practical problems encountered, which provides a wider understanding of real world implementation challenges.
