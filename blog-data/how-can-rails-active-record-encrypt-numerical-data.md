---
title: "How can Rails Active Record encrypt numerical data?"
date: "2024-12-23"
id: "how-can-rails-active-record-encrypt-numerical-data"
---

Alright, let's tackle encrypting numerical data in rails active record. it's a common challenge, and i’ve definitely seen my share of implementations, both good and less so, in my time. The trick here isn’t necessarily finding *a way* to do it – there are many – but finding a way that's secure, performant, and maintainable within the rails ecosystem. I've seen projects where the approach was… less than ideal, let's say, and it resulted in performance bottlenecks and security concerns down the line.

The core problem with numerical data is that it’s inherently structured, unlike plain text. Directly encrypting integers or floats with naive approaches can introduce predictable patterns, especially with small ranges of values. For instance, if you're just directly encrypting product prices, a savvy attacker might start correlating encrypted values with price ranges and potentially break your security. So, we need a more sophisticated strategy.

The most effective route generally involves a combination of cryptographic techniques with some custom logic inside your rails models. We’re essentially striving to keep the data encrypted at rest in the database, and then decrypt it only when the application needs to access the raw value. This isn’t about completely hiding the presence of numerical data - that's extremely difficult to do in a database environment - but it's about ensuring that the actual values cannot be interpreted without the correct cryptographic key.

Let’s start with the basic approach, using `ActiveSupport::MessageEncryptor`. Rails provides this out-of-the-box and it’s a fairly good starting point. However, note that without taking extra steps, using this directly on numbers can lead to those predictable pattern problems mentioned earlier.

Here's how i’ve seen it implemented with some modifications to be more robust:

```ruby
# app/models/concerns/encryptable_number.rb
module EncryptableNumber
  extend ActiveSupport::Concern

  class_methods do
    def encrypts_number(attribute, key: nil, salt: nil)
      key ||= Rails.application.credentials.encryption_key
      salt ||= 'some_default_salt_for_numbers' #consider a per-attribute salt if needed
      define_method("#{attribute}=") do |number|
        return super(nil) if number.nil? # Ensure nil values are handled properly
        encryptor = ActiveSupport::MessageEncryptor.new(key, salt: salt)
        encrypted_value = encryptor.encrypt(number.to_s) # Convert to string for encryption
        super(encrypted_value)
      end

      define_method(attribute) do
        encrypted_value = super()
        return nil if encrypted_value.nil?
        encryptor = ActiveSupport::MessageEncryptor.new(key, salt: salt)
        decrypted_value = encryptor.decrypt(encrypted_value) # decrypting
        begin
         Float(decrypted_value) # Convert back to numeric, handling possible nil or non number responses
        rescue ArgumentError, TypeError
          nil
        end
      end
    end
  end
end
```

This module is designed to be included into your active record models. It handles both encryption when setting the value and decryption when retrieving it. Importantly, it converts the number to a string before encryption, which avoids issues when dealing with active record and ensures that the encryption is performed on a text representation. The `begin/rescue` block ensures that errors during decryption don’t crash the application if, for some reason, corrupted data exists in that column. It is critical that you store the encryption key securely, ideally using rails credentials, which is referenced via `Rails.application.credentials.encryption_key`.

Here's an example of how to use it within a model:

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  include EncryptableNumber
  encrypts_number :price
end
```

Now, any time you set or retrieve the `price` attribute of a `product` instance, the value will be transparently encrypted and decrypted.

One potential weakness with this approach is the possibility of generating predictable encrypted values if the numerical range is relatively small. To address this, we can use a technique called 'format-preserving encryption' which allows us to use a standard block cipher, but keeps the encrypted output in the same format (number in this case) as the original input. I have seen various implementations for this that have been a bit fragile over time, so i tend towards methods that introduce randomisation.

Here’s a slightly more complex method adding an element of randomness, by concatenating with a randomly generated string:

```ruby
# app/models/concerns/encryptable_number.rb
module EncryptableNumber
  extend ActiveSupport::Concern

  class_methods do
      def encrypts_number_randomized(attribute, key: nil, salt: nil)
        key ||= Rails.application.credentials.encryption_key
        salt ||= 'some_default_salt_for_numbers'
        define_method("#{attribute}=") do |number|
          return super(nil) if number.nil?

           random_string = SecureRandom.hex(10)
          combined_string = "#{number.to_s}#{random_string}"

          encryptor = ActiveSupport::MessageEncryptor.new(key, salt: salt)
          encrypted_value = encryptor.encrypt(combined_string)
          super(encrypted_value)
        end

        define_method(attribute) do
          encrypted_value = super()
          return nil if encrypted_value.nil?

          encryptor = ActiveSupport::MessageEncryptor.new(key, salt: salt)
          decrypted_string = encryptor.decrypt(encrypted_value)

          return nil unless decrypted_string

          begin
           Float(decrypted_string[0...-20]) # extract the original number, taking 20 chars from the end - can be adjusted depending on the size of your random string.
          rescue ArgumentError, TypeError
            nil
          end
        end
      end
  end
end

```

```ruby
# app/models/transaction.rb
class Transaction < ApplicationRecord
  include EncryptableNumber
  encrypts_number_randomized :amount
end
```
This version adds a 20 character hex string to the number before encrypting and then slices it off after decryption. This ensures the input for encryption isn't predictable based solely on the number you are encrypting. You should adjust the length of the random string to your needs, which means you will also need to adjust the amount you slice off.

Finally, if you require more stringent control over the encryption process, it may be worth exploring `ActiveEncryption`, as introduced in Rails 7.1. This provides support for transparently encrypting entire columns using database level encryption techniques and external key management services. You will need to be careful with the migration process, and the performance and setup can be quite complex. I'd recommend looking at the official rails documentation and the `activeencryption` gem’s documentation if you plan on going this route.

For further reading i strongly suggest reading *Applied Cryptography* by Bruce Schneier. It's a thorough and comprehensive guide to the theory and practice of cryptography. For a more practical rails oriented approach, you can check out the official rails guides, especially around `ActiveSupport::MessageEncryptor` and the activeencryption gem documentation. Also, the *Cryptography Engineering* book by Niels Ferguson, Bruce Schneier and Tadayoshi Kohno is an excellent resource, especially for building your intuition around these topics.

These are the general approaches that I have used successfully. When implementing these, always test them thoroughly and be particularly careful when migrating your existing data, as incorrect key management can lead to permanent data loss. Start simple, and incrementally improve the security and complexity based on the specific needs of your application.
