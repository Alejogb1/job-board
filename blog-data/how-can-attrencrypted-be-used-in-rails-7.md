---
title: "How can attr_encrypted be used in Rails 7?"
date: "2024-12-23"
id: "how-can-attrencrypted-be-used-in-rails-7"
---

Alright, let’s tackle this. I've spent a fair bit of time wrangling encrypted data in Rails, and `attr_encrypted` has been a tool in my arsenal for a long while. It's particularly handy when you need to store sensitive information in your database without resorting to full-disk encryption, though, keep in mind it's not a replacement for it. It also pairs well when compliance requirements around handling user data come into play. Now, in Rails 7, the core principles remain the same, but let’s delve into how you’d typically approach it, and we can look at some concrete examples to make it crystal clear.

Essentially, `attr_encrypted` allows you to define ActiveRecord attributes that are automatically encrypted before being stored in the database and decrypted when retrieved. It's transparent to the application layer – you work with the attributes as you normally would, without needing to worry about the encryption and decryption process directly. This makes for cleaner and more maintainable code.

The first thing you will absolutely need is the `attr_encrypted` gem. Add it to your Gemfile:

```ruby
gem 'attr_encrypted'
```

Then, run `bundle install`. This gets you up and running. Now, let's jump into a scenario. Let’s say we have a `User` model and we want to encrypt the `credit_card` attribute, and perhaps, their `social_security_number`. It's best to treat each sensitive attribute uniquely as we wouldn’t want to reuse an encryption key unnecessarily. Here's how you would modify the model:

```ruby
class User < ApplicationRecord
  attr_encrypted :credit_card, key: ENV['CREDIT_CARD_ENCRYPTION_KEY'], attribute: 'encrypted_credit_card'
  attr_encrypted :social_security_number, key: ENV['SSN_ENCRYPTION_KEY'], attribute: 'encrypted_ssn',  mode: :per_attribute_iv_and_salt

  validates :credit_card, presence: true
  validates :social_security_number, presence: true
end
```

There are a few critical aspects here. Firstly, we use `attr_encrypted` to declare which attributes to encrypt. Secondly, we specify a `key` for encryption, which for security should come from environment variables, and never hardcoded. Using a unique key per attribute adds an extra layer of security. Thirdly, I've included `attribute: 'encrypted_credit_card'` and `attribute: 'encrypted_ssn'`. By default, `attr_encrypted` stores the encrypted value in a column with the same name as the original attribute with "_encrypted" appended. However, explicitly specifying it makes your intention clear and can help avoid issues later, especially when doing schema modifications. I have also specified the `mode` for `social_security_number`. By default, `attr_encrypted` uses the `per_attribute_iv` mode. This encrypts the sensitive data using an initialization vector but reuses the IV for each encryption. Since SSN's are of a very sensitive nature, we want to leverage `per_attribute_iv_and_salt` to add a per attribute salt on each encryption and decryption operation, and also generate unique IV's for the encryption. This drastically minimizes chances of pattern exploitation.

The 'encrypted_credit_card' and 'encrypted_ssn' columns would be added to your database through a migration:

```ruby
class AddEncryptedColumnsToUsers < ActiveRecord::Migration[7.0]
  def change
    add_column :users, :encrypted_credit_card, :string
    add_column :users, :encrypted_ssn, :string
  end
end
```
and remember to run `rails db:migrate`.

Now, when you interact with the `User` model, the encryption and decryption are handled automatically. You can access or set `user.credit_card` as if it were unencrypted. `attr_encrypted` takes care of the rest.

Let's look at a slightly different use case, dealing with a `Settings` model where we might store configuration values. This time, we’ll store the encrypted value as a json value.

```ruby
class Setting < ApplicationRecord
  attr_encrypted :api_key, key: ENV['API_KEY_ENCRYPTION_KEY'], attribute: 'encrypted_settings', marshal: true

  def self.api_key
    setting = find_by(name: 'api_settings')
    setting&.api_key
  end

  def self.api_key=(value)
    setting = find_or_create_by(name: 'api_settings')
    setting.api_key = value
    setting.save
  end
end
```

```ruby
class AddEncryptedSettingsToSettings < ActiveRecord::Migration[7.0]
  def change
    add_column :settings, :encrypted_settings, :jsonb
  end
end
```
and run `rails db:migrate` again.

Here, we're also using a `marshal: true` option. This allows us to serialize the encrypted value using marshal. This can be extremely useful when you don't want the encrypted value to be a string, but potentially a complex data structure. You have to ensure your data is marshalable. The 'encrypted_settings' attribute is a `jsonb` column in this case and will store a hash that includes both the encrypted api key and the per attribute IV.

In the Setting model, we also see helper methods for retrieving the `api_key` and setting it. In my past, this structure has been useful for avoiding having different setting models for different parts of the application. It allows all sensitive configurations to be kept in one place.

It's vital to understand the implications of key management here. If you lose your encryption keys, you'll lose access to your encrypted data permanently. Consider using something like AWS KMS, HashiCorp Vault, or Azure Key Vault to manage your keys securely. There's a great book, "Cryptography Engineering" by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno, that goes deep into these concerns, so if you are looking to have a more in-depth understanding of the cryptography involved, this would be a great place to start.

One last example involves an `Order` model. Suppose we need to encrypt the shipping address but store it in a different format within the same model.

```ruby
class Order < ApplicationRecord
   attr_encrypted :shipping_address, key: ENV['ADDRESS_ENCRYPTION_KEY'],
              attribute: 'encrypted_shipping_address_data',
              marshal: true,
              encode: true,
              encode_iv: true,
              encode_salt: true,
              serializer: JSON
end
```
```ruby
class AddEncryptedAddressToOrders < ActiveRecord::Migration[7.0]
  def change
    add_column :orders, :encrypted_shipping_address_data, :text
  end
end
```
and run `rails db:migrate`.

In this case, we're also introducing the `encode` option which converts the output into base64. This option has other related options, `encode_iv` and `encode_salt`, that similarly convert the initialization vector and the salt to base64 strings, which ensures the values can be serialized properly in case of string based columns. We're also specifically saying we want to serialize the value using `JSON`. This can be helpful if you need the flexibility to store serialized data in a plain text column.

For detailed documentation, the `attr_encrypted` github repository and its Readme are good resources. Always ensure you understand the implications of how the gem handles the encryption process. Check for the latest security vulnerabilities. For a better understanding of secure coding practices in ruby, the OWASP Ruby on Rails Security Cheat Sheet is a valuable document that I have personally relied on.

In conclusion, `attr_encrypted` offers a solid way to manage the encryption of individual attributes in your Rails 7 applications. The key to using it effectively and securely lies in understanding the configuration options, managing the keys properly, and recognizing its limitations as a security measure. Hopefully, these examples give you a clearer path forward with its use.
