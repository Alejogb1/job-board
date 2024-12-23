---
title: "How can Rails 6 I18n be used for custom user-defined naming schemas?"
date: "2024-12-23"
id: "how-can-rails-6-i18n-be-used-for-custom-user-defined-naming-schemas"
---

Alright, let's tackle this. I've spent a fair amount of time in the Rails trenches, and user-defined naming schemas with i18n always present a unique set of challenges. It’s not always as straightforward as the typical label or message translation. We’re talking about allowing users to effectively create their own ‘dictionaries’ within the application's i18n framework, which demands a more flexible approach than what's readily available out of the box.

The standard i18n implementation in Rails 6 is fantastic for translating predefined text, based on locale keys. However, when you need user-defined schemas, it’s critical to understand how to bend the framework to your will without compromising its core functionality or creating a maintenance nightmare. Imagine, for instance, an application where users can create custom forms with their labels, and those labels need to be consistently displayed in different languages as chosen by the users themselves, not just the application admin. This is the scenario I faced during a project involving a multi-tenant, highly customizable platform. It was the challenge of adapting i18n for a world where the 'translations' are constantly changing, defined by the end-users.

The first thing we have to accept is that we won't be directly mapping our user-defined names to the standard yml locale files. Instead, think of these custom names as specific, dynamically generated data points that need to be managed alongside our translation framework, rather than within it. We’re essentially creating an i18n-aware lookup service.

Here’s how I handled this with a method centered around a combination of ActiveRecord models and custom i18n backends:

**Approach 1: Storing User-Defined Names in a Database**

The core idea is to store the user-defined schema in our application’s database and build a custom lookup mechanism. This usually involves at least three tables: one for the user-defined schemas themselves, one for the fields within the schema, and finally one for the translated versions of the field names.

Let's illustrate it with a few models.

```ruby
# app/models/custom_schema.rb
class CustomSchema < ApplicationRecord
  has_many :custom_fields, dependent: :destroy
  belongs_to :user

end

# app/models/custom_field.rb
class CustomField < ApplicationRecord
  belongs_to :custom_schema
  has_many :custom_field_translations, dependent: :destroy

  def translated_name(locale)
      custom_field_translations.find_by(locale: locale)&.translation || name
  end
end

# app/models/custom_field_translation.rb
class CustomFieldTranslation < ApplicationRecord
    belongs_to :custom_field
end
```

Here, `CustomSchema` represents the entire schema created by the user, while `CustomField` defines the individual fields, which could be labels for a form or columns in a report. `CustomFieldTranslation` keeps the actual translated names in different locales. Crucially the field `name` inside `CustomField` acts as a default if no translation is provided. This is essential for situations where translations might be missing, ensuring a fallback.

The key here is the `translated_name(locale)` method in `CustomField`. This method attempts to locate a translation for the requested locale and falls back to the default field name if a translation is not available. This separation of the primary name and its translated counterpart enables both display flexibility and ease of management.

Now, we don't directly involve `I18n.t` here, we instead create helper method, like so:

```ruby
# app/helpers/custom_schema_helper.rb

module CustomSchemaHelper
  def translated_field_name(custom_field, locale = I18n.locale)
    custom_field.translated_name(locale)
  end
end
```

This is how we would use it inside our templates:

```erb
<label><%= translated_field_name(@custom_field, :fr) %></label>
```

This keeps the templating logic clean and abstracted, making it easier to manage, scale and modify later.

**Approach 2:  A Custom I18n Backend**

While the database approach works well for persistence and management, we can also leverage the i18n framework's backend flexibility to directly integrate user-defined translations. This approach involves implementing a custom `I18n::Backend` to fetch translations from our data source, typically a database table. This allows using the `I18n.t` method with user-defined keys. I've used this when user translation data was less structured than full blown schemas, often with simple key-value pairs.

Here’s how this looks in practice:

```ruby
# lib/i18n/backend/user_defined.rb

module I18n
  module Backend
    class UserDefined < Simple
      def initialize(options = {})
        super(options)
        @translations = {} # Cache for translations
        load_translations_from_database # Load initial set of user translations
      end

      def available_locales
         # We need to query the database for all existing locales used,
        # this is a simplified version for example purposes
        CustomFieldTranslation.distinct.pluck(:locale)
      end

       def store_translations(locale, data, options = {})
        load_translations_from_database # Reload whenever translations are stored
        super # Fall back to the simple backend for storing default translations
      end


      protected

       def lookup(locale, key, scope = [], options = {})
        # Use the previously loaded translations
        flattened_key = (scope + [key]).join('.')
        @translations.dig(locale.to_sym, flattened_key) || super
        end

      private

      def load_translations_from_database
        @translations = {}
        CustomFieldTranslation.all.each do |translation|
          @translations[translation.locale.to_sym] ||= {}
          @translations[translation.locale.to_sym][translation.custom_field.name] = translation.translation
        end

      end
    end
  end
end

```

This class defines a custom `I18n::Backend::UserDefined` that extends the `I18n::Backend::Simple` backend. The `load_translations_from_database` method fetches all user-defined translations from the database and stores them in a cache for quick lookup. The `lookup` method first checks the cache and falls back to the default backend if no user-defined translation is found.

To integrate this, you'll need to configure your application's i18n to use this custom backend:

```ruby
# config/initializers/i18n.rb
I18n.backend = I18n::Backend::UserDefined.new
```

With this configuration, we can now use `I18n.t` like we are used to using, however this time fetching translations from user defined values:

```erb
<label><%= I18n.t(@custom_field.name, locale: :fr) %></label>
```

**Approach 3: Hybrid Approach**

In many scenarios, it’s beneficial to blend elements from both of the above methodologies. That could mean utilizing the database to store user-defined values, yet constructing the keys for the i18n lookups in a structured, hierarchical way, mirroring standard translation file structures. This approach aims to provide the advantages of both worlds: the robust storage and management capabilities of a relational database and the organized lookup patterns of Rails’ i18n system.

For example, when storing the translations in the database, we can use a hierarchical key like this "user_schemas.#{user.id}.#{field_name}". The structure now allows the developer to keep a logical structure when developing functionality that interacts with these fields.

In practice we can change the previous approach and use the following methodology:

```ruby
# lib/i18n/backend/user_defined.rb

module I18n
  module Backend
    class UserDefined < Simple
      # (Previous code remains mostly unchanged, we adapt the load_translations_from_database method)

      private
      def load_translations_from_database
           @translations = {}
            CustomFieldTranslation.all.each do |translation|
                key = "user_schemas.#{translation.custom_field.custom_schema.user_id}.#{translation.custom_field.name}"
                @translations[translation.locale.to_sym] ||= {}
                @translations[translation.locale.to_sym][key] = translation.translation
            end
      end
    end
  end
end

```

Now instead of fetching by field name, we fetch by user and field name. This allows us to have a greater flexibility when retrieving user specific translations.

```erb
<label><%= I18n.t("user_schemas.#{@custom_field.custom_schema.user_id}.#{@custom_field.name}", locale: :fr) %></label>
```

**Closing Thoughts**

Implementing user-defined naming schemas with Rails 6 i18n requires a careful and thoughtful approach. The standard i18n framework is excellent for static text, but dynamic user data requires a level of flexibility that often needs custom extensions. The key is to understand how to extend the existing framework without re-inventing the wheel.

For further study on this topic, I would recommend exploring the official Rails internationalization guides and digging into the source code for the i18n gem. It's also beneficial to study database design best practices when handling hierarchical data, and the concept of custom i18n backends is a very beneficial pattern to understand. Finally, the "Refactoring" book by Martin Fowler is a cornerstone text in designing scalable systems and this subject benefits a lot from these principles. Understanding both your current and future needs and planning accordingly will make your journey through i18n much smoother.
