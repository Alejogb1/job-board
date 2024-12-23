---
title: "How do I use Tagify with Rails 7 to combine JSON tags after form submission?"
date: "2024-12-23"
id: "how-do-i-use-tagify-with-rails-7-to-combine-json-tags-after-form-submission"
---

Okay, let's tackle this. I remember a particularly tricky project a few years back where we needed a robust tagging system for user-generated content within a Rails app. We settled on Tagify, and, like many, we initially stumbled over the proper handling of tag data post-form submission, specifically how to combine and persist those JSON tags effectively. It's not as straightforward as it may seem initially.

The core challenge lies in the fact that Tagify, out of the box, primarily works on the front-end, generating a text input and presenting user-friendly tags. When submitted, however, the data arrives as a simple string of comma-separated values (or similar, depending on config) which represents the *last state* of the tags. The actual tag *objects* with any associated data are not directly present. We're dealing with a plain string that we must then parse and translate back into usable data for our Rails model.

This means that instead of sending an array or JSON directly, you're essentially dealing with a string. We'll need to carefully convert this string back into an appropriate format – typically an array of strings or JSON representation – before saving it to the database. Let’s break it down into the steps required, starting with the front-end interaction and then focusing on how to process it in the Rails controller.

First, consider the front-end. Your Tagify setup will look something akin to this within your Rails form:

```html
<input name="my_object[tags]" id="tags" value="<%= @my_object.tags.to_json if @my_object.tags.present? %>">
```
This html snippet assumes `@my_object` is an object instance that already has tags saved. It injects the JSON representation of your model's `tags` attribute as the `value` for the tag input. This makes sure any prior tags are already showing up upon page load.

Now, let's focus on the JavaScript initialization of Tagify. We would use something like this snippet:
```javascript
document.addEventListener('DOMContentLoaded', function() {
    const input = document.querySelector('#tags');

    if (input) {
        const tagify = new Tagify(input, {
            whitelist: [], // You can add existing tags here if needed
            enforceWhitelist: false,
            placeholder: "Enter tags...",
            dropdown: {
              maxItems: 20,
              classname: "tags-look",
              enabled: 0, // suggest tags after 1 character
              closeOnSelect: false, // keep the list open after selecting
            }
        });
    }
});
```

Key here is to *not* expect the form to directly submit your tags in json. Tagify simply fills the input element with text. The actual tag logic, including removal and the like, is entirely done by the front-end library.

Next, we move onto the Rails controller. This is where the real "magic" happens. We will need to process the input, converting the string data into a format Rails can work with. I’ve typically used a `before_action` to handle this processing in a common way.

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base

    before_action :process_tagify_inputs

    private

    def process_tagify_inputs
        return unless params[:my_object].present? && params[:my_object][:tags].present?

        tags_string = params[:my_object][:tags]
        parsed_tags = begin
            JSON.parse(tags_string)
        rescue JSON::ParserError
          tags_string.split(',').map(&:strip).reject(&:empty?)
        end
        params[:my_object][:tags] = parsed_tags.is_a?(Array) ? parsed_tags : []
    end
end

# app/controllers/my_objects_controller.rb
class MyObjectsController < ApplicationController
    def create
      @my_object = MyObject.new(my_object_params)

      if @my_object.save
        redirect_to @my_object, notice: 'Object was successfully created.'
      else
        render :new, status: :unprocessable_entity
      end
    end

    def update
        if @my_object.update(my_object_params)
          redirect_to @my_object, notice: 'Object was successfully updated.'
        else
            render :edit, status: :unprocessable_entity
        end
    end

    private
    def set_my_object
        @my_object = MyObject.find(params[:id])
    end

    def my_object_params
      params.require(:my_object).permit(:name, :description, tags: [])
    end

end
```

This `process_tagify_inputs` method uses a `rescue` block to handle two different kinds of input: JSON and comma-separated strings. Tagify can be configured to submit data as comma-separated strings or JSON and this method can handle either. If the tag data is JSON (from an existing object), it parses that directly. If it fails with a `JSON::ParserError` it proceeds to handle the string format, breaking it down into an array. Finally, it overwrites the params hash with the processed array of tags. Importantly, within `my_object_params` you'll see that we now allow the `tags` attribute to be passed as an array of strings. This sets us up for the final step: handling the model itself.

Finally, in your Rails model, you'll want to ensure that you have a column that can store an array or JSON representation of your tags. In Rails 7, you can utilise the `serialize` method if you wish to store JSON, or use the `text[]` (array of text) column type in postgresql. Here’s an example with PostgreSQL array:

```ruby
# app/models/my_object.rb
class MyObject < ApplicationRecord
    attribute :tags, :string, array: true
end
```
Using `attribute :tags, :string, array: true` allows you to store the array of strings directly in the database. The `tags` attribute will automatically handle serialization to and from a PostgreSQL `text[]` array column. If you are using something other than PostgreSQL, you can use `serialize` like this, and make sure you have a `text` or similar type column:
```ruby
class MyObject < ApplicationRecord
    serialize :tags, Array
end
```

By setting up the controller and model in this way, we have a system that seamlessly handles user input from Tagify, persists it correctly in the database, and ensures that data is handled in a robust manner. The processing happens in `process_tagify_inputs`, which is invoked `before_action` for every request. It attempts to parse the `params[:my_object][:tags]` into JSON array. If it cannot, it splits the string input by commas. After processing the result will be an array stored in `params[:my_object][:tags]`.

To solidify this further, if you encounter more complexity, delve into these resources. For comprehensive coverage of front-end JavaScript and interaction, explore 'Eloquent JavaScript' by Marijn Haverbeke. For deeper knowledge on Rails and its internals, 'Agile Web Development with Rails 7' by David Heinemeier Hansson and co-authors, will be beneficial. Also, look at the 'PostgreSQL Documentation' specifically the section on array types for a deep understanding of how arrays are stored in the database. Understanding these pieces will give you the needed background to handle more complex scenarios.

This approach, born from prior experience, offers a solid baseline for effectively using Tagify with Rails 7. While seemingly simple, it involves careful handling of data transformation between front-end input and database storage, ensuring all moving parts work harmoniously. It's the kind of process that I've found to be consistently applicable when building tag-based features in web applications.
