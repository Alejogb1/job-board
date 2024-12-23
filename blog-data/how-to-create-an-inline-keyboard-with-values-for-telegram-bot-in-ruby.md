---
title: "How to create an inline keyboard with values for Telegram bot in Ruby?"
date: "2024-12-16"
id: "how-to-create-an-inline-keyboard-with-values-for-telegram-bot-in-ruby"
---

,  I've definitely had my fair share of Telegram bot development headaches, and inline keyboards were definitely one of those areas where initial implementations could get… finicky. Let's break down how to properly construct them with values using Ruby, ensuring a solid and functional solution.

First off, understand that Telegram's bot api uses a json-based structure for defining these inline keyboards. It’s not just a flat list; we're constructing a nested structure of buttons, each having a specific associated callback data that our bot will receive when clicked. The key here is precision and clarity in how you construct this json. Let’s dive into the technical details.

We are essentially creating an array of arrays, where each inner array represents a row of buttons. Each button has at least two key pieces of information: the `text`, which is what the user sees, and the `callback_data`, which is the value returned to your bot when the button is pressed. I found that meticulously planning this structure ahead of time, even just on paper initially, saved a *lot* of debugging time.

Let’s start with a straightforward single-row example:

```ruby
require 'json'

def build_single_row_keyboard(button_texts_and_data)
  buttons = button_texts_and_data.map do |text, data|
     { text: text, callback_data: data }
  end

  { inline_keyboard: [buttons] }.to_json
end


keyboard_data = [
  ["Option 1", "option_1_value"],
  ["Option 2", "option_2_value"],
  ["Option 3", "option_3_value"]
]

keyboard_json = build_single_row_keyboard(keyboard_data)
puts keyboard_json
# Output example:
# {"inline_keyboard":[[{"text":"Option 1","callback_data":"option_1_value"},{"text":"Option 2","callback_data":"option_2_value"},{"text":"Option 3","callback_data":"option_3_value"}]]}

```

Here, `build_single_row_keyboard` takes an array of text/data pairs, converts each into a button hash, and wraps them up in the correct json structure. This demonstrates the base structure; a single array of buttons within an `inline_keyboard` array. This is our bare minimum starting point. It’s crucial, before I moved on, to ensure this fundamental structure was correct in my past projects. A minor misspelling in the json keys can cause silent failures where the buttons simply don't show, or worse, don’t provide the expected callback information when pressed.

Now, let's advance to a more practical, multi-row example which is very common in actual bot implementations:

```ruby
require 'json'

def build_multi_row_keyboard(keyboard_layout)
  rows = keyboard_layout.map do |row|
    row.map { |text, data| { text: text, callback_data: data } }
  end
  { inline_keyboard: rows }.to_json
end

keyboard_layout = [
  [ ["Button A", "button_a_value"], ["Button B", "button_b_value"] ],
  [ ["Button C", "button_c_value"], ["Button D", "button_d_value"] ]
]

keyboard_json = build_multi_row_keyboard(keyboard_layout)
puts keyboard_json

# Output Example:
# {"inline_keyboard":[[{"text":"Button A","callback_data":"button_a_value"},{"text":"Button B","callback_data":"button_b_value"}],[{"text":"Button C","callback_data":"button_c_value"},{"text":"Button D","callback_data":"button_d_value"}]]}
```

The `build_multi_row_keyboard` function now takes an array of arrays (or a matrix) where each inner array represents a row of buttons. This lets us structure our keyboard in a way that's more visually usable for the end-user. When building such keyboards, you need to map and iterate correctly through all dimensions of the structure so you don’t inadvertently cause exceptions or generate malformed json. The key here is understanding the nested structure that Telegram requires – essentially, a two-dimensional array representing rows and the buttons within them, each button holding the `text` and `callback_data`.

One important lesson I learned the hard way was ensuring the callback data is small and relevant. Telegram has limits on the size of this data field. If you need to pass larger information, a better pattern involves using the `callback_data` to identify the action/option and use your bot’s local state or database to hold the actual information required for processing. This also decouples your button representation from any specific data, leading to cleaner code over time and easier maintenance. It makes it more scalable as well.

Let’s move to a real-world example to solidify the best practice: imagine we want to build a confirmation dialogue with yes/no buttons with associated data. Also, let's suppose that a user can select from a list of categories to filter a list of posts displayed by a bot and those posts IDs are too long to be included in callback data.

```ruby
require 'json'

def build_confirmation_keyboard
  keyboard = [
    [ { text: "Yes", callback_data: "confirm_yes" },
      { text: "No",  callback_data: "confirm_no"  } ]
    ]
  { inline_keyboard: keyboard }.to_json
end


def build_category_keyboard(categories)
  keyboard_rows = categories.map do |category|
      [{text: category[:name], callback_data: "category_select:#{category[:id]}"}]
  end
  {inline_keyboard: keyboard_rows}.to_json
end

confirmation_keyboard_json = build_confirmation_keyboard()
puts "Confirmation Keyboard:"
puts confirmation_keyboard_json

categories = [{name: "Tech", id: 1}, {name: "Cooking", id:2}, {name: "Travel", id: 3}]
category_keyboard_json = build_category_keyboard(categories)
puts "\nCategory Keyboard:"
puts category_keyboard_json


#Output Example:
#Confirmation Keyboard:
#{"inline_keyboard":[[{"text":"Yes","callback_data":"confirm_yes"},{"text":"No","callback_data":"confirm_no"}]]}
#
#Category Keyboard:
#{"inline_keyboard":[[{"text":"Tech","callback_data":"category_select:1"}],[{"text":"Cooking","callback_data":"category_select:2"}],[{"text":"Travel","callback_data":"category_select:3"}]]}
```

In the `build_confirmation_keyboard` example, the callback data directly indicates the user's choice, and you would handle the `confirm_yes` or `confirm_no` cases in your bot’s callback processing logic, for instance updating your database or doing some sort of action. The `build_category_keyboard` builds a list of categories as a menu, using the `category_select` prefix to help your bot's logic decipher what action the user took. Instead of including full article id lists in the callback data, which may quickly exceed the length limitations, the category id is sent and your bot can look up the relevant posts for the particular category based on that id. This is a far more robust and scalable solution, and one that I consistently advocate for when working with telegram bots.

For further reading, I’d recommend delving into the Telegram Bot API documentation itself. It's available online and is the definitive resource. In addition, ‘Programming Telegram Bots’ by Syed Muhammad Bilal is also a quite useful resource if you’d like a more focused approach with practical examples. Lastly, familiarize yourself with the json format documentation to understand how it works in detail. I found that having solid command of the foundation, the structures in the telegram bot API and general json, helped me drastically reduce debugging time when building complex Telegram bots.

I hope these explanations and examples provide a solid basis for working with inline keyboards in Ruby. It's definitely an area where a little bit of careful structuring pays dividends. Focus on keeping your callback data minimal, well-structured, and use your bot’s state or database to manage larger data elements associated with the user interaction. That’s the key takeaway.
