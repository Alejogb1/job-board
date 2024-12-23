---
title: "How to make inline keyboards for Telegram bots with values in Ruby?"
date: "2024-12-23"
id: "how-to-make-inline-keyboards-for-telegram-bots-with-values-in-ruby"
---

Alright,  It's a situation I've found myself in more than once, crafting interactive Telegram bots with Ruby, and those inline keyboards are indeed a critical component for a smoother user experience. The core challenge, as you've probably already encountered, isn't just *displaying* buttons; it's connecting those buttons to actionable data—effectively, transmitting values that your bot can then process.

From experience, I know the standard `reply_markup` approach, while suitable for some use cases, often falls short when you need more complex interactions, where each button press triggers a distinct, data-rich operation. This is where inline keyboards with callback data shine. Let’s break down how to do this effectively in Ruby.

The principle is straightforward, albeit with some details to manage. Instead of sending a simple message, we craft a message with an `inline_keyboard` nested within the `reply_markup` field. This keyboard is comprised of rows, and each row contains one or more buttons. Critically, these buttons don’t just have text; they possess a `callback_data` field. This `callback_data` is the value that’s sent back to your bot when a user presses a particular button. This is vital; it’s how your bot knows *which* button was pressed and, consequently, what action should be taken.

Let’s dive into some practical Ruby code examples using the `telegram-bot-ruby` gem. I’ve found it to be reasonably straightforward for creating these interactions. I'll assume you already have the gem installed, set up your Telegram API token, and understand the basic structure of a Ruby bot with this gem.

**Example 1: Simple Button with a Static Value**

This is the foundational case. We’ll create a simple message with an inline keyboard that has a single button. Pressing this button will send back a predefined string.

```ruby
require 'telegram/bot'

token = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token

Telegram::Bot::Client.run(token) do |bot|
  bot.listen do |message|
    case message
    when Telegram::Bot::Types::Message
      if message.text == '/start'
        kb = [
          [Telegram::Bot::Types::InlineKeyboardButton.new(text: 'Press Me!', callback_data: 'button_pressed')]
        ]

        markup = Telegram::Bot::Types::InlineKeyboardMarkup.new(inline_keyboard: kb)

        bot.api.send_message(chat_id: message.chat.id, text: 'Here is a button:', reply_markup: markup)
      end
    when Telegram::Bot::Types::CallbackQuery
      if message.data == 'button_pressed'
         bot.api.send_message(chat_id: message.from.id, text: 'You pressed the button!')
      end
    end
  end
end
```

In this example, when the user sends `/start`, the bot sends a message with an inline keyboard containing a button labeled 'Press Me!'. The crucial aspect is the `callback_data: 'button_pressed'`. When the user taps this button, the `callback_query` event triggers, and the bot receives the `button_pressed` string in the `message.data` field. It then sends a confirmation message. This basic structure handles all the button press handling within the loop and makes it easy to scale.

**Example 2: Buttons with Dynamic Values (Using an Array)**

Let's say you need to display a list of items and have the user select one. We can easily construct the keyboard based on an array of values.

```ruby
require 'telegram/bot'

token = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token

options = ['Option A', 'Option B', 'Option C']

Telegram::Bot::Client.run(token) do |bot|
  bot.listen do |message|
    case message
    when Telegram::Bot::Types::Message
      if message.text == '/options'
        kb = options.map { |option|
            [Telegram::Bot::Types::InlineKeyboardButton.new(text: option, callback_data: "option_selected:#{option}")]
        }

        markup = Telegram::Bot::Types::InlineKeyboardMarkup.new(inline_keyboard: kb)

        bot.api.send_message(chat_id: message.chat.id, text: 'Choose an option:', reply_markup: markup)
      end
    when Telegram::Bot::Types::CallbackQuery
      if message.data.start_with?('option_selected:')
          selected_option = message.data.split(':').last
          bot.api.send_message(chat_id: message.from.id, text: "You selected: #{selected_option}")
      end
    end
  end
end
```

Here, we iterate over the `options` array to create an array of `InlineKeyboardButton` objects. The key is the `callback_data` which uses string interpolation to embed the selected option. On callback, we use a simple string parsing method using split to extract the selected option based on the `option_selected:` prefix of the data. This allows you to respond appropriately, perhaps showing more details about what they selected.

**Example 3: Buttons With Structured Data (JSON)**

When you need more complex data behind the button clicks, it's often better to encode it as a JSON string.

```ruby
require 'telegram/bot'
require 'json'

token = 'YOUR_TELEGRAM_BOT_TOKEN' # Replace with your actual token

items = [
  { id: 1, name: 'Item One', details: { price: 10, quantity: 5 } },
  { id: 2, name: 'Item Two', details: { price: 20, quantity: 2 } }
]

Telegram::Bot::Client.run(token) do |bot|
    bot.listen do |message|
        case message
        when Telegram::Bot::Types::Message
            if message.text == '/items'
                kb = items.map { |item|
                  [Telegram::Bot::Types::InlineKeyboardButton.new(text: item[:name], callback_data: JSON.generate( {action: 'view_item', item_id: item[:id]} ))]
                }

                markup = Telegram::Bot::Types::InlineKeyboardMarkup.new(inline_keyboard: kb)

                bot.api.send_message(chat_id: message.chat.id, text: 'Select an item:', reply_markup: markup)
            end
        when Telegram::Bot::Types::CallbackQuery
            begin
                data = JSON.parse(message.data)
                if data['action'] == 'view_item'
                    item_id = data['item_id']
                    selected_item = items.find { |item| item[:id] == item_id }
                    if selected_item
                        bot.api.send_message(chat_id: message.from.id, text: "Item: #{selected_item[:name]}, price: #{selected_item[:details][:price]}, quantity: #{selected_item[:details][:quantity]}")
                    else
                        bot.api.send_message(chat_id: message.from.id, text: "Item not found.")
                    end
                end
            rescue JSON::ParserError => e
                bot.api.send_message(chat_id: message.from.id, text: "Error processing callback data.")
                puts "JSON Parsing error: #{e.message}" #log the error in a way that doesn't expose data to the user
            end
        end
    end
end
```

In this more elaborate example, we are encoding structured item data, including the item id, which is then used to fetch further details in the callback function. Note the use of `JSON.generate` to serialize the hash to a JSON string and `JSON.parse` to deserialize it when a callback is triggered. This is particularly effective for passing more structured data, such as IDs and actions, to help guide bot behavior. Remember, these JSON strings have to fit within the callback character limit defined by Telegram. This is crucial as going over the limit will simply cause the button to not respond. I recommend keeping all data minimal to avoid issues with this limitation. I’ve also added some basic error handling when parsing the JSON, which is a good practice to keep in mind.

For further study, I’d recommend looking at the official Telegram Bot API documentation for a deep dive on all available options and limitations related to inline keyboards. Also, explore the `telegram-bot-ruby` gem’s specific documentation for details on how they handle these types. If you want a broader and deeper understanding of callback patterns, I would recommend studying asynchronous programming using any book on the topic, as it’s fundamental to working with such APIs and understanding message handling. These are all important in order to craft robust and user-friendly Telegram bots.

Ultimately, these examples are building blocks. You can combine and adapt these concepts to construct very sophisticated interactions. Just be sure to prioritize error handling and concise data payload management to keep everything working seamlessly. I've personally found that structured approach to crafting these interactions, especially when data starts becoming complex, makes your bot much more maintainable and less prone to bugs.
