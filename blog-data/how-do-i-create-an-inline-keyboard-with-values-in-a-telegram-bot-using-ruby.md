---
title: "How do I create an inline keyboard with values in a Telegram bot using Ruby?"
date: "2024-12-23"
id: "how-do-i-create-an-inline-keyboard-with-values-in-a-telegram-bot-using-ruby"
---

, let’s dive right into this. I recall a project back in 2018, dealing with a rather complex logistics bot, where we needed exactly this functionality: inline keyboards that dynamically presented different values to the user. The implementation in Ruby for Telegram bots, while seemingly straightforward, does require a solid grasp of the Telegram Bot API structure and how to serialize data correctly for transmission. Let's break down how to achieve this, focusing on a technical yet easily understood explanation.

The core concept revolves around constructing a `reply_markup` object within the message payload you send to the Telegram API. This `reply_markup` object, specifically when using inline keyboards, holds a nested structure of `inline_keyboard` arrays, which contain individual `InlineKeyboardButton` objects. The structure isn't particularly complex but requires careful crafting to avoid errors.

Let’s start with the basic skeleton. We’ll assume you have your bot's token set up and are able to send basic messages. If not, I highly recommend checking the official Telegram Bot API documentation, specifically the "Making requests" section. That’s your absolute first port of call for any Telegram bot development. Once we can send a simple text message, we can move to this more complex example.

The primary element we are working with is an `InlineKeyboardButton`. Each button is an object with at least two parameters: `text` and either `callback_data`, `url`, or `switch_inline_query`. Since we want to perform actions when a user taps a button, we'll be using `callback_data` the most. The `callback_data` is a string that is sent back to your bot when the user interacts with the button. It is important to understand this string has a maximum length limit and is often the place where you would encode state relevant to the user's choice.

Here's a simple code snippet that demonstrates this, creating a single row inline keyboard with three buttons:

```ruby
require 'telegram/bot'
require 'json' # ensure JSON support

token = 'YOUR_BOT_TOKEN' #replace with actual token

Telegram::Bot::Client.run(token) do |bot|
  bot.listen do |message|
    case message
    when Telegram::Bot::Types::Message
        if message.text == '/start'
            markup = {
              inline_keyboard: [
                  [
                  { text: 'Option A', callback_data: 'option_a' },
                  { text: 'Option B', callback_data: 'option_b' },
                  { text: 'Option C', callback_data: 'option_c' }
                ]
              ]
          }
          bot.api.send_message(chat_id: message.chat.id, text: 'Choose an option:', reply_markup: JSON.generate(markup))

      end
    when Telegram::Bot::Types::CallbackQuery
      case message.data
      when 'option_a'
        bot.api.send_message(chat_id: message.message.chat.id, text: 'You chose option A!')
      when 'option_b'
        bot.api.send_message(chat_id: message.message.chat.id, text: 'You chose option B!')
      when 'option_c'
        bot.api.send_message(chat_id: message.message.chat.id, text: 'You chose option C!')
      end

    end
  end
end
```

In the code above, we've used the standard `telegram-bot-ruby` gem. When the user types `/start`, the bot sends a message with an inline keyboard. Each button has a distinct `callback_data`. The bot also listens for `CallbackQuery` types which are generated when a user taps an inline keyboard button. The `message.data` will contain the callback data associated with the button, and the bot can react to that data accordingly. Always check the specific attributes available inside a `CallbackQuery` object in the Telegram API documentation to ensure you're extracting the correct identifiers.

Now, let's move to a slightly more complex scenario. Imagine you want to dynamically generate an inline keyboard based on a list of options retrieved from an external source. In my logistics bot project, we had product categories coming from a database, which were then used to generate inline keyboards. Consider this slightly modified example:

```ruby
require 'telegram/bot'
require 'json' # ensure JSON support

token = 'YOUR_BOT_TOKEN' #replace with actual token

def fetch_options
  # Imagine this fetches data from a database or other source
  # In real world cases, you should be careful with the type of your ids and how to handle them.
  [ {id: 1, name: 'Item One'}, {id: 2, name: 'Item Two'}, {id: 3, name: 'Item Three'}]
end

Telegram::Bot::Client.run(token) do |bot|
  bot.listen do |message|
    case message
      when Telegram::Bot::Types::Message
        if message.text == '/items'
            options = fetch_options()
            buttons = options.map { |option|
                [{ text: option[:name], callback_data: "item_selected_#{option[:id]}" }]
            }
            markup = {inline_keyboard: buttons}

            bot.api.send_message(chat_id: message.chat.id, text: 'Select an item:', reply_markup: JSON.generate(markup))
        end
    when Telegram::Bot::Types::CallbackQuery
        if message.data.start_with? 'item_selected_'
            item_id = message.data.split('_').last.to_i # Extract the item ID
            bot.api.send_message(chat_id: message.message.chat.id, text: "Item with id #{item_id} was selected")
        end
    end
  end
end
```

In this version, the `fetch_options` method (which would, in a real system, query your database or service) now returns an array of hashes, each representing an option. We generate our `inline_keyboard` array using ruby's map function. It's important to notice how we generate the callback data; we add a prefix to the item id in order to distinguish it from other potential callback queries. When a user taps on a button, the callback query data will be in the format `item_selected_1`, `item_selected_2` and so forth. Using this pattern allows us to easily extract the item id.

Finally, let’s look at an example that includes pagination. It is very important to consider pagination for inline keyboards when you have a substantial number of options, as the API has a limitation on the maximum number of buttons you can have in a single keyboard. We also need to track the current page in this case. We'll use a simple variable to track the current page. In a real world scenario, this should most likely be tracked on a per-user basis inside a database or other similar storage mechanism. This is a common pitfall: to not store the state of your bot in any proper form of persistence, causing unexpected behavior when multiple users interact with it concurrently.

```ruby
require 'telegram/bot'
require 'json'

token = 'YOUR_BOT_TOKEN'
ITEMS_PER_PAGE = 3
$current_page = 1 # Note the global variable, use an external storage mechanism for prod.
ITEMS =  [
        { id: 1, name: 'Item One'}, {id: 2, name: 'Item Two'}, {id: 3, name: 'Item Three'},
        {id: 4, name: 'Item Four'}, {id: 5, name: 'Item Five'}, {id: 6, name: 'Item Six'},
        {id: 7, name: 'Item Seven'}, {id: 8, name: 'Item Eight'}, {id: 9, name: 'Item Nine'}
        ]

def get_items_for_page(page, items_per_page, items)
  start_index = (page - 1) * items_per_page
  end_index = start_index + items_per_page
  items[start_index...end_index]
end

def build_pagination_keyboard(items, items_per_page, current_page)
    items_for_page = get_items_for_page(current_page, items_per_page, items)

    buttons = items_for_page.map { |item|
        [{text: item[:name], callback_data: "item_selected_#{item[:id]}"}]
    }

    pagination_row = []
    if current_page > 1
        pagination_row << {text: 'Previous', callback_data: "previous_page"}
    end
    if (current_page * items_per_page) < items.size
      pagination_row << {text: 'Next', callback_data: "next_page"}
    end
    if not pagination_row.empty?
      buttons << pagination_row
    end

    {inline_keyboard: buttons}
end

Telegram::Bot::Client.run(token) do |bot|
  bot.listen do |message|
    case message
      when Telegram::Bot::Types::Message
        if message.text == '/items'
          markup = build_pagination_keyboard(ITEMS, ITEMS_PER_PAGE, $current_page)
          bot.api.send_message(chat_id: message.chat.id, text: "Select an item (Page: #{$current_page}):", reply_markup: JSON.generate(markup))
        end
      when Telegram::Bot::Types::CallbackQuery
        case message.data
          when 'next_page'
            $current_page += 1
            markup = build_pagination_keyboard(ITEMS, ITEMS_PER_PAGE, $current_page)
            bot.api.edit_message_reply_markup(chat_id: message.message.chat.id, message_id: message.message.message_id, reply_markup: JSON.generate(markup))
          when 'previous_page'
            $current_page -= 1 if $current_page > 1
            markup = build_pagination_keyboard(ITEMS, ITEMS_PER_PAGE, $current_page)
            bot.api.edit_message_reply_markup(chat_id: message.message.chat.id, message_id: message.message.message_id, reply_markup: JSON.generate(markup))
          when /item_selected_\d+/
            item_id = message.data.split('_').last.to_i
            bot.api.send_message(chat_id: message.message.chat.id, text: "You selected item with ID #{item_id}")
        end
    end
  end
end

```

Here we introduce the concept of `edit_message_reply_markup` instead of sending a new message for pagination controls. This edits the current message, updating the keyboard rather than sending a flood of similar messages. This greatly improves the UX, as it avoids cluttering the user's chat. It's important to track the message id and chat id to use this API call. The bot also handles the ‘next’ and ‘previous’ page buttons by adjusting the `$current_page` global variable (remember to use a proper storage mechanism!) and generating a new keyboard with the updated page.

It's important to remember that the Telegram API is updated regularly, so always consult the official documentation at core.telegram.org/bots. Also, for a deeper dive into handling API requests more efficiently and understanding concepts like webhooks for more robust bot deployments, I’d recommend reading "Programming Telegram Bots" by Shahriyar Rzayev. Additionally, for a detailed overview of working with JSON in Ruby, "Programming Ruby 1.9 & 2.0" by David Thomas et al. would be beneficial. These books provide fundamental knowledge for crafting effective and stable telegram bots using ruby. Remember, building bots is an iterative process. Don't hesitate to experiment with different data structures and API calls to understand how they behave in practice.
