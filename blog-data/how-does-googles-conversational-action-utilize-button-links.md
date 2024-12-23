---
title: "How does Google's conversational action utilize button links?"
date: "2024-12-23"
id: "how-does-googles-conversational-action-utilize-button-links"
---

Let's tackle this one. From my experience building a conversational interface for a smart home system, I’ve become quite familiar with how Google's conversational actions handle button links, particularly the subtleties that go beyond just simple website redirects. It’s more nuanced than it might initially appear, and it's definitely worth understanding if you’re serious about creating a compelling conversational experience.

The core concept is to provide users with avenues to explore further, often outside of the immediate dialog flow, without completely disrupting their conversation. Button links, or 'suggestions' in Google’s parlance, serve as a bridge between the interactive voice/text interface and a deeper level of engagement. They are not just clickable hyperlinks; they are strategically placed action prompts that expand the user's options.

Now, let's break down the mechanics. The interaction primarily revolves around responses sent back from your fulfillment webhook to the Google Assistant. These responses are not just plain text; they’re structured JSON objects adhering to the Actions on Google schema. Within this schema, you can include `suggestions`, which are essentially the container for your button links. Each suggestion comprises a `title` and a `url`.

Think of it this way: when you're providing information, especially detailed information, a single voice response might not be enough, or the user may want to see a related website to further investigate a product or service. So rather than having the conversation go off topic, or having the user ask repeatedly for that website, we use these suggestions to give direct access.

Let's start with a simple example. Suppose a user asks about a specific recipe. Instead of just providing the recipe instructions, you can offer a button link to the complete recipe on your website, thus allowing the user to explore additional content or related recipes.

Here's how the corresponding JSON response from your fulfillment might look:

```json
{
  "fulfillmentMessages": [
    {
      "text": {
        "text": [
          ", here are the instructions for that recipe. If you'd like to see the full recipe with pictures, you can use the link below."
        ]
      }
    },
    {
      "suggestions": {
        "suggestions": [
          {
            "title": "Full Recipe Details",
            "url": "https://www.example.com/recipes/chocolatecake"
          }
        ]
      }
    }
  ]
}
```

Here, the `fulfillmentMessages` array contains both a text response and a `suggestions` object. The `suggestions` object, in turn, contains an array of suggestion objects, each with its `title` and `url`. When the Assistant renders this response on a device with a screen, it will display the text response along with a button that reads “Full Recipe Details”. Tapping that button opens the associated URL in the device’s browser, letting the user proceed further.

However, things can get a bit more complex, particularly when dealing with dynamic data. For instance, imagine a scenario where you are running an ecommerce platform. Instead of hardcoding URLs, you need to generate links based on the specific product a user has asked about. This involves incorporating logic within your fulfillment webhook.

Suppose a user asks to know about a 'smart thermostat'. Let's say your backend uses a product id in the urls, such as `https://www.example.com/products/product?id=123`. You can generate such a link like in the example below.

```javascript
const productId = "123"; // Assuming you've extracted the product ID from the user's query
const productUrl = `https://www.example.com/products/product?id=${productId}`;

const responseJson = {
    "fulfillmentMessages": [
        {
            "text": {
                "text": [", I found a smart thermostat. Here’s a link to the product details."]
            }
        },
        {
            "suggestions": {
                "suggestions": [
                    {
                        "title": "View Product Details",
                        "url": productUrl
                    }
                ]
            }
        }
    ]
};

console.log(JSON.stringify(responseJson));
```

In this JavaScript snippet (similar logic can be applied in any server-side language), the `productId` is dynamically included in the URL using template literals. The resulting `responseJson` would then be returned to Google, generating the required button. This demonstrates how button links can effectively become part of an interactive system and are not just static elements.

Moreover, the integration extends to actions beyond simple browser redirects. Button links can trigger intents within your conversational action, allowing you to deep-link into specific features. For instance, a button could trigger a custom intent that displays more information or executes a function within the Assistant interface, rather than just redirecting to a webpage. You do that by specifying the `action` key rather than the `url` key.

Consider an example where your action has an intent `check_order_status`. Here is an example how to achieve that using action.

```json
{
  "fulfillmentMessages": [
    {
      "text": {
        "text": [
          ", I've found an order that matches your request. To check the status, click the link below."
        ]
      }
    },
    {
      "suggestions": {
        "suggestions": [
          {
            "title": "Check Order Status",
             "action": {
                 "intent":"check_order_status"
               }
          }
        ]
      }
    }
  ]
}
```
In this example, instead of a url, the suggestions objects contains an action key, which defines the target intent. This allows the conversational action to navigate the conversation without the need of the user to say further sentences.

From a practical perspective, when you are building conversational interfaces, I’d advise focusing on clarity and relevance when using button links. The title of the button should accurately reflect its destination, whether it's a webpage or another part of your action. Avoid generic titles like "Click Here"; instead, use descriptive titles that give the user a clear idea of what will happen after they click. Pay attention to mobile responsiveness, especially if your button links redirect to external webpages.

For further reading and deeper understanding, I would recommend reviewing Google's official Actions on Google documentation, particularly the sections dealing with suggestions and rich responses. Additionally, exploring works like "Designing Voice User Interfaces" by Cathy Pearl and "Conversational Design" by Erika Hall offers invaluable insight into structuring conversations and choosing effective ways to use visual prompts like button links. These resources provide a more profound view of the design principles behind conversational interfaces and can significantly enhance your implementation strategy.

In conclusion, button links within Google conversational actions are versatile tools for extending the functionality and user engagement beyond the confines of a purely verbal exchange. They offer a structured way to incorporate more complex functionalities and richer experiences, provided that they are correctly implemented and strategically used. They are not just a design afterthought, but a crucial aspect of a well-built conversational action, and understanding their mechanics is essential for anyone working in this space.
