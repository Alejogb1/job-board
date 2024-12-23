---
title: "How can I translate Stimulus JS values using internationalization?"
date: "2024-12-23"
id: "how-can-i-translate-stimulus-js-values-using-internationalization"
---

Let's tackle this—translating Stimulus values with i18n is a nuanced task I've encountered a few times in previous projects, and it’s more involved than just slapping some labels on static elements. It's about managing dynamic content based on the user's locale, which often shifts depending on application context. We’re not dealing with simple text replacements here; rather, we're dynamically pulling localized strings into our Stimulus controllers, ensuring the application remains globally accessible. It requires a thoughtful approach to structuring both the i18n setup and how we integrate it into our javascript codebase.

My first encounter with this was on a multi-language dashboard application. We used a Ruby on Rails backend coupled with a Javascript front end using Stimulus. Initially, the temptation was to directly embed locale-specific strings in our HTML, rendering them server-side. This works, but it completely breaks down when dealing with dynamically updated elements or content generated client-side. Furthermore, it’s a nightmare to maintain. Instead, we moved towards a more decoupled architecture, where the translation of the labels and values was handled directly within the Stimulus controllers, fetching the translated text as needed.

The core issue lies in the separation of concerns: Stimulus handles the interactivity and state, and your chosen i18n library (or approach) handles the translation logic. You need a bridge to connect the two without tightly coupling them, ensuring both remain modular. I’ve found that the most flexible way is to utilize a data attribute on the html element to specify a translation key, rather than the actual translated string itself. Then, inside the Stimulus controller, you retrieve this key and feed it to your i18n library or handler function to obtain the correct localized string.

Here's a breakdown of how I generally tackle this, along with some example code to clarify things.

First, let’s assume you’re using something like i18next for your i18n needs, though the same principles apply to other i18n libraries or custom implementations. The library offers functionalities to load translations, switch locales, and perform translations using the keys.

Here’s the first code example showing how a typical Stimulus controller might be set up to handle simple text translations using `i18next`:

```javascript
// text_controller.js
import { Controller } from "@hotwired/stimulus";
import i18next from 'i18next';

export default class extends Controller {
  static targets = ["element"];

  connect() {
    this.translate();
  }

  translate() {
      this.elementTargets.forEach((el) => {
          const key = el.dataset.i18nKey;
          if (key) {
              el.textContent = i18next.t(key);
          }
      });

  }

  changeLanguage(event) {
      const newLocale = event.target.value;
      i18next.changeLanguage(newLocale).then(() => {
          this.translate();
      });
  }
}

```

And the corresponding HTML might look like this:
```html
<div data-controller="text">
    <h2 data-text-target="element" data-i18n-key="greeting">Hello</h2>
    <p data-text-target="element" data-i18n-key="description">This is a simple example</p>
    <select data-action="text#changeLanguage">
        <option value="en">English</option>
        <option value="fr">French</option>
        <option value="es">Spanish</option>
    </select>
</div>
```
In this example, `data-i18n-key` specifies which translation to apply to an element. In your `i18next` initialization, you would provide translations that correspond to ‘greeting’ and ‘description’ for all your supported locales.

Now, let’s consider a more complex case: translating attributes of an element. For example, you might need to translate the `placeholder` or `aria-label` attributes. Here's the second code snippet:

```javascript
// attribute_controller.js
import { Controller } from "@hotwired/stimulus";
import i18next from 'i18next';

export default class extends Controller {
  static targets = ["element"];

    connect(){
        this.translateAttributes();
    }

  translateAttributes() {
    this.elementTargets.forEach((el) => {
      for (const attr in el.dataset) {
        if (attr.startsWith('i18nAttr')) {
          const attributeName = attr.substring(8).toLowerCase();
          const key = el.dataset[attr];
          if (key) {
            el.setAttribute(attributeName, i18next.t(key));
          }
        }
      }
    });
  }


  changeLanguage(event) {
        const newLocale = event.target.value;
        i18next.changeLanguage(newLocale).then(() => {
            this.translateAttributes();
        });
    }
}
```
And a matching HTML example for this:
```html
<div data-controller="attribute">
   <input type="text" data-attribute-target="element" data-i18n-attr-placeholder="input.placeholder" data-i18n-attr-ariaLabel="input.label" />
    <select data-action="attribute#changeLanguage">
        <option value="en">English</option>
        <option value="fr">French</option>
        <option value="es">Spanish</option>
    </select>
</div>
```
Here, we're iterating over the element's `dataset`, checking for keys that start with `i18nAttr`.  We then extract the attribute name (like `placeholder` or `ariaLabel`) and the i18n key and set the attribute using translated string provided by i18next. It's more generic than hardcoding specific attributes.

Finally, consider a case where you’re dealing with data values themselves – for example, formatting numbers or dates based on the current locale. We can achieve this by passing extra parameters with the key:

```javascript
// format_controller.js
import { Controller } from "@hotwired/stimulus";
import i18next from 'i18next';

export default class extends Controller {
    static targets = ["amount", "date"];

    connect(){
        this.formatData();
    }


    formatData(){
        this.amountTargets.forEach( el => {
            const amount = parseFloat(el.dataset.value);
           el.textContent = i18next.format(amount, { style: "currency", currency: "USD", locale : i18next.language } )
        });


        this.dateTargets.forEach( el =>{
            const dateString = el.dataset.value;
            const date = new Date(dateString);
            el.textContent = i18next.format(date, {dateStyle:"medium", timeStyle:"short",  locale: i18next.language})
        });
    }


    changeLanguage(event) {
        const newLocale = event.target.value;
        i18next.changeLanguage(newLocale).then(() => {
            this.formatData();
        });
    }
}
```
HTML code to go with the above:
```html
<div data-controller="format">
    <span data-format-target="amount" data-value="1234.56">1234.56</span>
    <span data-format-target="date" data-value="2024-07-28T14:00:00Z">2024-07-28T14:00:00Z</span>
    <select data-action="format#changeLanguage">
        <option value="en">English</option>
        <option value="fr">French</option>
         <option value="es">Spanish</option>
    </select>
</div>
```

Here, we fetch the numerical amount or date directly from data-attributes, then utilize the i18next format function to render localized numbers and dates. i18next provides advanced features beyond simple text translation, and these examples highlight how one can exploit those using Stimulus.

For further learning, I'd recommend delving into the official `i18next` documentation for a deep understanding of advanced topics like plurals, context-based translations, and namespaces, all of which can be used with Stimulus. Additionally, “JavaScript Patterns” by Stoyan Stefanov, can help with better organizing this type of code. Also, the ECMAScript Internationalization API documentation is useful if you ever want to use the browser's native i18n capabilities. Finally, studying the `stimulus-rails` gem’s implementation can be informative if you're using Rails.

This approach ensures that your application is not only localized but is also flexible and maintainable. Avoiding hardcoded text strings in your HTML leads to a cleaner and scalable codebase, where changing translations can happen in one centralized place instead of spread across your templates. This pattern allows your front-end and i18n mechanism to evolve independently, which is vital for long-term maintainability and global accessibility.
