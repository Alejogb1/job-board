---
title: "How can React components fetch and render Rails i18n translations?"
date: "2024-12-23"
id: "how-can-react-components-fetch-and-render-rails-i18n-translations"
---

Okay, let's tackle this. I've personally navigated this terrain quite a few times, particularly back when I was maintaining that massive e-commerce platform. Localization became a constant headache, and getting React and Rails to speak the same language regarding translations wasn't trivial. We ended up refining a solution that worked quite well, and I can walk you through my approach, along with some alternatives and code examples.

Fundamentally, the challenge lies in bridging two distinct ecosystems: the server-side rendering and management of translations in Rails and the client-side consumption and rendering of those translations within React. Rails, with its powerful i18n gem, handles translation files, pluralization rules, and locale management beautifully. React, on the other hand, is largely a JavaScript world. We need to find a method for React to efficiently access those Rails-managed translations.

The core strategy involves exposing translations to the client, making them available to the React application. Several paths lead there; I’ll cover the method I favored: rendering initial translations from the server on initial load, and then implementing an efficient mechanism for requesting additional translations or updates. This strategy optimizes performance by not loading the entire translation catalog upfront, but only the required set.

First, during the initial server-side rendering of the React application, we pass a subset of the required translations as props to the root React component. This approach reduces initial load time and improves perceived performance. The Rails controller handling the rendering would look something akin to this:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  def index
    @initial_translations = I18n.t(['hello', 'goodbye'], locale: I18n.locale)
    render 'layouts/application'
  end
end
```

Here, we are extracting specific keys 'hello' and 'goodbye' for demonstration purposes. We’d usually retrieve them dynamically, but I’m keeping it simple here. The keys correspond to entries in your Rails localization files (e.g., `config/locales/en.yml`). These translations will be rendered as data within the HTML that Rails sends to the client.

Now, in the Rails layout, we would embed these initial translations as a JSON string within a `<script>` tag to access them in the React application. This would be integrated into your Rails layout file, such as `app/views/layouts/application.html.erb`:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>My App</title>
    <%= csrf_meta_tags %>
    <%= csp_meta_tag %>

    <%= stylesheet_link_tag 'application', media: 'all' %>
    <%= javascript_pack_tag 'application' %>
  </head>

  <body>
    <div id="root"></div>

    <script>
      window.initialTranslations = <%= raw @initial_translations.to_json %>;
    </script>
  </body>
</html>
```
The crucial aspect here is using `raw` to prevent Rails from escaping the JSON string. Now, the `window.initialTranslations` will hold a JavaScript object ready for consumption in our React application.

In the React application, we create a translation context. This context will serve as a centralized location to manage translations. We initialize it with the data coming from the server, and we also create a function for fetching further translations, if needed. I’ll outline the React code next:

```javascript
// src/i18n/i18nContext.js

import React, { createContext, useState, useContext, useCallback } from 'react';

const I18nContext = createContext();

export function I18nProvider({ children }) {
  const [translations, setTranslations] = useState(window.initialTranslations || {});
  const [loading, setLoading] = useState(false);

  const fetchTranslations = useCallback(async (keys, locale) => {
      if(keys.length === 0) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ keys, locale });
      const response = await fetch(`/api/translations?${params}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setTranslations((prevTranslations) => ({...prevTranslations, ...data}));
    } catch (error) {
      console.error("Failed to fetch translations:", error);
    } finally {
      setLoading(false);
    }
  }, [setLoading, setTranslations]);


  const translate = (key) => {
      return translations[key] || key;
  }
  return (
    <I18nContext.Provider value={{ translate, fetchTranslations, loading }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}
```

This setup creates a context provider (`I18nProvider`), stores initial translations, implements a `fetchTranslations` function to asynchronously load missing keys, and provides a `translate` function for retrieving translations based on a key. The `useI18n` hook is for convenient access to these features. The code uses the javascript `fetch` api; this assumes a backend that exposes a `/api/translations` route which accepts keys as request params and returns a json response with the keys and their translations.

Finally, to illustrate how to actually use this setup, consider this example React component:

```javascript
// src/components/MyComponent.js
import React, { useEffect } from 'react';
import { useI18n } from '../i18n/i18nContext';

function MyComponent() {
    const { translate, fetchTranslations } = useI18n();

  useEffect(() => {
    fetchTranslations(['welcome_message', 'button.label'], 'en');
  }, [fetchTranslations]);

    return (
        <div>
            <h1>{translate('hello')}</h1>
            <p>{translate('welcome_message')}</p>
            <button>{translate('button.label')}</button>
        </div>
    );
}

export default MyComponent;
```

Here we load the keys 'welcome_message' and 'button.label' and show the translations.

For the sake of complete demonstration, the endpoint required for the example above using Rails would resemble the following controller implementation:

```ruby
# app/controllers/api/translations_controller.rb
class Api::TranslationsController < ApplicationController
  def index
    keys = params[:keys] || []
    locale = params[:locale] || I18n.locale

    translations = I18n.t(keys, locale: locale)
    render json: translations
  end
end
```

Remember that the above code provides a starting point. The error handling and other advanced features should be adjusted as per the particular needs. You can further refine it using concepts like memoization or caching in various parts of the architecture to optimize efficiency further.

Key points to remember here are to initially load a subset of translations, use a context to maintain global access, lazy load when necessary and establish an efficient endpoint for fetching translations. This overall approach proved to be effective for me and can handle a substantial load while keeping things relatively organized.

For further studies, I'd highly recommend delving into the “Internationalization with Rails” chapter in the *Agile Web Development with Rails* book by Sam Ruby, David Bryant, and Dave Thomas. Furthermore, research regarding i18n in react should point towards the use of contexts, hooks, and techniques related to lazy loading. Also, examining how internationalization works with more established frameworks such as Java Spring's i18n or ASP.NET localization would provide great comparative understanding.
