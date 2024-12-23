---
title: "How can I change the language in a React/Rails application using I18next, either through URL parameters or server-side rendering?"
date: "2024-12-23"
id: "how-can-i-change-the-language-in-a-reactrails-application-using-i18next-either-through-url-parameters-or-server-side-rendering"
---

, let’s dive into this. I've faced this exact challenge a few times, particularly on one project involving a multi-national user base. Getting internationalization (i18n) working smoothly in a React/Rails application can be tricky, but it's immensely satisfying when done correctly. We're talking about a setup using i18next, which is a solid choice for managing translations, combined with React on the front end and Rails on the backend. We'll address both URL parameter-based language switching and server-side rendering (SSR) approaches, outlining the pros, cons, and necessary configurations.

First, let's nail the core concept: locale detection. You need to figure out which language your user prefers. There are a few ways to determine this; user browser settings, explicitly selected by the user in your application, or through some default setting. Today, we are going to focus on the URL parameter and SSR methods.

**URL Parameter-Based Language Switching**

The first method involves parsing the URL for a language code. Imagine a user navigating to `https://example.com/en/products` or `https://example.com/fr/products`, with 'en' representing English and 'fr' representing French. The approach is to intercept this language segment in React and use it to tell i18next which translation set to load.

Here’s how I’d usually handle this, building on i18next's capabilities:

*   **Configuration:** You'll need to set up i18next with a basic configuration specifying resources (your translation files). The following code shows a basic configuration. Usually I'd use a single `i18n.js` file that handles the config for the whole app.

```javascript
    // i18n.js
    import i18n from 'i18next';
    import { initReactI18next } from 'react-i18next';

    import en from './locales/en.json';
    import fr from './locales/fr.json';

    i18n
    .use(initReactI18next)
    .init({
        resources: {
          en: { translation: en },
          fr: { translation: fr },
        },
        fallbackLng: 'en', // The default language
        interpolation: {
            escapeValue: false, // Not needed for React
        },
    });

    export default i18n;
```

*   **Route Handling:** In your React router, you'll need to parse the URL. I prefer using a library like `react-router-dom` for routing. Inside a route component, you will need to extract the locale from the path params and update the `i18n` instance.

```jsx
    // App.jsx
    import React, { useEffect } from 'react';
    import { BrowserRouter as Router, Route, useParams } from 'react-router-dom';
    import i18n from './i18n'; // Import the config file
    import ProductsPage from './ProductsPage'; // Example of the main app content
    import HomePage from './HomePage';
    import { I18nextProvider } from 'react-i18next';


    function LanguageSwitcher() {
      const { lang } = useParams();

      useEffect(() => {
        if (lang && i18n.language !== lang) {
          i18n.changeLanguage(lang);
        }
      }, [lang]);

      return null; // This component does not render anything
    }

    function App() {
      return (
      <I18nextProvider i18n={i18n}>
        <Router>
           <Route path="/:lang/" component={LanguageSwitcher} />
           <Route exact path="/:lang/" component={HomePage}/>
          <Route path="/:lang/products" component={ProductsPage} />
        </Router>
       </I18nextProvider>
      );
    }

    export default App;
```

*   **Usage:** Now, any component can use the `useTranslation` hook to get translations.

```jsx
    // ProductsPage.jsx
    import React from 'react';
    import { useTranslation } from 'react-i18next';

    function ProductsPage() {
        const { t } = useTranslation();

        return (
            <div>
            <h1>{t('products.title')}</h1>
            <p>{t('products.description')}</p>
            </div>
        );
    }
    export default ProductsPage;
```

This approach is straightforward and works well for most situations. However, it has a few drawbacks, one being that the initial render is in the fallback language while `changeLanguage` is applied. This can lead to a flash of untranslated content. Also, you lose some SEO points with this setup.

**Server-Side Rendering (SSR) with i18next**

To avoid the flash of untranslated content and improve SEO, server-side rendering is often preferred. This is where the Rails backend becomes crucial. We need Rails to figure out the requested locale and pass it to the React app during the initial render. This means that the React application renders with the correct language from the beginning of the process.

Here’s the general procedure:

*   **Backend Locale Determination:** In your Rails controller, inspect the incoming request, determine the desired language (e.g., from the url path, user preferences or an accepted language header) and pass that as an initial prop to React. For example, if you have a path like `/en/products`, you will want to extract the `en` and pass it on as part of your initial payload:

```ruby
    # app/controllers/products_controller.rb
    class ProductsController < ApplicationController

    def index
      locale = params[:lang] || I18n.default_locale # Or whatever logic

      render_react_app(locale)
    end

    private
      def render_react_app(locale)
        render html: "<div id='root'></div>".html_safe, layout: 'application', locals: {locale: locale}
      end
    end
```

*   **Initial Data Injection:** Inside the rails layout file, you will need to inject this initial locale so your react app is able to use it.

```erb
    <!-- app/views/layouts/application.html.erb -->
    <!DOCTYPE html>
    <html>
        <head>
            <title>My App</title>
            <%= csrf_meta_tags %>
            <%= csp_meta_tag %>
            <%= stylesheet_link_tag 'application', media: 'all', 'data-turbolinks-track': 'reload' %>
            <%= javascript_pack_tag 'application', 'data-turbolinks-track': 'reload', data: { locale: locals[:locale]} %>
        </head>

        <body>
            <%= yield %>
        </body>
    </html>
```

*   **React App Initialization:** You’ll modify your React app to use the passed initial locale. Your `App.jsx` will be altered to pick the language up.

```jsx
    // App.jsx
    import React from 'react';
    import { BrowserRouter as Router, Route } from 'react-router-dom';
    import i18n from './i18n'; // Import the i18n configuration
    import ProductsPage from './ProductsPage';
    import HomePage from './HomePage';
    import { I18nextProvider } from 'react-i18next';

    function App(props) {
        const { locale } = props;
        if(i18n.language !== locale) {
            i18n.changeLanguage(locale);
        }

        return (
        <I18nextProvider i18n={i18n}>
             <Router>
                <Route exact path="/" component={HomePage}/>
                <Route path="/products" component={ProductsPage} />
           </Router>
          </I18nextProvider>
       );
     }

   export default App;

```

*   **Webpack:** You will also need to add the `locale` param to your react component. This will look like this:

```javascript
 // application.js
 import React from 'react';
 import ReactDOM from 'react-dom';
 import App from '../components/App';

 document.addEventListener('DOMContentLoaded', () => {
  const locale = document.body.querySelector('[data-locale]').dataset.locale;
  ReactDOM.render(
     <App locale={locale}/>,
     document.getElementById('root'),
   );
 });
```

With this, when a user requests `/en/products`, the Rails server determines the locale as 'en' and provides it to the React app which then renders immediately with the correct English translations.

**Resource Recommendations**

For deeper understanding, I'd recommend the following:

1.  **"Internationalization in React" by Flavio Copes:** This is an excellent introductory guide. While specific examples might be slightly outdated, the fundamental concepts are thoroughly explained. You will find that it will walk you through the i18next setup as well as using the hook.
2.  **i18next Documentation:** The official documentation for i18next is comprehensive and covers all aspects of configuration, usage and implementation.
3.  **"React Router" Documentation:** Having a strong understanding of React Router is essential for URL parameter-based approaches. Read the latest documentation for version 6.
4. **"The Pragmatic Programmer" by Andrew Hunt and David Thomas:** While not specifically about i18n, this book provides a general philosophy and best practices that can improve your overall development practices, particularly when tackling complex application problems.

**Conclusion**

Both URL parameter parsing and server-side rendering approaches are viable for handling translations in a React/Rails app using i18next. URL parameter routing offers simplicity for many applications and can be an easy implementation to begin with. However, when SEO and initial page load performance are paramount, server-side rendering is the superior option. From my personal experience, I found the SSR approach is much better for larger projects with a global user base. The slight extra work is very much worth it in the long run.
