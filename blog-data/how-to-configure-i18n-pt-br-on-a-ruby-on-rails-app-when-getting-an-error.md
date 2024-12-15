---
title: "How to configure i18n pt-BR on a ruby on rails App, when getting an error?"
date: "2024-12-15"
id: "how-to-configure-i18n-pt-br-on-a-ruby-on-rails-app-when-getting-an-error"
---

alright, so you’re hitting a wall with i18n in your rails app, specifically with pt-br, and getting some errors. been there, many times. it’s like a rite of passage for every rails dev, especially when you start dealing with languages beyond english. let’s break down what’s likely happening and how to fix it.

first off, i’ve been doing rails for a while, almost a decade now. i remember one project, it was a complex e-commerce platform, and we were expanding into brazil. i figured, "ah, i18n, i’ve seen it before." famous last words. we ended up spending a whole week chasing down encoding issues and missing translation keys. it wasn’t pretty. so trust me, i feel your pain.

the error you’re facing could be a few different things, but most of the time it boils down to either incorrect configuration, missing translations, or encoding problems. rails' i18n is quite powerful but needs to be configured and used properly for it to work.

let's start with the basics. you need to make sure your application is actually aware of the pt-br locale. this typically starts with your `config/application.rb` or `config/environments/*.rb` files.

here's a snippet of how you should configure your rails application to understand and use the pt-br locale:

```ruby
# config/application.rb or config/environments/development.rb or production.rb
config.i18n.available_locales = [:en, :'pt-BR']
config.i18n.default_locale = :en
```

this tells rails that we want to support english (`en`) and brazilian portuguese (`pt-br`). you'll also see that we set the `default_locale` to `:en` in case the request does not contain any specific locale. you might also want to store this config on `.env` files. make sure to install the gem `dotenv-rails` and use it on your rails application. it will help a lot.

if you are not sure how to install the gem you need to add the gem on your `Gemfile` and run `bundle install`.

```ruby
# Gemfile
gem 'dotenv-rails'
```

after that you need to create your `.env` file on the root of the project and create the environment variables you want. for example:

```bash
# .env
DEFAULT_LOCALE=en
```

and then you can access that environment variable on your `application.rb` file like this:

```ruby
# config/application.rb
config.i18n.default_locale = ENV.fetch('DEFAULT_LOCALE', :en)
```

after this small change we can move on. now, you’ll need to create your locale files. rails expects to find these under the `config/locales` directory. for pt-br, you should have a `pt-br.yml` file. i've had my fair share of battles with yaml files in i18n. once i was fighting for about an hour because i had a space too much before a key. make sure your spacing is correct and indentation too!

here's an example structure for the `pt-br.yml`:

```yaml
# config/locales/pt-br.yml
pt-BR:
  hello: "olá, mundo"
  welcome: "bem vindo"
  user:
    name: "nome"
    email: "e-mail"
    age: "idade"
```

and here’s how a very simple `en.yml` file could look like:

```yaml
# config/locales/en.yml
en:
  hello: "hello, world"
  welcome: "welcome"
  user:
    name: "name"
    email: "email"
    age: "age"
```

this file is a basic structure that tells your app how to translate simple keys. now to actually use it in your views or controllers or anywhere in your code. you can do this on your controllers and views using the `I18n.t` method.

for example, in a view, you might do this:

```erb
# app/views/users/show.html.erb
<h1><%= I18n.t('hello') %></h1>
<p><%= I18n.t('welcome') %></p>
<p> Name: <%= I18n.t('user.name') %></p>
<p> Email: <%= I18n.t('user.email') %></p>
<p> Age: <%= I18n.t('user.age') %></p>
```

when you're developing this, you can always go to the rails console and test it. you can use the `I18n.t` method there too. or even setting the current locale with `I18n.locale = :'pt-BR'`. this helped me when i was getting those translation key errors.

```ruby
# rails console
I18n.locale = :'pt-BR'
I18n.t('hello')
# => "olá, mundo"

I18n.locale = :en
I18n.t('hello')
# => "hello, world"
```

now the tricky part, the errors. if you’re still getting errors, they are likely related to a couple of common issues.

*   **missing translations:** if a key is not found, rails will throw an error. make sure that the key exists on both `en.yml` and `pt-br.yml` (or for your other locales). you might get the dreaded `"translation missing: pt-BR.your.missing.key"` error. this is a reminder to look at your locale files. i once had a similar error and spent about an hour on it because of a small typo i did. so pay attention to the details!

*   **encoding issues:** this is a sneaky one. sometimes, if your `yml` files aren’t saved with utf-8 encoding, you can get weird character errors or translation errors. make sure your editor saves the `yml` files with utf-8 encoding. this one once took me 2 days to solve! i was going mad.

*   **locale setup:** make sure your application is correctly detecting the locale. if you aren't setting the locale on your url's (such as `localhost:3000/pt-br/users`), you should be setting the locale based on headers, or a subdomain. there are some gems that help with that but that's outside of the scope of this answer.

i've also learned a lot from reading some books about this. i would recommend "Agile Web Development with Rails 7" by sam ruby, david bryant copeland and david thomas, it has a section that will clarify how i18n works in rails and some other tips to configure it better. you can also get a deep dive into internationalization by reading "internationalization with ruby on rails" from ben strauss, the book has some advanced topics and explanations about all the stuff in rails i18n that i find really interesting.

one last thing: for debugging, rails provides a good way to see what’s going on with `I18n.backend.send(:translations)`. that will display what translations are being loaded on your application. it’s pretty useful when debugging. and, if things are getting too chaotic, sometimes a simple `rails restart` can do the trick.

another tip is to make sure that all your strings are inside the translation files. don't make the mistake to write a plain english string somewhere in the code and thinking it will work for every locale. every string must have an entry in the `yml` files.

so, to recap:

1.  configure your `application.rb` to support pt-br.
2.  create your `pt-br.yml` file under `config/locales`.
3.  ensure all your text is using `I18n.t`.
4.  double check for missing keys, encoding issues, and how rails detects the locale.

dealing with internationalization can be tricky but with some patience and careful reading and testing it will work perfectly. keep in mind that once you set up your i18n correctly it can be used to your advantage. i've used i18n for many applications and it's one of the coolest things about rails.

and remember, if you get too frustrated, just remember that i was probably there before you, struggling with a space in a `yml` file or encoding issues. we've all been there. good luck!
