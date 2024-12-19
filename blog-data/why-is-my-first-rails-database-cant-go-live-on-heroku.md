---
title: "Why is my first rails database can't go live on heroku?"
date: "2024-12-15"
id: "why-is-my-first-rails-database-cant-go-live-on-heroku"
---

so, your rails app's database isn't playing ball with heroku, right? i've been there, spent way too many late nights staring at cryptic heroku logs. let's break it down, this is usually a mix of a few common culprits and i've probably seen all of them at some point in my career, which is longer than i'd care to disclose.

first off, let's talk about the database itself. heroku doesn't just magically know about your local postgres setup. usually, you're using sqlite in development, which is perfect for getting things rolling fast, but it's a no-go for production on heroku. it's a different system, so the connection stuff is all different. heroku expects you to use postgres, and they provide a managed postgres service (or you can bring your own, but let's keep it simple).

most likely, your `database.yml` file, that's in `config/database.yml` is not set up for heroku's postgres database. heroku's postgres database connection info is given through an environment variable named `database_url`. this is usually not something you set manually it's handled by heroku when you provision the database. your local setup will be different, so it's all about managing environments. your `database.yml` should handle both cases. i've messed this up more times than i'd like to remember. i had this one time where i accidentally pushed a local database.yml config that was pointing to localhost which then broke everything and cost me a good part of a day to fix, lesson learned the hard way.

here's how your `database.yml` might need to look, note the key difference for production:

```yaml
default: &default
  adapter: postgresql
  encoding: unicode
  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>
  username: <%= ENV['PGUSER'] || ENV['USER'] || 'postgres' %>
  password: <%= ENV['PGPASSWORD'] || '' %>
  host: <%= ENV['PGHOST'] || 'localhost' %>
  port: <%= ENV['PGPORT'] || 5432 %>

development:
  <<: *default
  database: my_app_development

test:
  <<: *default
  database: my_app_test

production:
  <<: *default
  url: <%= ENV['DATABASE_URL'] %>
```

see that `url:` line under production? that's the key. it picks up the `database_url` heroku provides. we also set up default values for development to ensure our local environment works as expected. it's not magic, just a way to make sure your database configurations are correct in different environments. notice how we also set default username, password, host and port which is nice to have when dealing with local development with postgres. this helps a lot with local configurations.

now, assuming you've got that sorted, the next common issue is migrations. heroku doesn't automatically run your migrations. you need to tell it to do so. after pushing your code to heroku, you'll want to run:

```bash
heroku run rails db:migrate
```

this command executes your pending migrations. i had this one time where i forgot to do this and just kept wondering why nothing was working, felt a bit silly after that. it creates tables, adds columns, and makes the database match your model setup in ruby. also, ensure you don't have any pending migrations locally before deploying. usually if you are using git, you want to make sure you've applied your local migrations before pushing your code or heroku will complain. if that happens you need to do something like a `heroku run rails db:migrate:status` and `heroku run rails db:migrate`. it can get complicated fast.

another problem i've seen several times is not having the `pg` gem in your `gemfile`. it's the postgres adapter for ruby. check you have it in your `gemfile`:

```ruby
# Gemfile
gem 'pg', '~> 1.4'
```

if it's not there, add it and run `bundle install`, commit and push it. if you forget to commit the lock file, you'll be spending an extra time deploying, also a mistake i did more times than i am willing to admit. this might seem obvious but it is something that happens quite frequently.

also, check that you've provisioned a postgres database on heroku. it's not enough to just have it in the config, you need the actual heroku postgres add-on. usually it's just as simple as:

```bash
heroku addons:create heroku-postgresql:hobby-dev
```

this command creates a free postgres database that you can use to start with. make sure this is done before you run your migrations for the first time. another really common mistake is to forget to push your assets to heroku. this can happen if you are having some issues with `precompile` assets. to fix that you can run `heroku run rails assets:precompile` and it's usually a good idea to do a `heroku restart` after this for good measure. it is one of the most overlooked steps, as most developers would assume that deploying the code is everything, but sometimes that is not the case and you have to also tell heroku that you need to precompile the assets. usually when you deploy and your website is looking weird it is because this step was forgotten. it's like ordering a pizza and forgetting to tell them you actually want it cooked, you know?

one more thing that often trips people up is the use of environment variables. heroku handles things differently from your local machine. on heroku, environment variables are typically set through the heroku command-line interface, or the heroku dashboard, rather than defined in some `.env` file. it is important to learn and understand how environment variables work, especially in the heroku context. for example, if you are using some api key for an external service, it might not work as expected if you are not correctly setting them in heroku. something like this needs to be run: `heroku config:set API_KEY=your_actual_key`.

if after checking all that things are still not working, you need to inspect the heroku logs. this helps to understand what is going on with your app in heroku. run the following:

```bash
heroku logs --tail --app your-app-name
```
replace `your-app-name` with the actual name of your app. this will give you a live view of what is happening on heroku servers. look for error messages. usually errors are pretty explicit and you can google them or at least you'll get a direction to fix the issue you're experiencing. heroku logs are your best friend when debugging deploy issues.

also, if you are using any external dependencies, like redis or other databases, make sure you provision them correctly and have the connection strings configured right. each one has its own specific way to do this and it is important to be aware of how it is done for each particular service you are using. if you use a cache like redis, you might also need to provision that in heroku and point your app to that. it's a common pattern to not think about it until it is too late. it is one of those situations that are very difficult to debug, until you actually find it, then it is obvious, but it usually involves more than one hour of lost time.

lastly, keep in mind heroku's dynos work differently from your local development environment. dynos restart, they scale, they might have resource limits that you don't experience locally. if something works locally but not in heroku, it's often a configuration issue related to environment variables, connection details, or migrations. i can't emphasize this enough, start simple, don't go fancy with configurations and start from the beginning when debugging these issues.

i would recommend, for more in depth understanding, to read the official heroku documentation which has a lot of examples and specific details on how to deploy rails apps, also you can find in rails guides specific explanations on how to configure rails apps to work with a production environment. also, "the rails 7 way" is a good book to learn all these details. it will make the whole experience deploying rails app to production a lot less stressful.

i hope this helps. deployment can be a bit of pain sometimes, but once you get the basics, it becomes much easier.
