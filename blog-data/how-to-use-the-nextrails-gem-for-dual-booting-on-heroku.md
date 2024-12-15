---
title: "How to use the next_rails gem for dual booting on heroku?"
date: "2024-12-15"
id: "how-to-use-the-nextrails-gem-for-dual-booting-on-heroku"
---

alright, let's talk about getting next.js and rails playing nicely together on heroku, specifically with that next_rails gem. i've been down this rabbit hole before, and it can get a little hairy if you're not careful. it's not as straightforward as deploying a single rails or next app, but it's manageable once you understand the pieces involved.

basically, what next_rails does is it sets up a kind of hybrid system. you still have your main rails app handling all the backend stuff, like api endpoints and data persistence, but it also acts as a proxy for your next.js frontend. this is often called a monorepo or a hybrid app approach. the next_rails gem helps with the connection, setting up the routing and the build process.

first off, let's recap what you'll likely need before starting. you will need to have node.js and yarn or npm installed. a functioning rails application with postgresql, and also a working next.js application. these should be on their own separate directories at first, i found it easier to structure it this way before merging them into one. once those are ready, you install the gem in your rails application and begin integrating next within rails with it. i spent a good chunk of a summer in 2021 trying to make this work properly on a side project. it was a full on struggle, believe me.

so, assuming you've got all that, here's the basic rundown of how it usually goes.

1.  **gem install and initial setup**. you start by adding `gem 'next-rails'` to your `gemfile`, and run `bundle install`. this pulls in the gem and its dependencies. then, you run `rails generate next_rails:install` this command sets up a `next` directory in your rails app, and adds the basic scaffolding. it creates a `package.json` and a basic next app structure inside this directory. i spent a whole evening once, banging my head against a wall, only to find i had not even done the `bundle install`, i felt so dumb that day.

    ```ruby
    # Gemfile
    gem 'next-rails'
    ```

    after running bundle install then run the generator:

    ```bash
    rails generate next_rails:install
    ```

2.  **moving your next.js app**. now you'll want to move the contents of your existing next.js application into the `next` folder created in the previous step. this will overwrite the basic scaffold generated. after the copy, carefully inspect if there are any issues with any kind of absolute paths. you may need to tweak the package.json with the correct dependencies and scripts. if you are using yarn, consider adding the same version of yarn for the project level and also the one inside the next directory. consistency is key here.

3.  **configuring rails to serve next.js**. the key to understanding how this works, is that rails now serves the next application through a proxy. the next_rails gem adds some routing to rails, so requests to your frontend endpoints get routed to the next server that it spins up. this usually means that requests like `/` or `/some/page` get handled by the next.js app, and other request to `/api/*` go to rails backend.

    take a look at this example in rails `config/routes.rb`:

    ```ruby
    # config/routes.rb
    Rails.application.routes.draw do
      # other routes here

      # this will make sure all non rails assets are handled by the next application
      mount_next_app at: '/', constraints: lambda { |request| !request.path.start_with?('/api') }
    end
    ```

    this bit of code ensures any request that doesn't start with `/api` goes to the next.js application. this allows for both apps to operate together. the constraint on the route is important. without it you would run into routing issues. in my earlier attempts i always forgot about that, and it was a mess.

4.  **heroku deployment configuration**. deployment on heroku involves some tweaks. since you're running two separate processes (rails and next.js), you will need a `procfile` that starts both apps. this is where many people hit a snag. your `procfile` should look like this:

    ```
    web: bin/rails server -p $PORT -b 0.0.0.0
    next: bundle exec rails next:start
    ```

    `web` is the rails app, and the `next` worker starts the next.js development server. heroku automatically assigns an ephemeral port using the `$PORT` variable for the web process but the next process will be running on a hardcoded port inside next application. by default it uses 3000, which the rails app proxies to. to properly run the next.js application on production, you would need to build it, and run that build.

    i was once deploying the development version of the next.js app on production. the app was slow, and it kept getting restarted all the time, i did not understand why, until i read the documentation again, and did the build, it is really important to run the production build on the next process.

    here's how the updated procfile should look:

    ```
    web: bundle exec rails server -p $PORT -b 0.0.0.0
    next: cd next && npm run build && npm run start
    ```

    notice, i'm using `npm` here, but you may have to adjust the `cd next && npm run build && npm run start` part depending on your next setup. `npm run build` builds your next project and generates a folder called `.next` this folder contains all the files necessary to run the next application.

5.  **handling assets and static files**. you will need to ensure that your static assets from the next app are accessible. usually, next takes care of this automatically. in your rails app's `config/environments/production.rb` or in `config/application.rb` verify that you have set correctly the assets prefix. usually is something like this:

    ```ruby
    # config/environments/production.rb
    config.assets.prefix = '/assets' # or whatever you need it to be
    ```

    this will make sure that the static assets created by next and served by rails are properly loaded. usually this is handled by next itself, but sometimes you might have to tweak it.

6.  **dealing with environment variables**. when you're working with both next.js and rails, make sure you're handling your environment variables correctly. heroku provides a way to set config vars, and both apps need to access them. make sure that those secrets are set on heroku, and be careful with variables being exposed on the client side. never ever expose keys or tokens directly to the browser. you can use the `.env` file on the next folder, but that is only meant for local development.

    and finally, since you're running two applications on the same platform, you should be aware that heroku allocates cpu based on the amount of process on the dynos. if the next process takes a lot of cpu power to build, the rails application might not have enough to even start. so, usually, on a low tier free heroku dyno, the build process might take forever, and even crash. you may have to use different strategies if the build takes a lot of resources. some of them may include adding more memory to the dyno or using a third party build process.

now, for resources, rather than just dumping links, let me point you to some places that helped me along the way:

*   **the official next.js documentation**: their docs are phenomenal, and it is essential to understand all the nuances of next. i recommend you read their build and deployment documentation carefully.
*   **the rails guides**: the official rails guides are amazing. especially the ones about production deployment and how to handle assets. there is a specific guide for asset pipelines.
*   **"designing data-intensive applications" by martin kleppmann**: although this book is not directly related to the issue at hand, it taught me that there is more that one path to solve a problem, and you need to carefully evaluate the trade-offs, it is an amazing book nonetheless.

the last time i had a similar issue, it involved a complicated path configuration. it turns out, i was using the incorrect path on the next.config.js file. after some debugging with `console.log()` i was finally able to identify it. the lesson here is: always double check every line of code.
i hope this gets you moving. it's a tricky setup, but once you get the hang of it, you'll be flying. let me know if you get stuck. it is always fun to find new issues to debug. good luck.
