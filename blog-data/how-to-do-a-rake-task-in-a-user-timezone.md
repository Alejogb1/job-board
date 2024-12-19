---
title: "How to do a rake task in a user timezone?"
date: "2024-12-15"
id: "how-to-do-a-rake-task-in-a-user-timezone"
---

alright, so you're hitting the classic timezone wall with rake tasks, right? been there, done that, got the t-shirt… and probably a few lingering headaches. i remember back in '09, working on this monstrosity of a rails app for a global e-commerce platform, we had users all over the place, and scheduling stuff was, well, a *mess*. cron jobs firing off at seemingly random times, emails going out at 3 am for some folks while it was lunchtime for others… absolute chaos. lesson learned: timezone awareness is crucial, especially when background jobs are involved.

the core problem, as i see it, is that rake tasks, by default, operate in the server’s timezone. the clock on your server isn't the clock of your users, and if your app caters to people across different timezones you've got a major problem. the user's timezone is a piece of contextual data you need to carry with you. you can't assume it, or everything goes haywire.

the first thing you need to grasp is that this isn't really a 'rake' specific issue but more of a 'background process' one. rake tasks are just ruby code, and if that code is making use of time-sensitive operations, it must know the user’s timezone. so, here's how i've generally handled this, breaking it down:

**1. capturing the user's timezone:**

before we get into the rake task bit, let's nail the user timezone part. usually, this means storing the timezone in the user's record.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  #... other stuff
  def timezone
      self.time_zone || 'utc'
  end

  def in_user_timezone(&block)
    Time.use_zone(timezone, &block)
  end
end
```

the `time_zone` attribute should be a string from the `tzinfo` gem. you will need this. i have seen it with bad data before. i remember spending an entire day to just find out we had a user with a 'not a timezone' timezone entry that was breaking the job for everyone. so please remember sanitizing your inputs, i learned it the hard way. the `in_user_timezone` method there is a convenient helper so we can easily wrap the code we need. we are defaulting to utc just in case there is no data for a user. it is just better to have it defined.

**2.  passing the context to the rake task:**

now, the actual rake task. you need a way to pass in the user's id (or whatever identifies the user), so that the rake task can load the user's timezone information. i like passing user ids via environment variables in a good old bash call.

here's a basic example rake task:

```ruby
# lib/tasks/my_task.rake
namespace :my_tasks do
  desc "process something for a user"
  task :process_user, [:user_id] => :environment do |t, args|
    user_id = ENV['USER_ID'] || args[:user_id]
    unless user_id.present?
      puts "error: user id not set"
      exit 1
    end
    user = User.find(user_id)
    user.in_user_timezone do
       # process here based on the user’s current time
       puts "processing in user #{user.id} time: #{Time.now}"
       # maybe send an email here or do a calculation
    end
  end
end
```

you'll see i am retrieving the user id from the environment `user_id`, and if it's not set i take it from the parameters passed to the rake task, which is the usual way of calling rake tasks, from the console for example. `user.in_user_timezone` makes sure that any calls to methods like `Time.now` will use the user's specific timezone. this is super important! otherwise, all timestamps within this block will be relative to the server's timezone.

**3. invoking the rake task:**

how would you actually invoke this rake task? well, it depends on how you want to schedule or run your rake task. normally, when i need a manual trigger i just do this:

```bash
export USER_ID=123 #replace 123 with your user id
bundle exec rake my_tasks:process_user
unset USER_ID
```

i always unset the environment variable after running it, because i am paranoid, in case you see it weird or just too much. for the scheduled jobs, i normally do something like this:

```bash
#example crontab entry
0 0 * * *  cd /path/to/your/app && export USER_ID=$(/path/to/your/app/get_user_ids.sh) && bundle exec rake my_tasks:process_user && unset USER_ID
```
this is a bit more complex, and the `get_user_ids.sh` script is something that returns a list of user ids, in a loop and executing the rake task per user. the script should contain something like:
```bash
# get_user_ids.sh
ruby -e 'puts User.pluck(:id).join(" ")'
```

**important considerations and why this works:**

*   **tzinfo:** the gem `tzinfo` is used by active support and helps in dealing with the various timezone databases and how they shift and change their offsets. i highly recommend reading the documentation of this gem, it is very valuable and helps you understand a lot of the problems related to timezones.

*   **active support:** active support in rails makes timezone handling relatively easy. if you're not using rails, the principle is the same, but you'll have to bring your own time zone handling code. `time.use_zone` method is the best friend of any backend engineer. you should read a bit about `time.zone` in the rails documentation.

*   **persistence:** storing the timezone on the user record, or related data table, is not only a good practice, but absolutely necessary. it’s the context your app needs. i have tried to 'infer' the timezone of the user by their location or ip, and it always backfired. the best approach is to let the user set it manually or let the app know it through the client settings, and then storing it.

*   **testing:** of course, don't forget to test your timezone logic. create some test users with different timezones and check everything works as expected. this should be integrated into your build and deployment pipeline.

*   **edge cases:** you may have some special edge cases like users who travel. this is when having a timestamp for the timezone is important. also, you may need to refresh the timezone if it has changed, for example, if the user travels. a good read here is "time and date in ruby" by Stefan Wintermeyer. this book goes really deep in the underlying problems. also, maybe look at the “managing time in ruby on rails” book by jake yesbeck.

*  **performance:** you need to consider the impact this could have in your app. if you iterate in millions of records, think about how you load the data of users in batches or by chunks, so you do not stress the database. i have seen many servers failing because of this issue.

*   **logging:** make sure you're logging the timezone used for each job. this will help you troubleshoot problems later. if you run a big batch of jobs and everything fails at one point it is hard to pinpoint the exact cause without the proper logging.

one time, i accidentally ran a rake task with the wrong timezone for *all* users, and it sent out thousands of notifications at the wrong hours. let's just say that my boss wasn't too happy, and that's why i am sharing all this information now. never again will i repeat this mistake, and hopefully, with this information, neither will you. i also remember reading a paper on the implementation of timezones in mysql, that helped understand a lot of how they handle timezone conversions. it is hard to remember the paper because it was something from 2012 or so, but it helped a lot.

this approach has worked reliably for me in production environments. it may look simple, but that's exactly the goal: keep it simple. remember to always prioritize clarity and avoid overcomplicating.
