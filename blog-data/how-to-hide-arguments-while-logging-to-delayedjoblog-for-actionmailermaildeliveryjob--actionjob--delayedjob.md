---
title: "How to hide arguments while logging to delayed_job.log for ActionMailer::MailDeliveryJob / ActionJob / DelayedJob?"
date: "2024-12-14"
id: "how-to-hide-arguments-while-logging-to-delayedjoblog-for-actionmailermaildeliveryjob--actionjob--delayedjob"
---

so, i see your question about hiding arguments when logging with delayed\_job, actionmailer, actionjob – yeah, i’ve been there. it's a common pain point when dealing with background jobs and sensitive data. the standard logging can be a bit too verbose, especially when you're passing passwords or api keys as arguments. those end up plastered all over the log files, which is not good. let me tell you about a similar situation i had once and then i’ll share my solutions.

i remember this one time, back when i was working on a legacy app – it was a real beast – the kind where you spend half your day deciphering what some previous dev did 5 years ago. we were using delayed\_job for sending emails, and i noticed that the whole email sending logic with all the parameters including user details was logged into the `delayed_job.log`. one of the args was a user’s reset token for password recovery. yikes! talk about a security flaw. needless to say, that was a code red situation. i had to scramble to fix that, and that’s when i started exploring ways to sanitise the logs. i spent a good couple of days deep into rails internals, delayed\_job code and figuring out what was going on.

the core issue lies in how delayed\_job, actionmailer, and actionjob typically serialize job arguments for logging. they often just call `inspect` or `to_s` on the arguments which shows too much detail. so, what i ended up implementing involved overriding how those arguments are logged. here are the patterns i’ve found to work consistently:

first, the straightforward method is to modify the job itself. it's a good approach if you have control over the job’s definition. you can redefine the `perform` method to log the sanitised arguments instead of letting delayed\_job log the original ones. here’s an example for an actionmailer mail delivery job:

```ruby
class MyMailer < ApplicationMailer
  def welcome_email(user, extra_info = {})
    @user = user
    @extra_info = extra_info
    mail(to: user.email, subject: "Welcome")
  end
end

class ActionMailer::MailDeliveryJob
  def perform(*args)
    mailer, method_name, mailer_args = args
    sanitized_args = mailer_args.map { |arg|
      case arg
      when User
        "<user:#{arg.id}>"
      when Hash
          arg.transform_values {|v| v.is_a?(String) ? "[REDACTED]" : v }
      else
        arg
      end
    }

    Rails.logger.info "performing #{mailer}.#{method_name} with args #{sanitized_args}"
    super(*args)
  end
end

```

in this snippet, i've overridden the `perform` method within `ActionMailer::MailDeliveryJob`. when you call `MyMailer.welcome_email(user, {password: 'secret'})`, instead of logging the actual user and the password, it logs `<user:123>` (assuming user id is 123), and `[REDACTED]` for any string value in hashes. i'm assuming you are using a `User` model and also assuming the user information is sensitive. you'd need to adjust this based on your actual data structure. if you are passing api keys, you might have to add another `when` case in the `case` statement to handle api keys. it's important to note that you will need to restart the application if you change the ActionMailer classes.

note that this example is for `ActionMailer::MailDeliveryJob`, but it will work mostly the same with any `ActionJob` class. you can use the same principle with actionjob or delayedjob classes, all you need to do is override the `perform` method and sanitise the args before logging.

the second method involves using a custom logger. this can be beneficial if you want more control over logging across the board or if you don’t want to modify each individual job. you can create a custom logger that pre-processes the log messages. first, you need to implement a custom logger:

```ruby
class SanitizedLogger < ActiveSupport::Logger
  def add(severity, message = nil, progname = nil, &block)
    message = block.call if message.nil? && block

    if message.is_a?(String)
        sanitized_message = message.gsub(/"password"=>"[^"]*"/, '"password"=>"[REDACTED]"')
        sanitized_message = sanitized_message.gsub(/password: "[^"]*"/, 'password: "[REDACTED]"')
        sanitized_message = sanitized_message.gsub(/api_key: "[^"]*"/, 'api_key: "[REDACTED]"')
    else
        sanitized_message = message
    end

    super(severity, sanitized_message, progname)
  end
end
```

and then you need to configure your application to use that logger:

```ruby
# in config/environments/production.rb or your other environments file.
config.logger = SanitizedLogger.new(config.paths["log"].first)
```

this custom logger class intercepts the log messages before they are written to the log file. it uses regex to find and replace sensitive information like "password" and "api_key", and replaces it with `[REDACTED]`. of course, you will have to expand this regex to account for other variable names that contain sensitive data. i've included examples for json like parameters and also for simple key/values. this makes the logs much safer, avoiding the need to change every job. please be careful when using regex because it can affect performance in very high throughput applications, but in most cases it should not be an issue. this also requires an application restart.

finally, if you have very specific argument formats or you want a more declarative way, you can use a decorator pattern. this method involves creating a separate class that wraps each of your jobs and sanitises the arguments. this is a bit more complex but makes the job code cleaner and is easier to maintain if you have many jobs to sanitise.

```ruby
class SanitizedJob
    def initialize(job)
        @job = job
    end
    def perform(*args)
        sanitized_args = args.map { |arg|
            case arg
            when Hash
                arg.transform_values {|v| v.is_a?(String) ? "[REDACTED]" : v }
            else
                arg
            end
        }
        Rails.logger.info "performing #{@job.class} with args: #{sanitized_args}"
        @job.perform(*args)
    end
end

class MyDelayedJob < Struct.new(:user_id, :email, :password)
  def perform
      Rails.logger.info "doing the email"
    # do the real work
  end
end


# somewhere you enqueue your jobs

wrapped_job = SanitizedJob.new(MyDelayedJob.new(1, "test@test.com", "secret"))
Delayed::Job.enqueue(wrapped_job)

# instead of
# Delayed::Job.enqueue(MyDelayedJob.new(1, "test@test.com", "secret"))
```

in this example we wrap the job class with the `SanitizedJob` class. this class performs the sanitisation of the args and then it calls the perform method of the original job. you don't have to change anything inside of your original job classes, so it's great for cleaner and maintainable code. in particular, you don't have to override classes. note that this example will not work out of the box if you have active job because active job does not send an instance of the job but an `ActiveJob::QueueAdapters::DelayedJobAdapter::JobWrapper` instead.

you have to choose what fits your context better, for simple needs overriding the `perform` is good, if you need more control the custom logger might fit, and for more complex problems and cleaner code, the decorator pattern is a more solid choice. it's a bit of a judgment call really, it all depends on what you're working on.

as for recommended readings, i found "effective ruby" by david a. black to be very helpful in understanding ruby internals and general coding best practices. another good resource is "rails anti patterns" by chad p. hietala; this book covers common mistakes, and pitfalls in rails applications and it was very useful during my years working with rails apps. also, going through the source code of gems like delayed\_job itself can be very informative, especially if you need to debug some internal problems.

one last funny thing i remember is this colleague i had, he once tried to sanitise the logs by base64 encoding the passwords, and then someone else had to come and decode all the logs to figure out what was going on. yeah, that didn't work out too well.

i hope these solutions and my experiences help you. let me know if you have any other questions. i’ve been there, and i’m happy to share what i’ve learned.
