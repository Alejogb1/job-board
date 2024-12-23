---
title: "How can Grails 4.0.3 application.yml files reference config values in separate files?"
date: "2024-12-23"
id: "how-can-grails-403-applicationyml-files-reference-config-values-in-separate-files"
---

,  I've seen this come up countless times, and, frankly, it's one of those things that can trip up even experienced Grails developers if they’re not careful. The situation is this: you want your `application.yml` in Grails 4.0.3 (and frankly, the solutions are largely similar across Grails 3 and 5 as well) to pull configuration values from other files. This might be for different environments, for secret keys you don't want directly in the main config, or even just for better organization. The good news is, it's entirely achievable, although not through direct yaml referencing like you might hope. Instead, we leverage Spring Boot's powerful configuration system which underpins Grails.

My past experiences have involved projects where a single monolithic `application.yml` simply became unwieldy. We had different database settings, API keys, and feature flags that varied wildly between development, staging, and production. Keeping it all straight, secure, and manageable within one file was a nightmare. That's when I started looking at externalizing configuration, and specifically, utilizing the `spring.config.location` approach, combined with placeholders.

The fundamental concept here is that Grails uses Spring's property resolution mechanisms. Spring looks in specific locations and in a specific order for configuration properties. The `application.yml` file is just one such source. The key is understanding how to introduce additional sources and have them be part of this resolution process.

Let's start with the primary strategy: using the `spring.config.location` property. This property allows you to point to other configuration files or folders, which Spring will then load into its property resolution system. Critically, this configuration needs to be provided to the application *before* the core `application.yml` is parsed, typically via environment variables or system properties. I frequently default to environment variables because they offer flexibility without altering the core `application.yml` itself.

Here's how this works in practice. Assume I want to maintain a separate file for development configurations called `dev-config.yml`. This could live in the same directory or a relative location like `/config`.

First, start by setting the environment variable `SPRING_CONFIG_LOCATION`. In a bash environment, this is `export SPRING_CONFIG_LOCATION=file:./config/dev-config.yml`.

Inside `dev-config.yml`, I would have something like:

```yaml
my:
  api:
    key: "development_api_key"
  database:
    url: "jdbc:h2:mem:devdb"
```

And then, your main `application.yml` file might look like this:

```yaml
---
environments:
    development:
        my:
          api:
              key: "${my.api.key}"
          database:
              url: "${my.database.url}"
    production:
        my:
          api:
             key: "production_api_key"
          database:
             url: "jdbc:postgresql://...."
```

Notice that within `application.yml`, the values are not static, they are referencing the placeholders `${my.api.key}` and `${my.database.url}`. When the application starts up, Spring first loads the properties found in `dev-config.yml`, as dictated by `SPRING_CONFIG_LOCATION`, and then it loads properties from `application.yml`. During this process, when it encounters `${my.api.key}`, it resolves the value by checking its resolved properties. If `dev-config.yml` was read first, then the placeholder is populated with `development_api_key`. If the environment variable wasn't set, it would resolve to the value defined directly in the production environment, which it loads by default, meaning it loads both configurations but prioritizes one over the other if keys match.

Let's look at another method: using folders for configuration. Instead of a single file, you can point `spring.config.location` to a directory and have Spring pull all the configurations from files within that directory. I often use this for separating different module configuration into their own files.

Suppose I created a directory `/config/env-config/` and placed files `dev.yml`, and `prod.yml` inside it. Then set your environment variable to `export SPRING_CONFIG_LOCATION=file:./config/env-config/`. In `dev.yml` we might have:

```yaml
my:
  feature:
    flag: true
```

And `prod.yml` contains:

```yaml
my:
  feature:
    flag: false
```

Then the `application.yml` might include the placeholder:

```yaml
environments:
    development:
        my:
            feature:
                flag: "${my.feature.flag}"
    production:
        my:
            feature:
                flag: "${my.feature.flag}"
```

When the application starts in a development environment, it'll load configurations from every `yml` file within the `env-config/` folder, meaning it would load both `dev.yml` and `prod.yml`. Spring uses the last-wins approach, so if it encounters duplicate keys, the last loaded configuration will take precedence. In this case, since Spring loads files in alphanumeric order (unless specified otherwise), `dev.yml` will take precedence. During runtime, it prioritizes the properties based on profiles specified in the environment or during startup.

Now, for a more granular approach, consider profiles, this builds upon the previous solutions by making environments more dynamic. In your `application.yml` you would have separate `environments`.

```yaml
---
spring:
  profiles:
    active: dev

my:
    app:
      message: "Default message"

---
spring:
  profiles: dev
my:
    app:
        message: "Dev specific message"
---
spring:
  profiles: prod
my:
    app:
        message: "Prod specific message"
```

Here, the `spring.profiles.active: dev` activates the 'dev' profile. If you ran this in a development environment, it would override the default `my.app.message` with the value from the `dev` profile: 'Dev specific message'. If the active profile wasn't set, the default one would be loaded ('Default message').

To use external configurations with profiles, you can combine this technique with the previously discussed `spring.config.location`. If you were to run with `export SPRING_CONFIG_LOCATION=file:./config/profile-config/`, the files within `profile-config` directory will also be parsed. Within this directory create two files, `dev.yml` and `prod.yml`. In each of these, define the same configuration key. For `dev.yml`:

```yaml
my:
    app:
        message: "Ext dev message"
```

And for `prod.yml`:

```yaml
my:
    app:
        message: "Ext prod message"
```

Then, if your application is active with the `dev` profile it will override the previously defined profile key with the one loaded from the external `dev.yml`.

To deepen your understanding of this and more intricate Spring Boot configurations I suggest taking a look at "Spring Boot in Action" by Craig Walls. For a more exhaustive approach to the underlying Spring framework, "Pro Spring" by Chris Schaefer et al. is an excellent resource.

In summary, avoid placing all configurations in one gigantic `application.yml` file. Instead, leverage Spring's property resolution, set `SPRING_CONFIG_LOCATION`, and make good use of profiles. This keeps your configuration manageable, environment-aware, and, crucially, less error-prone.
