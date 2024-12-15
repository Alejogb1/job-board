---
title: "Why is Heroku for Grails 5?"
date: "2024-12-15"
id: "why-is-heroku-for-grails-5"
---

so, you're asking about heroku and grails 5, specifically why someone might use heroku for grails 5 deployments. i get it. been there, done that, got the t-shirt (and probably a few debugging scars to go with it).

look, heroku, it’s not like this mythical unicorn of cloud platforms. it's a platform-as-a-service, or paas, and it comes with pros and cons like everything else. for grails 5, i’ve seen people use it for a few key reasons. i've also had my share of head-scratching moments figuring out how to make them play nicely together, so let me spill the beans based on my personal experience.

first, the big draw is simplicity, seriously. when you’re in the trenches building a grails app, the last thing you want is to become a devops guru overnight. heroku basically takes care of a lot of that server stuff. think of it this way: you write the grails code, you set up a `procfile`, and you push your code. heroku handles the build, the containerization, the routing, and the infrastructure stuff that you would rather not touch. it's pretty compelling compared to setting up an entire aws ec2 instance and configuring the web servers yourself. i used to do that back in the '00s – it's really not something i miss. i remember spending a whole week just getting tomcat to work just so. that's a week i'll never get back.

second, it’s pretty friendly for experimentation. if you're just starting a project or prototyping something new, heroku is a good choice because it’s fast to get going. you don't have to worry about allocating a virtual server or configuring a database yourself initially. you can quickly create a heroku app, hook it up to a database, and get your grails app deployed without too much hassle. in my early days with grails, i probably deployed dozens of small projects to heroku for testing and demos before moving to something more robust. i once even created a weird api to fetch cat pictures from the net for training purposes and deployed it to a test app on heroku to learn about the heroku api itself.

now, there are also a few bits to consider when deploying a grails 5 application. you'll have to think about your `procfile`. you need to specify how heroku should run your application. for a typical grails application, it's going to look similar to this:

```
web: java $JAVA_OPTS -jar build/libs/*.jar
```
this simply tells heroku to execute the built jar file for our grails app using the java command. nothing super complicated. the `$JAVA_OPTS` there are useful for configuring heap size and other settings.

but, there's more. database setup. usually, you will use something like heroku postgres. this means you'll want to adjust your grails application to work with heroku's specific database configurations. you can do it in the `application.yml` (or `application.groovy` if you're old-school). here’s how i usually approach it:

```yaml
dataSource:
  pooled: true
  jmxExport: true
  driverClassName: org.postgresql.Driver
  url: ${DATABASE_URL}
  username: ${DATABASE_USER}
  password: ${DATABASE_PASSWORD}
  properties:
    autoReconnect: true
    validationQuery: 'select 1;'
    maxLifetime: 10000
```

the key part here are the `${DATABASE_URL}`, `${DATABASE_USER}`, and `${DATABASE_PASSWORD}`. heroku provides these as environment variables, and this setup makes the grails app fetch these credentials at runtime. it's less clunky than hardcoding the credentials inside your application and keeps things secure-ish.

if you need additional libraries or plugins in grails, heroku also lets you use buildpacks. these buildpacks are essentially scripts which execute during deployment and can help you add support for other tech such as nodejs, or python for any other reason. for example, if you need to compile assets like javascript, then you can use the nodejs buildpack. here's an example of how to combine the java and nodejs buildpacks:

```
buildpacks:
  - heroku/nodejs
  - heroku/java
```

these must be listed in the `app.json` file of your application. you can then make sure the build process runs the necessary node commands to generate all the needed assets in your grails project.

as for downsides, heroku can get a bit pricey as your app grows. scaling vertically is pretty straightforward, but it has its limits. if you need more control and resources, you could migrate to a different solution, such as running on kubernetes with a service like aws eks, but that requires a lot more setup. that's something you consider when the budget allows it. also, if you have a highly customized infrastructure requirement, heroku may not be the best choice, as it abstracts you away from the underlying machine. there also can be some vendor lock-in; migrating your infrastructure away from heroku is not that simple. also, be mindful of the ephemeral file system. that's something that i once tripped over. i had an application storing temporary data and it disappeared when the dyno restarted. learned that the hard way.

one more thing people may stumble upon, the logs, initially i had issues understanding the heroku logging system and integrating them with other log systems. you can use heroku's cli to view the logs, but once you have a large number of dynos, you need a log management solution. and that's where services such as datadog, splunk, or logentries come in handy, so be mindful of those.

in conclusion, heroku is a solid choice for grails 5, especially for those starting out, prototyping, or teams that want a less operations-heavy workflow. it is simple, and straightforward, it is pretty good for the common use case of deploying a basic grails application. but like any platform, it’s essential to understand its limitations and plan accordingly. i do have a bad memory regarding one of the times my application failed, the server was probably having a bad monday.

if you're looking for more resources, i'd recommend checking out "continuous delivery" by jeszczyk, and humbles, its a classic on devops concepts and "the twelve-factor app" from heroku, which gives good recommendations about how to build modern web applications. i'd also look into the official heroku documentation since it is very detailed and usually has everything that you need in it. these resources are very helpful to understand why heroku does things in a certain way and how to make the most out of it. i hope this helps and good luck with your grails application.
