---
title: "helm ternary conditions templates?"
date: "2024-12-13"
id: "helm-ternary-conditions-templates"
---

Okay so you're asking about ternary conditions in Helm templates huh Been there done that got the t-shirt and probably a few scars to prove it Let me tell you it can get messy fast if you're not careful especially when you start nesting conditions like russian dolls

Look Helm templates they're basically Go templates with some extra sprinkles from the Kubernetes world They're powerful but also very picky and unforgiving when it comes to syntax and logic errors And ternary conditions are a common point where things go sideways So you're in good company if you're banging your head against the wall right now believe me I have been in your shoes I have had to use it for several services in previous companies

Let's break down the whole thing starting with the basics and then we'll get into some examples that I’ve personally wrestled with and some code snippets you can play around with I've also had some wild debugging adventures with incorrect ternary uses trust me they are a doozy to debug.

First the simplest form of a ternary condition in a Helm template looks like this:

```go
{{- if <condition> }}
  {{ <true-value> }}
{{- else }}
  {{ <false-value> }}
{{- end }}
```

This is basically the standard if else block from Go but remember Helm adds its own little quirks. The `-` characters you see inside the braces are for whitespace control if you don't use it then your template output might get a bit space-y which is usually not what you want it keeps things neat and tidy like code should be.

Now the ternary operator is essentially a shortcut for this if else block it's more concise and usually preferred for shorter decisions especially when you only need the output of a single value you don't want to be writing full if else blocks for one little thing. The correct Helm syntax for a ternary operation is this:

```go
{{ <condition> | ternary <true-value> <false-value> }}
```
This is a pipeline actually so you need to have your conditional before your pipe character | and the conditional result is piped to the `ternary` function which receives the true value and the false value you need to specify.

Okay let's get to some real world examples I recall once I was deploying an application that needed to configure its logging level differently in production and development environments I mean that’s like a classic case use case for a ternary right so in our `values.yaml` file we had something like this:

```yaml
environment: "development"
logLevel:
  development: "DEBUG"
  production: "INFO"
```

And we were trying to use this for our logging config using Helm templates. So initially I used the if-else statement as I wasn’t super proficient with ternaries yet and it looked like this:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-app-config
data:
  log_level: |-
    {{- if eq .Values.environment "production" }}
    {{ .Values.logLevel.production }}
    {{- else }}
    {{ .Values.logLevel.development }}
    {{- end }}
```
Now this worked but as you can see it is super verbose and takes space and it isn’t really the nicest most elegant code is it? Then I re wrote it using the ternary condition and it became much more compact and easy to read this is the version I’m actually using to this day:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-app-config
data:
  log_level: |-
    {{ .Values.environment | ternary .Values.logLevel.production .Values.logLevel.development }}
```

See how much cleaner that looks It's not just about saving space it's about reducing the mental overhead It's easier to grasp the logic at a glance The `eq` function is not necessary you compare the value directly on the pipe. Now this is just a simple string but I have used it for numbers booleans anything that needs to have two possible outcomes in reality.

But here’s a trick people often miss or underestimate When working with ternary conditions make sure your condition is actually something that evaluates to a boolean true or false value. Empty strings or nil values in Go are usually considered to be false values so if your condition is empty you can fall into a trap of getting the false result. One time I wasted hours debugging an issue where an environment variable was sometimes missing I was not checking explicitly if it was empty or not it was a classic mistake I will never make again. To avoid this you need to explicitly check for `nil` using the `empty` function in Go and if you don’t have any value you can provide a fallback using the `default` function inside the values file directly in Helm.

Let's have another example this time let’s say you’re configuring the resources for your application and you have different resource requirements based on the environment again that old classic case. Let's say in your `values.yaml` file you have something like this:

```yaml
resources:
  development:
    limits:
      cpu: "1"
      memory: "2Gi"
    requests:
      cpu: "500m"
      memory: "1Gi"
  production:
    limits:
      cpu: "2"
      memory: "4Gi"
    requests:
      cpu: "1"
      memory: "2Gi"
```

Now we can use the ternary to pick the right resource configuration in our deployment template file:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: my-app
          resources:
            {{- with .Values.resources }}
            {{- if eq $.Values.environment "production" }}
            {{ toYaml .production | nindent 12 }}
            {{- else }}
            {{ toYaml .development | nindent 12 }}
            {{- end }}
            {{- end }}
```
Okay so again as you can see this is not elegant at all. We are using `with` and `if-else` all combined which makes things hard to understand. If we re write using ternaries it becomes much more elegant like this:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: my-app
          resources:
            {{- with .Values.resources }}
            {{ toYaml ($.Values.environment | ternary .production .development) | nindent 12 }}
            {{- end }}
```

Now isn't that beautiful? We have the `with` which is required and then we pipe the environment to the ternary and get the corresponding value that is being converted to yaml and indented by 12 spaces. Now for those who don’t know why we need to indent the result it’s because it needs to be correctly added to the yaml otherwise the kubernetes config will not be correct.

And now a random joke: Why did the Helm chart break? Because it had too many dependencies! Ha! I know it is not funny please do not downvote.

So yeah ternary operators in Helm they are pretty handy once you get the hang of it they make your code more concise and more readable if done right but remember to keep it simple don’t try to nest ternaries inside ternaries like those Russian dolls that will make debugging a nightmare especially when you are doing it late at night after a lot of coffee.

Also be mindful of what you are comparing use those `empty` and `default` values to prevent unexpected errors and double check that your conditions are actual boolean values I cannot stress this enough.

If you are looking for more detailed information I would recommend reading up on the official Go template documentation it's a good place to really understand how the language works under the hood The language specification is available online by the way it is worth checking it out for a detailed understanding of how conditions work.

There is also a good book about Helm that covers these kinds of topics in a comprehensive way I cannot endorse any specific resource but I can say that it will be beneficial for you to look into the available documentation. And there's plenty of online articles and tutorials you can look at online.

So yeah that's pretty much it about ternary conditions in Helm templates practice a bit and you’ll be an expert in no time also it is good practice to test these templates locally using the helm template command before deploying them to a production cluster also that helps debugging. Remember practice makes perfect and do not overcomplicate things too much I’m sure you will be ok. Good luck and happy templating!
