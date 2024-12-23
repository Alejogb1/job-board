---
title: "ingress yaml template returns error in renderring nil pointer evaluating int?"
date: "2024-12-13"
id: "ingress-yaml-template-returns-error-in-renderring-nil-pointer-evaluating-int"
---

 so you're hitting the classic "nil pointer evaluating int" snag in your Kubernetes ingress YAML template rendering sounds familiar let me tell you

I've been wrestling with YAML for way too long I swear its like its always plotting against us just waiting for one little mistake it seems like it's specifically designed to give programmers existential dread when something like this shows up so here's the breakdown from someone who's been in the trenches with this and yes I've seen this exact error way too many times its the kind of error that wakes you up at 3 am

First thing's first when you see "nil pointer evaluating int" in a templating context like with Helm or even just a basic templating engine it screams at you "I tried to access a value that isn't there it's missing buddy" and the thing is when we are dealing with Kubernetes ingress YAML files they can get complex real fast we have multiple nested objects lists loops conditional logic and it's a mess

This usually means you're trying to do something with a variable that you think is an integer but at that moment of evaluation it’s actually nil or null the templating engine tries to treat it like a number but nope there's nothing there and thats when the fun starts

Now the specific location of your error is important it's most likely in a place where you are trying to use an integer value inside the ingress spec things like resource limits port numbers maybe even something more obscure It's in the places we expect numbers to be you know when we expect numbers to behave like they should which they dont

Lets start with the usual suspects and the things I've run into personally. For me this was in a templated ingress that was part of a Helm chart I was responsible for. We were dynamically setting the service port based on a configuration value and turns out the configuration value wasn't always set or if it was there it could be invalid because it came from an external API (a real fun experience of "oh what's this data format now" sort of thing). I thought everything was hunky dory but nope Kubernetes said otherwise I think my blood pressure went through the roof that week

So how do you troubleshoot this kind of thing first things first print debug you want to see what the heck is going on with the variable. If you are using Helm you can use the `-debug` flag and examine the resulting rendered yaml or even you can set debug variables using --set something like debug=true and then your template can render a specific value in the yaml conditionally you would need to work this around your templates and it's a bit annoying I know. If you are using simple templating you can just print to the standard output if you are executing on the command line

Here's a general way you can check what's going on:

```yaml
# Example snippet of Helm chart values.yaml
someConfig:
  myPort: 8080  # This could be coming from anywhere in reality
someOtherConfig:
    debug: false # debugging flag
```

```yaml
# Example snippet from the template that's causing trouble ingress.yaml.
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: {{ if .Values.someOtherConfig.debug}}{{.Values.someConfig.myPort | default "port not provided"}}{{else}}{{.Values.someConfig.myPort | int }}{{end}}
```

In the above example if you are not in debug mode, which is default we are trying to convert the value to an integer using `int` template function and if that value is nil or not an integer the templating engine will throw the nil pointer error and it will fail to render. If debug flag is set to true it will render the value with no int conversion or at least show "port not provided" if it is missing

Another common problem is when you use list elements to index values like this and the list is empty or shorter than what you are expecting :

```yaml
# Example snippet of Helm chart values.yaml
servicePorts:
 - 8080
 - 8081
 - 8082

someOtherConfig:
    debug: false

```

```yaml
# Example snippet from the template that's causing trouble ingress.yaml.
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: {{ if .Values.someOtherConfig.debug}}{{.Values.servicePorts | index 1 | default "index missing"}}{{else}}{{.Values.servicePorts | index 1 | int }}{{end}}
```

In the above example we are trying to access the second element of a list if the debug flag is false we are also trying to convert it into an integer. If the list is empty or only contains one element this will trigger a nil pointer error because the index doesn't exist. The debugging conditional and the default value helps a lot

Finally I've seen cases where nested objects in values files are not consistently available this gets a bit more complicated because then you need to check not only if values are there but also if all the nested objects exist something like this:

```yaml
# Example snippet of Helm chart values.yaml
serviceConfig:
  backend:
    ports:
      http: 80
      https: 443
someOtherConfig:
    debug: false
```

```yaml
# Example snippet from the template that's causing trouble ingress.yaml.
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: my-service
            port:
              number: {{ if .Values.someOtherConfig.debug}}{{.Values.serviceConfig.backend.ports.http | default "ports missing"}}{{else}}{{.Values.serviceConfig.backend.ports.http | int }}{{end}}
```

Here we are trying to access a nested object which might not exist the "backend" object or the "ports" object or even the "http" key could be missing and if they are missing we will have a nil pointer error. The default value helps if you want to debug or you can simply check if the object exists before you try to access the value but at that point your template becomes complex.
Remember even if the http key is available but if its value is something that's not an integer such as an string this will also throw an error in case debug flag is not set

Now when it comes to solving this "nil pointer evaluating int" error the core problem is missing values but the solutions are varied depending on your use case and template complexity. One important rule is that you should always set default values for most of your template variables. Another very useful technique is to have debug flags and you can conditionally print all sorts of information from your value files. Also it is useful to print information in the template itself using conditional logic this will allow you to debug faster when you hit a similar wall again and of course use the correct data types this is a bit annoying I know.

Now that was a lot and I feel like a robot but I've seen this kind of error so much that I think I started understanding the machine's brain a little bit. Here's a bad joke that I've heard, a developer once asked a computer "Why are you so bad at YAML?" and the computer said "because I have no pointer". I know it's terrible.

Anyway for solid resources to make your life easier especially with Kubernetes and YAML. I would suggest checking out "Kubernetes in Action" by Marko Lukša. It gives a really detailed look at how Kubernetes works with YAML configurations I can't recommend enough. For a more general view on templating and dealing with missing data "The Pragmatic Programmer" by Andrew Hunt and David Thomas is a must-read. Even though it doesn't specifically cover YAML templating concepts like defensive programming and error handling it can make your coding skills much better and more robust even on template configurations.

Keep an eye on your values and stay away from those nil pointers its a dark place I promise
