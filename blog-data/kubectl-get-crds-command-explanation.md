---
title: "kubectl get crds command explanation?"
date: "2024-12-13"
id: "kubectl-get-crds-command-explanation"
---

so you're asking about `kubectl get crds` right I've been there trust me It's one of those things that seems simple enough on the surface but gets kinda deep the moment you start actually using it on a regular basis Let's break it down from my own experiences

First off `kubectl get crds` is your gateway into the world of Custom Resource Definitions in Kubernetes You know how Kubernetes has built-in resources like pods deployments services etc well CRDs let you extend that and define your own resource types It's like building your own Lego bricks for your Kubernetes cluster and the `kubectl get crds` command is how you see all those bricks you've built or are available

Think of it like this you've got a Kubernetes cluster right that's your playground Then you realize that the built-in resources aren't enough you need something that represents lets say a database instance or a message queue or maybe even a very specific application config Well that's where CRDs come into play

The `kubectl get crds` command itself simply lists all the CRDs that are currently installed in your cluster It doesn't interact with those resources it just shows you the definitions of those resources that have been configured for your cluster

When I started out I remember fumbling around and messing this up horribly I was trying to understand operators and Custom Resources and I was just completely confused about how CRDs were even related or if they were some weird kubernetes internal thing So yeah believe me I've been there and this command is your best friend in this journey

Now a quick look at the output you'll probably see something like this on your terminal

```
NAME                                          CREATED AT
applications.apps.example.com                    2023-10-26T14:35:20Z
databases.database.example.com                  2023-10-26T14:35:20Z
messagequeues.message.example.com               2023-10-26T14:35:20Z
```

The `NAME` column shows the fully qualified name of the CRD This is important because it’s structured like `<pluralized resource name>.<group>.<domain>` Notice how they’re plural thats key It's how Kubernetes identifies each CRD and this naming convention it's really useful to understand the connection between the CRD and the actual custom resources you'll create later

The `CREATED AT` column well it's pretty self-explanatory it’s the timestamp when that CRD was first added to your cluster This can be useful for debugging or just keeping track of when things were deployed I learned that the hard way because I made a lot of deployments and I didn't remember when I created what so this command really helped me

You might be thinking I have the list but what about the details Well you can get more info with a combination of the command and the `-o yaml` or `-o json` flags For example if you want to see the full definition of a CRD in YAML format you would use

```bash
kubectl get crds applications.apps.example.com -o yaml
```

This will give you a massive YAML output that details everything about the CRD from the schema to validation rules to even the spec and status fields The full definition is quite verbose and something you will want to become very familiar with If you want you can use `jq` to parse that output which is what I usually do If you’re not familiar with `jq` start learning it now it'll pay off trust me it’s like the swiss army knife for command line JSON processing

I once spent hours debugging a custom operator because I didn't double check the CRD definition and I had a mismatch in the schema between my operator and the CRD This resulted in my operator constantly trying to reconcile and failing miserably and it was all just a typo I had one field as string when it should've been an integer It was a painful experience so I can’t stress enough how important it is to check this definition

Now if you just want a quick overview of the schema without looking at all of that YAML you can use the following command with `jq` because life is too short to read massive blocks of yaml

```bash
kubectl get crds applications.apps.example.com -o json | jq '.spec.versions[].schema.openAPIV3Schema.properties'
```

This command pulls the JSON output and then filters through the structure to just display the properties defined in the schema of the CRD You’ll see that there's a nested structure but it gives you a quick overview of the spec of the resource you’ve defined It’s basically what’s available in your custom resources to use and it’s really helpful in the development lifecycle

Now I know what some of you are thinking what if you want to list CRDs based on a specific field well you can do that with some clever `jq` filtering for example lets say you want to find the name of the crd with a certain kind name well

```bash
kubectl get crds -o json | jq '.items[] | select(.spec.names.kind == "Application") | .metadata.name'
```

That command will output the name of the CRD where the `kind` name is equal to Application This is just a small example of what you can do and you can filter based on anything that's in the output of the `get crds` command

And just a quick little joke for the road why did the Kubernetes pod keep restarting? because it didn't understand the `restartPolicy: Always` flag so always check your yaml people haha  moving on

Now as for resources to learn more I would recommend not just relying on the Kubernetes documentation that can get a bit overwhelming sometimes I've personally found that "Kubernetes in Action" by Marko Luksa is a great book for understanding all the Kubernetes concepts and it does delve quite deep into Custom Resources and CRDs also the official kubernetes docs are pretty good but it can be overwhelming at first I'd recommend focusing on the "Custom Resources" part specifically which will get you what you are looking for Another good resource is the Kubernetes API reference this is important because it is very specific to the versions you use This reference is important as well because all the field names and the expected data type are in that reference So that would be my recommendations for you

So in summary `kubectl get crds` is your go-to command to explore the world of Custom Resource Definitions It's a fundamental command that you'll be using a lot as you dive deeper into extending Kubernetes and the examples here should be very helpful in your journey and it’s a common tool you'll be using so keep it handy
