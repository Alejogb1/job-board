---
title: "How do I create a funded account to use my local nearup docker container for development?"
date: "2024-12-14"
id: "how-do-i-create-a-funded-account-to-use-my-local-nearup-docker-container-for-development"
---

let's break this down, creating a funded account to link to your local docker dev setup, i've been there, done that, and got the t-shirt (a slightly faded one, i might add). it's a common hurdle, and it involves stitching together a few components but it is quite doable once you see the pattern. it usually boils down to these core areas: cloud provider account, api keys, proper docker network setup and finally, secure credential management.

first off, you will need a cloud provider. for illustration, we are going to use aws, but the principles are adaptable to google cloud or azure (or others). you will need to navigate to the aws console and create an account if you do not have one already. make sure you add a payment method for sure. then, inside of aws, you would usually search for "iam". this service will handle the identity and access management. basically, you'll create an 'iam user' with programmatic access. this is absolutely crucial, never use your root credentials directly. once you create the user, you should generate an access key and a secret access key. these are the keys you'll use from your docker container to authenticate with aws. keep these secret, like your old source code you are not proud of.

now, let's say you want to use aws s3 for storage, or aws lambda for some serverless function. usually, you will need to give your user specific permissions. in iam, we attach policies to users, so your new user will need an s3 policy to access it or an a lambda policy if you are planning to use lambda. this will require you to find out the correct permission structure. it can be a bit of trial and error at first, but generally, aws policies follow a json format.

i remember once, i was trying to get a lambda function to access s3, and i had the right keys but, i spent a whole day looking at "permission denied" errors. it turned out i was missing a specific s3:getobject permission. that day was educational for sure, a classic example of the devil is in the details. i think that experience alone probably increased my understanding of iam by about 200 percent.

now, onto the docker part of the equation. i assume you already have docker installed, and you can run containers locally. the idea now is to make your container aware of those aws credentials we created earlier. there are a few ways to do this. one way, which works for local testing, is to pass the credentials as environment variables when you run the container. here's an example docker run command:

```bash
docker run -d \
  -e aws_access_key_id=your_access_key_id \
  -e aws_secret_access_key=your_secret_access_key \
  -e aws_region=your_aws_region \
  --name my-app \
  my-docker-image
```

obviously, you have to replace `your_access_key_id`, `your_secret_access_key` and `your_aws_region` with your actual credentials and region. you will also have to replace `my-app` with a name you prefer for your container. the same for `my-docker-image` the image name you are using. now, inside the container, your application can access these environment variables and use them to authenticate with aws.

i've found that using environment variables this way is ok for local testing, but you should avoid doing it in production or even in development when collaborating with others because it can lead to a lot of security issues. instead, a better way is to use aws profiles or roles, but it requires a little more setup outside of the docker container scope.

another common approach is to use aws cli configured within the container. this means you would need to install the aws cli inside the docker image and then use the `aws configure` command with your credentials. then, any application that uses the aws sdk will automatically pick up those credentials. it's not hard but it does add some extra work inside your docker build process.

here's a sample dockerfile snippet which adds the aws cli inside the container:

```dockerfile
from python:3.9-slim-buster

run apt-get update && apt-get install -y python3-pip
run pip install awscli

copy requirements.txt .
run pip install -r requirements.txt

copy . .

# configure aws in dockerfile is not good practice, better to configure from outside the container using aws profiles
# or through roles when possible
#run aws configure set aws_access_key_id your_access_key_id
#run aws configure set aws_secret_access_key your_secret_access_key
#run aws configure set aws_region your_aws_region

cmd ["python", "your_app_entry_point.py"]
```

again, note the commented-out `aws configure` command. that's because putting credentials directly in the dockerfile is a bad practice, and it can lead to security breaches. also, note that we are assuming we are using python here, but you can adapt it to your own needs.

now, let's focus on using roles. this is what i recommend for development environments. this involves assigning an iam role to the ecs (elastic container service) task, or ec2 instance, where your docker container will run. the magic here is that aws takes care of passing those credentials to your application at runtime without needing any access keys or secret keys. when you deploy to ecs or ec2, you configure them to use an iam role, and then you dont have to provide anything else in the container for authentication. your application simply asks aws for temporary credentials and it gets them automatically.

it's like having a waiter bringing you what you need, instead of having to go to the kitchen with a map. if you want to use docker for local development, you can use a tool like aws session manager to start a shell inside an ec2 instance. then run docker commands inside it using a role instead of user based keys. this is very convenient once it is setup.

using roles requires that you are launching the docker container in aws services like ecs or ec2. for local development you might not have that, but you can mock it with a temporary aws profile configuration that matches the role you are targeting.

this method needs you to have the aws cli on your machine already. so you have to install that on your local environment. usually with something like `pip install awscli`. and then you will have to execute on your machine this: `aws configure --profile <name-of-your-profile>` and add the key and secret for the user that matches the role you are planning to use in ecs.

here's an example of how your application code will usually look like:

```python
import boto3

# the code checks if any role or profile is provided, and it will try to use it. otherwise it uses user credentials
session = boto3.session.Session()
s3 = session.resource('s3')

try:
    bucket = s3.Bucket('your-bucket-name')
    print(f"bucket '{bucket.name}' found ok")
    # perform further operations on the bucket
except Exception as e:
    print(f"error: {e}")
```

i like how simple this becomes with proper role management, the code does not even need any keys at all.

regarding documentation, i highly recommend reading the aws documentation on iam and roles. there are also quite a few good books on aws architecture which will teach you how to use iam properly. a good starting point could be "aws certified solutions architect official study guide". the official aws documentation is usually very good, even though sometimes it can be a little too dense and a bit too "marketing-y".

also, i would suggest to research different docker container orchestration options. if you're only using docker locally, you don't need to learn kubernetes right away. but if you plan to scale up the number of containers, tools like docker compose, docker swarm, or kubernetes will eventually be necessary. once you grasp the basics of docker, you can then move on to these more complex tools that will help you automate the process of starting and scaling containers.

i'm not gonna lie, setting all this up can feel a bit complicated the first time. especially getting the permissions of iam correct. it's an exercise in patience and debugging and it is quite normal to stumble along the way. there is a learning curve, but it is one worth tackling because once you grasp it, it will improve your development practices. sometimes i think this is why most of the jokes are related to debugging. but don't worry, you'll get there. if i can do it, anyone can.
