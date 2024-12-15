---
title: "How to debug a Node app in IntelliJ with an AWS profile?"
date: "2024-12-15"
id: "how-to-debug-a-node-app-in-intellij-with-an-aws-profile"
---

well, alright then, let's talk about debugging node apps in intellij while juggling aws profiles. i've been down this road more times than i care to remember, and it can definitely feel like you're chasing a ghost at times. it's one of those things that seems straightforward on paper, but the devil is in the details of configuration, environment variables, and aws sdk quirks. so, i will just give you my personal experience and a way i made it work, and you may find it useful.

first off, the core problem usually boils down to this: your intellij debugger needs to be pointed at the correct process, and that process needs to have the correct aws credentials to interact with aws services. pretty simple stated, not so simple in practice. when you're running locally, typically, you would be relying on your default aws profile, but when debugging a specific app running within a process spawned by intellij, things get a little bit complicated. you have to make sure the correct aws profile is loaded by the app that is being debugged.

the most common pitfall i’ve seen, and i’ve definitely tripped over this myself, is forgetting that the environment variables accessible to intellij when running in debug mode might not be the same as what's active in your terminal or what you expect. so you are running the node app from intellij, but the aws sdk in your node app is not able to find your credentials from the aws cli configuration because the environment is not setup correctly, which obviously will throw aws authentication errors. the aws sdk will then try to use iam roles associated to the machine it is running, and if you are not running from an aws ec2 instance you will be having a big issue.

i remember one time, i spent nearly a whole afternoon trying to debug a lambda function locally using `serverless offline` and intellij, and my aws calls were failing with credential errors. i was so frustrated that i almost threw my keyboard to the wall. turned out i had configured the profile in my `.aws/credentials` file, but i had not set the `aws_profile` environment variable in the intellij run configuration. that simple environment variable can save you from pulling your hair out.

so here's the approach i've found works well, step by step:

**1. ensure your aws cli is configured correctly.**

this part seems obvious, but double-checking is a good habit to get into. make sure you have the aws cli installed and configured with the profile you intend to use for debugging. you can verify this by running aws `sts get-caller-identity` from your terminal. if it works there, then the problem is not there. you probably know this but it is so easy to oversee this.

**2. create an intellij run configuration for your node app.**

in intellij, go to *run* > *edit configurations*. add a new *node.js* configuration. configure the `javascript file` path to your main entry point of the app, the `working directory` should be where your `package.json` is located, and any required parameters in the `application parameters` input. at this point you would test it by clicking run, it should start your application. if it does not work, fix the problems you find before trying to debug.

**3. set the aws profile as an environment variable in your intellij run configuration.**

here's where the magic happens. under the *environment variables* section of your intellij run configuration, add a new variable:
  *   name: `aws_profile`
  *   value: the name of your aws profile, e.g., `my-dev-profile`.

this environment variable will make sure that your node application knows which profile to use from your `~/.aws/credentials` file. this avoids the common problem of the aws sdk trying to fallback to default profiles or not finding any credentials at all.

**4. configure debugging in your node.js app**.

make sure you are starting your node app with debugging enabled. for example, you can do it like that:

```javascript
   // package.json scripts section
   "scripts": {
        "start": "node --inspect=9229 app.js",
        "start:debug": "node --inspect=9229 --nolazy app.js"
      }
```

then in your intellij run config at the `before launch` section, add a new `npm` action with a command named `start:debug`. this should start your application with debug mode enabled.

**5. set your breakpoints and debug**.

finally, go back to your intellij run configurations, select the node app and then click the debug button. that will start your node app, and also the intellij debugger, and you will be able to inspect all your code.

**code snippets:**

here’s a simple example of how you might configure your `package.json` to enable debugging:

```json
{
  "name": "my-node-app",
  "version": "1.0.0",
  "description": "a simple node app",
  "main": "app.js",
  "scripts": {
    "start": "node app.js",
    "start:debug": "node --inspect=9229 --nolazy app.js"
  },
    "dependencies": {
    "aws-sdk": "^2.1561.0"
  }
}
```

and here’s a basic `app.js` file which uses the `aws sdk`. please make sure you install the aws-sdk dependency with: `npm i aws-sdk`:

```javascript
const AWS = require('aws-sdk');

const s3 = new AWS.S3();

async function listBuckets() {
  try {
    const data = await s3.listBuckets().promise();
    console.log("buckets list:", data.Buckets);
  } catch (err) {
    console.error("error listing buckets:", err);
  }
}

listBuckets();
```

remember to set your `aws_profile` in the intellij run configuration environment variables to the correct name of the profile you want to use.

finally here is how you configure the `before launch` in the intellij node app run config.

```json
   "beforeLaunchTask": {
       "npm command": {
          "command": "start:debug"
        }
    }
```

this configuration will start your node app and the debugger will attach to the process correctly.

**troubleshooting:**

if you are still having problems you could try the following:

*   **check your aws credentials path**: the aws sdk looks for the `.aws` directory in your home directory (`~/.aws`). if the directory is not there, or the `credentials` file is not there, it cannot read your credentials, so double check that the directory and file are in the right place.
*   **double check the aws profile name**: ensure the profile you set in your intellij run configuration environment variable matches exactly the name in your aws configuration file. small typos can give you a headache.
*   **use `console.log`**: if all else fails, sprinkle `console.log` statements in your node app to verify the value of your environment variables, and to check that your code is actually getting called and the aws sdk is actually trying to make calls. print the output of the  `process.env.aws_profile` to see which profile the application is trying to use.
*   **check that your debug port is not being used by another application**: the default debug port in node is 9229, if it is being used by another application, you should use a different port. you can change the port in the `scripts` section of your `package.json`. and set the corresponding port in the intellij run configurations.
*   **ensure your aws sdk is up to date**: sometimes older versions of the aws sdk have issues with finding the aws profile using environment variables. you could try updating your aws sdk using `npm i aws-sdk@latest`.

**resources:**

rather than just linking to websites, i’d recommend a couple of books that have been very valuable for me in understanding these underlying mechanisms:

*   “programming aws sdk for javascript” by jesse wolgamott: this book goes in-depth on the aws sdk and how it handles authentication and configuration. a highly recommended read.
*   “node.js design patterns” by mario casciaro and luciano mammino: understanding how node apps work under the hood will help you debug much faster. especially if you are using asyn/await, promises or workers.

debugging can be a painful process, but with the proper setup and understanding of how the pieces fit together, you'll become much more efficient at it. i've spent many hours in front of my screen cursing at error messages, so take it from me, always remember to verify all the assumptions, and to log everything that you can. good luck, and let me know if there's anything else i can help with. hopefully, this will help you from not smashing your keyboard.
