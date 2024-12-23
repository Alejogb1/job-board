---
title: "How can Heroku containers be released using the Platform API?"
date: "2024-12-23"
id: "how-can-heroku-containers-be-released-using-the-platform-api"
---

Alright, let’s talk about deploying to Heroku using their Platform API; it's a process I've dealt with extensively over the years, often automating complex deployment workflows for various applications. Instead of relying solely on the Heroku CLI, leveraging the Platform API offers much finer control and integrates beautifully into CI/CD pipelines. It's a powerful tool, but it does require a solid understanding of its mechanics. So, let's dive in.

Essentially, the Heroku Platform API allows you to programmatically interact with your Heroku applications. This includes creating apps, managing resources, and, crucially for this discussion, releasing new code versions—or *slugs*, as Heroku calls them—to your containers, referred to as dynos. Doing this effectively involves several distinct steps, each corresponding to different API endpoints. My experience has taught me that a methodical approach, breaking down the process into manageable chunks, makes the complexity quite approachable.

The first step, and frequently overlooked, is ensuring you have proper authentication set up. This means you'll need an API key, obtainable from your Heroku account settings, and proper configuration within your environment. I've seen countless projects stumble right here, often due to misplaced credentials. After that, the deployment flow can generally be broken down into three core parts: preparing your code for upload, creating a release using a slug, and finally monitoring the release to ensure it completes successfully.

Let’s start with preparing the code for upload. Heroku doesn’t accept raw code directly; it needs a *slug*. A slug is essentially a compressed archive containing your application’s code and dependencies, bundled with a runtime environment suitable for Heroku. While Heroku's buildpack system often handles this process for you, when we're working directly with the Platform API, you're responsible for preparing this yourself. I typically use tools like `tar` and `gzip` for this. You must create a gzipped tar archive of your project, ensuring that the archive's root level contains your application's source code, usually the main folder that contains files like your `requirements.txt`, `package.json` or other dependency specification files.

Here’s a conceptual code example of creating that archive, using `bash`:

```bash
#!/bin/bash
# Assumes you are in your project directory
TAR_FILE="source.tar.gz"
tar -czf "$TAR_FILE" *
echo "Slug archive created at: $TAR_FILE"
```

This script, which I’ve used countless times with only minor variations, assumes you are already in your project's root directory and that it will bundle everything. For complex projects, a more selective inclusion using include and exclude patterns in the tar command is often necessary. I recommend looking into the `tar` manual page using `man tar` for specific filtering options.

Next, you’ll need to create a new *source* resource via the Platform API; this is where you upload the generated slug archive, making it available to Heroku. The following snippet demonstrates uploading the slug to the source endpoint. This uses `curl` and assumes that you have your Heroku API key stored in an environment variable named `HEROKU_API_KEY`. You'll also need your Heroku application id, or your app's name, in an env var `HEROKU_APP_ID` or `HEROKU_APP_NAME` respectively. I’ve often used these shell scripts in conjunction with CI/CD pipelines for automatic releases.

```bash
#!/bin/bash

HEROKU_API_KEY=${HEROKU_API_KEY}
HEROKU_APP_NAME=${HEROKU_APP_NAME}
TAR_FILE="source.tar.gz"
# create a source URL
source_url=$(curl -s -n -X POST \
 -H "Accept: application/vnd.heroku+json; version=3" \
 -H "Authorization: Bearer $HEROKU_API_KEY" \
 https://api.heroku.com/apps/$HEROKU_APP_NAME/sources | jq -r '.source_blob.put_url')
echo "Source URL: $source_url"

# upload the slug
curl -s -v -H "Content-Type:" --upload-file "$TAR_FILE" "$source_url"

# get the source blob id for the release
source_id=$(curl -s -n -X GET \
 -H "Accept: application/vnd.heroku+json; version=3" \
 -H "Authorization: Bearer $HEROKU_API_KEY" \
 "https://api.heroku.com/apps/$HEROKU_APP_NAME/sources" | jq -r '.source_blob.get_url' | cut -d"/" -f 8)
echo "Source id: $source_id"
```

Note that we need to extract the *put_url* returned by the `POST` request, which is where our archive will be uploaded. After a successful upload, we also get a *get_url* with the slug ID which we’ll need for the next step. The `jq` tool is invaluable here, providing a simple way to parse the JSON responses from the API; if you are not familiar with it, I recommend learning how to use it.

Finally, the last piece is creating the actual release. This involves submitting a `POST` request to the releases endpoint with the source id we obtained in the previous step. Here’s an example:

```bash
#!/bin/bash
HEROKU_API_KEY=${HEROKU_API_KEY}
HEROKU_APP_NAME=${HEROKU_APP_NAME}
SOURCE_ID=$SOURCE_ID

#create a new release with the latest source
release_id=$(curl -s -n -X POST \
  -H "Accept: application/vnd.heroku+json; version=3" \
  -H "Authorization: Bearer $HEROKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{ \"source_blob\": { \"id\": \"$SOURCE_ID\" } }" \
    "https://api.heroku.com/apps/$HEROKU_APP_NAME/releases" | jq -r '.id')

echo "Release id: $release_id"

# monitor the release
status="pending"
while [ "$status" != "succeeded" ] && [ "$status" != "failed" ]; do
  sleep 5
  status=$(curl -s -n -X GET \
  -H "Accept: application/vnd.heroku+json; version=3" \
  -H "Authorization: Bearer $HEROKU_API_KEY" \
 "https://api.heroku.com/apps/$HEROKU_APP_NAME/releases/$release_id" | jq -r '.status')

 echo "Release status: $status"

done
if [ "$status" == "failed" ]; then
   echo "Release failed, please review logs"
   exit 1
fi
echo "Release completed!"

```

This script posts a request to the releases endpoint using the source id, and it polls the release status until it is complete or has failed. This last step is crucial, as the release process is asynchronous. If you are building a production application, you'll want to incorporate proper monitoring and error handling of this final stage. Failure to do so may leave your application in a broken state, something I have seen happen in a few projects.

This whole process, of course, could be incorporated into a larger script, a makefile, or any CI/CD pipeline that supports running such sequences of commands. My own projects have often taken this approach. In more sophisticated cases, I've also integrated these steps using client libraries in languages such as python using the `requests` library or the Heroku platform API client, which can simplify the JSON handling involved.

For further study on this, I’d recommend examining the official Heroku Platform API documentation; it’s the most authoritative source of truth and contains detailed information on all available endpoints and their respective request/response structures. Additionally, “Effective DevOps” by Jennifer Davis and Katherine Daniels provides general insight into automation practices, helpful for this kind of infrastructure-as-code approach. Finally, if you use a particular programming language in your project, check for libraries that wrap the Heroku API; for example the `heroku3` library for python. These resources will deepen your understanding and equip you to handle more advanced scenarios.

This entire process might seem daunting at first, but once you break it down, the Platform API allows for very sophisticated deployment management. With practice and a good grasp of its mechanisms, you'll find it becomes an invaluable tool in your arsenal.
