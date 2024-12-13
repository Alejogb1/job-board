---
title: "gitlab project cannot be transferred because tags are present in its containe?"
date: "2024-12-13"
id: "gitlab-project-cannot-be-transferred-because-tags-are-present-in-its-containe"
---

Okay so you’re having that classic GitLab project transfer problem because of tags right yeah been there done that got the t-shirt feels like a rite of passage for anyone using Git at scale so let's unpack this a bit

first things first GitLab has this quirk or feature depending on how you look at it that prevents you from easily transferring a project with tags to another group or namespace basically it's a safety mechanism to prevent things from getting completely borked during the transfer process think about it tags are like little breadcrumbs pointing to specific commits or releases they might be crucial for your deployment pipelines or documentation or just about anything if you just blindly move the project without considering them things could go south real quick

now the error message is probably something like "Project cannot be transferred because it contains tags" or some variant of that which is incredibly helpful thanks GitLab for the illuminating insight but jokes aside that's about all the guidance we're getting from the platform which is kind of a bummer but we're devs we debug our way through problems not complain about them right

okay so I've been wrestling with this situation many times probably more than I care to admit I remember back in 2018 when I was working on a project that involved migrating our entire infrastructure to a new data center we used GitLab heavily for our code management and when we tried to move the projects to the new group well boom transfer failed because tags it took us hours to figure this out as we were mostly juniors and had no clue about this GitLab quirk We tried everything restarting GitLab doing random things we had no clue what to do We eventually had to get help from one senior dev who knew a bit about Git low level things

so there's actually more than one way to get around this obstacle it really comes down to what you need and how you want to manage your tags for example if you're okay with not keeping the existing tags you can just delete them and then perform the transfer and that's the easiest way actually but if you need them then you have to get creative

okay so here is option one if you want to remove them simple and quick

**Option 1: Delete existing tags**

this is the nuclear option but if you don't need the historical tags or you can recreate them manually it's the fastest way

```bash
git tag -l | xargs git tag -d
git push --delete origin --tags
```

break it down `git tag -l` lists all your local tags `xargs git tag -d` deletes them and `git push --delete origin --tags` will remove them from remote

it's simple its fast but it's destructive think twice before doing it use it if you just starting new project and tags are not needed

here is an alternative option if you want to keep them

**Option 2: Transfer the repository manually**

this is more involved but it allows you to keep your tags and history intact

```bash
git clone --bare <your_gitlab_repo_url>
cd <your_project_name>.git
git push --mirror <new_gitlab_repo_url>
```

it's pretty straightforward `git clone --bare` clones the entire repository including all branches and tags `cd` just changes your directory to your repo and `git push --mirror` pushes everything to the new repository including the tags

this way you essentially create a duplicate of the project with all the tags on the new location. After that you can delete the old repo and rename the new one

it's a bit more work but it's way less risky than just deleting stuff

and yet another option a bit more GitLab-specific but it works with GitLab API

**Option 3: Use GitLab API to transfer and recreate tags**

this one is for the more automated or scripting inclined person

```bash
# Get project ID
PROJECT_ID=$(curl --header "PRIVATE-TOKEN: <your_gitlab_token>" "https://<your_gitlab_domain>/api/v4/projects?search=<your_project_name>" | jq -r '.[0].id')

# Get all tags
TAGS=$(curl --header "PRIVATE-TOKEN: <your_gitlab_token>" "https://<your_gitlab_domain>/api/v4/projects/${PROJECT_ID}/repository/tags" | jq -r '.[].name')

# Transfer project
curl --request POST --header "PRIVATE-TOKEN: <your_gitlab_token>" --data "id=$PROJECT_ID&namespace_id=<your_new_group_id>" "https://<your_gitlab_domain>/api/v4/projects/$PROJECT_ID/transfer"

# Recreate tags in the new project
NEW_PROJECT_ID=$(curl --header "PRIVATE-TOKEN: <your_gitlab_token>" "https://<your_gitlab_domain>/api/v4/projects?search=<your_project_name>" | jq -r '.[0].id')

for tag in $TAGS
do
    COMMIT_SHA=$(curl --header "PRIVATE-TOKEN: <your_gitlab_token>" "https://<your_gitlab_domain>/api/v4/projects/${PROJECT_ID}/repository/tags/$tag" | jq -r '.target')
  curl --request POST --header "PRIVATE-TOKEN: <your_gitlab_token>" --data "tag_name=$tag&ref=$COMMIT_SHA" "https://<your_gitlab_domain>/api/v4/projects/${NEW_PROJECT_ID}/repository/tags"
done

```

this is a bit more complex we use curl to do API calls `PROJECT_ID` fetches id `TAGS` fetches tags names then we transfer the project to a new group using `transfer` then we fetch the new project `NEW_PROJECT_ID` then we loop each tag to its SHA then we recreate them with API call the token is your personal access token with api permission

this is a bit more cumbersome to set up but it lets you fully automate the entire process including keeping your tags intact without manual labor

okay so you're probably thinking which one should i use right well it depends on your needs and how comfortable you are with the command line or api the first option is easiest but it nukes the tags the second one is a bit more hands on but keeps the tags and the last option is very flexible and automatable but requires more setup and understanding of the GitLab API

now a word of caution about the API the way GitLab works changes pretty frequently so if you're using this approach make sure to keep an eye on the API documentation to avoid surprises down the road it's always a good habit to read those docs

for learning more about Git I would highly recommend "Pro Git" by Scott Chacon and Ben Straub it's an amazing resource that dives deep into all of Git's features and internals or "Version Control with Git" by Jon Loeliger it's very comprehensive also for GitLab specific things the documentation is actually very good and regularly updated

one final tip before I wrap this up always test your solution on a test project before applying it to a production one this is something that I learned the hard way it's always better to be safe than sorry no matter how simple things look on the surface trust me I've had my share of late nights debugging code and dealing with issues from the most unexpected places

hope that helps you move your project without losing your mind I know it was a pain for me when I first encountered this it's one of those weird things that you deal once and never forget ever and it might even help you teach someone else later on good luck
