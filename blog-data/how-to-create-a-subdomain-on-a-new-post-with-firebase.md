---
title: "How to create a subdomain on a new post with firebase?"
date: "2024-12-15"
id: "how-to-create-a-subdomain-on-a-new-post-with-firebase"
---

alright, so you're looking to spin up subdomains dynamically when a user creates a new post using firebase. i've been down this road a few times, it's not always a straight shot, but definitely doable. let me walk you through how i've tackled this in the past, focusing on a solid, workable approach.

first things first, the core issue here isn't firebase itself doing the subdomain creation – firebase is the backend, it deals with your data. the subdomain part is more of a dns and web server configuration thing. we're going to use firebase functions to handle the logic, and then configure your hosting provider to direct traffic.

my first attempt at this was... messy, to put it mildly. i tried using some convoluted cloud functions that triggered on document creation, and it ended up with rate limiting issues and some very, *very* strange behavior with cloud dns. lesson learned: keep it simple, keep it serverless, and *don't* try to manage dns records directly through code unless you absolutely have to. i also attempted initially to use firebase's custom domains feature directly with cloud functions and that lead to configuration hell. i ended up with a lot of 'page not found' errors and took me a while to figure out that wasn't the correct approach.

the approach that worked best for me involved a couple key steps:

1.  **generating the subdomain:** in your cloud function, when a new post is created, you'll generate a subdomain string. this could be a slug based on the post title, a unique id, or any other logic that makes sense for your application. make sure to sanitise this string so that it is valid as subdomain. lowercase, no weird characters.

    ```javascript
    const functions = require('firebase-functions');
    const admin = require('firebase-admin');
    admin.initializeApp();

    exports.generateSubdomain = functions.firestore
      .document('posts/{postId}')
      .onCreate(async (snap, context) => {
        const postData = snap.data();
        const postId = context.params.postId;

        // sanitize and make sure it is a valid subdomain
         const subdomain = postData.title
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '-')
             .replace(/-+/g, '-') //removes multiple consecutives dashes
            .replace(/^-+|-+$/g, '')//removes start and end dashes
          if(!subdomain){
            // handle error
            console.error('invalid title');
            return null;
         }

        const subDomainSlug = `${subdomain}-${postId}`
        // save it somewhere (posts document is a good place)
       await snap.ref.update({
           subdomain: subDomainSlug
        });

        console.log(`Subdomain for post ${postId} generated: ${subDomainSlug}`);
        return null;
      });

    ```
    this code snippet listens for a new post creation and generate a subdomain from the title and the postid. it saves the subdomain in the post's document.

2.  **setting up a wildcard dns record:** you need to configure your domain's dns to point all subdomains to your firebase hosting. this is done through your dns provider's control panel. you’ll typically create an `a record` with a `*` as the hostname (e.g., `*.yourdomain.com`) that points to firebase’s hosting ip address. check firebase hosting documentation for the correct ip.

3.  **dynamic hosting with firebase rewrites:** in your `firebase.json` file, you'll need to set up rewrite rules to handle the subdomains dynamically. firebase hosting can match incoming requests based on the hostname and forward them to specific functions or web pages. i use a simple function to handle all subdomains requests and then serve the posts data from it. i usually add the subdomain to the url, and then the function extracts it to query the post.

    ```json
    {
      "hosting": {
        "rewrites": [
         {
            "source": "**",
            "function": "subdomainHandler"
          }
        ]
      }
    }
    ```
    this configuration tells firebase to forward all request to a cloud function named `subdomainHandler`

4.  **cloud function to serve the subdomain content:** now this cloud function becomes the key part, it receives the subdomain url and extracts the subdomain from it, it queries the post from the database using the subdomain slug, and renders the post page. this method ensures the dynamic functionality, all posts will be accessible under their own subdomains with no additional configuration.

    ```javascript
    const functions = require('firebase-functions');
    const admin = require('firebase-admin');
    admin.initializeApp();


    exports.subdomainHandler = functions.https.onRequest(async (req, res) => {

       const host = req.headers.host; //grab the host
       const subdomain = host.split('.')[0]; //get the subdomain
      //query the database for the corresponding post
      const postsRef = admin.firestore().collection('posts');
      const query = await postsRef.where('subdomain', '==', subdomain).get();

        if (query.empty) {
          // handle error or redirect to 404
           res.status(404).send('post not found');
           return
        }

        query.forEach(doc => {
           const post = doc.data();
          // render the page with data of the post
           res.send(`<h1>${post.title}</h1> <p>${post.content}</p>`);
        });
      return null;

     });
    ```

    this cloud function gets the subdomain name from the host header and queries the database to find a post with matching subdomain, it renders a simple page with the post data.

  it's really important to make sure you're doing proper input sanitisation when generating your subdomains and querying the database, this is a very common place for vulnerabilities. a simple error could cause an unexpected error or a major security issue. also error handling is really important to avoid confusing errors and to handle edge cases.

  a few pointers i've learned over time:

*   **caching:** firebase functions have caching enabled by default. this can be a pain when you're deploying changes, make sure to deploy them with `-f` parameter to force the function to update if needed.
*   **dns propagation:** dns changes can take some time to propagate. be patient after changing your dns settings. usually i end up drinking an entire cup of coffee before it starts working.
*   **security rules:** make sure you've properly configured firebase security rules for accessing your posts and any other data. this is important to make sure you are in control of who can access your data.
*   **debugging:** cloud function logs are your best friend. if something is going wrong, check the logs and look for error messages. if you are a bit lazy like me you can use console log messages, even though is not a best practice.
*   **performance:** avoid heavy operations inside your firebase functions, this can lead to longer requests and cold starts. optimize your functions as much as you can.

for further reading, i’d recommend the following resources. while they're not specific to firebase + subdomains, they'll give you a solid foundation:

*   "high performance browser networking" by ilya grigorik – great for understanding the underlying mechanics of dns and web protocols. you should really read the entire thing.
*   "cloud native patterns" by kukielka & nelson – gives you great overview of cloud architecture and best practices for scalable applications.
*   firebase documentation: obviously. it's the place to look for official information on specific topics

this is what's worked for me in the past. it may seem complex initially but it's a pattern that scales nicely. always start with a simple example, like the examples i provided, before jumping into more complex setups. and test everything carefully on your staging environment before deploying it to production. let me know if anything else comes up, i'm happy to elaborate or help out.
