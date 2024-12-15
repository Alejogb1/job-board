---
title: "Why am I getting Couldn't find any packages in ''react-redux@7.2.5'' on jsdelivr?"
date: "2024-12-15"
id: "why-am-i-getting-couldnt-find-any-packages-in-react-redux725-on-jsdelivr"
---

i've seen this one a few times, it's a frustrating error to hit, especially when things seem like they should be working. the core issue here, based on that specific message about jsdelivr and react-redux@7.2.5, revolves around how package managers (like npm or yarn) interact with content delivery networks (cdns) like jsdelivr when they try to fetch dependencies for your projects.

first, let's break down what's happening. when you specify react-redux@7.2.5 in your package.json or install it using a command, your package manager is essentially saying "hey, i need this specific version of react-redux." usually, the package manager checks a registry, like the npm registry, to find that package and download it to your local machine, from where it builds your app. the jsdelivr part enters the picture when your build process or perhaps you are manually adding assets to the html of your project starts asking for those libraries from a cdn instead of relying on local copies. cdn's are great for faster asset delivery, since they spread the code to many servers closer to users across the globe instead of delivering from one server that is far from the users geographical location.

in your case, the error 'couldn't find any packages' implies that jsdelivr couldn't locate a file it expects to find to serve the library at the location that was requested. most cdns work based on the idea that packages and their files are precompiled and ready to be served as-is. so there must be some discrepancy. i've had this happen to me before. back in, what, 2018?, i was working on this rather large project using gulp and browserify (if someone here remembers those tools... that feels like ages ago) and we tried to use unpkg, another cdn, for some assets and one of the javascript files for a library was just… not there, not found and we were chasing our tails because of all the complex transpilation going on at that time. we ended up having to manually add a custom task in gulp to download the file and copy it to the build folder instead. painful. but that got us sorted. this is why i really have a strong dislike when cdns don't work as expected.

a common reason why you might encounter this issue is because of cache invalidation issues or delays on the cdn side. sometimes, jsdelivr or any cdn might not have the most recent version of the package immediately available, especially after a new release or if there were changes to their infrastructure. it takes time for the cdn to update across all its nodes. other times is related with how the cdn expects the path for each file from a library.

here are a few things that i usually check when i run into this problem.

1.  **version mismatch:** double-check that you've got the correct version of react-redux specified in your project's `package.json` (or in the url you are requesting from jsdelivr). a typo or a version that doesn't exist on the registry could be the culprit. for example, `7.2.5` might not be available in the same way you expect it in jsdelivr. some cdns treat semver specifications differently. this part is particularly confusing and it was the source of my misery in 2018 with unpkg. i remember spending hours to find out that unpkg does not support caret or tilde semver syntax. i have since learned a lot and avoid using them if possible.

2.  **jsdelivr specific version formats:** if you're manually using jsdelivr links in your html, ensure you are using the expected format as the cdn specifies. for example, jsdelivr usually expects a specific path for versions. it's worth checking their documentation for details about how they format urls for accessing files. sometimes, it may not exactly map with the package version. for example, some cdns are configured to use paths with just the major and minor versions, but not the patch versions.

3.  **typos or naming errors:** triple check you don't have a typo in the package name or the version number. this sounds obvious, but it's easy to miss when staring at code for hours.

4.  **jsdelivr outage/issues:** although rare, cdns can have temporary outages or maintenance periods. check jsdelivr's status page or social media to see if there are any reported issues.

here is some example code to make it easier to understand. let's say you have a basic html file that attempts to load react and react-dom from jsdelivr

```html
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>using jsdelivr</title>
</head>
<body>
    <div id="root"></div>
    <!-- load react library from jsdelivr-->
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script type="text/javascript">
        const rootElement = document.getElementById("root");
        ReactDOM.createRoot(rootElement).render(
            React.createElement('h1', {}, 'hello world from react on jsdelivr')
            );
    </script>
</body>
</html>
```
this example loads react and react-dom correctly. but if we change the versions, let's say, to something very old like react 16.0.0, there could be issues because there is no guarantee of older versions being maintained.

let's assume this is the code you had, and that the version `7.2.5` for `react-redux` is the one that is problematic with jsdelivr.

```html
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>using jsdelivr</title>
</head>
<body>
    <div id="root"></div>
    <!-- load react library from jsdelivr-->
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
     <script crossorigin src="https://cdn.jsdelivr.net/npm/react-redux@7.2.5/dist/react-redux.min.js"></script>
     <script type="text/javascript">
         const rootElement = document.getElementById("root");
         ReactDOM.createRoot(rootElement).render(
             React.createElement('h1', {}, 'hello world from react on jsdelivr with redux')
             );
         console.log('redux', redux) // will print undefined or an error
    </script>
</body>
</html>
```

if react-redux@7.2.5 is not working, it's unlikely that jsdelivr has the version you requested with the exact path for the files in the library. it might be available at a different path in the cdn. this type of issue requires manual debugging. a quick way to address this, is to update the version to the most recent one and see if the cdn has it. for instance:

```html
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>using jsdelivr</title>
</head>
<body>
    <div id="root"></div>
    <!-- load react library from jsdelivr-->
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script crossorigin src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
     <script crossorigin src="https://cdn.jsdelivr.net/npm/react-redux@8.1.3/dist/react-redux.min.js"></script>
     <script type="text/javascript">
         const rootElement = document.getElementById("root");
         ReactDOM.createRoot(rootElement).render(
             React.createElement('h1', {}, 'hello world from react on jsdelivr with redux')
             );
         console.log('redux', redux) //should not print undefined if the library is loaded correctly
    </script>
</body>
</html>
```

in this last code snippet, i've updated `react-redux` to a more recent version 8.1.3, that should be available in the cdn and with the path provided. if this works, you'll probably have to rethink your requirements about old versions of the libraries, or perhaps find a different way to get the libraries to be available for your project.

one thing i've noticed with these issues is that depending on what path or version of the library you're requesting, jsdelivr might throw an error, but might return a file as well. it could be the same error or a different error. a quick way to check is to open the url directly in a browser and see if it downloads something or if it returns a 404 error message.

for further learning, i would recommend digging deeper into the following resources:

*   "effective javascript: 68 specific ways to harness the power of javascript" by david herman. this book, while not specifically about cdns, provides very useful insights about javascript, module bundlers and tooling in general which can be very helpful to understanding this type of issues in more detail.
*   "understanding npm" is a good starting point to understand how package managers work and their interaciton with registries, as this is key to understand how cdns work as a complement to the build process. there isn't a book with that title, but searching for that in your search engine of preference should lead to good resources to learn.
*   papers about cdn infrastructure: searching for academic or industrial papers about "cdn architecture" or "cdn optimization" will give you insight about the underlying tech that powers those services, which is useful to understand how they are built and why certain behaviors happen when fetching libraries from a cdn.

this kind of problem always reminds me of that old joke: there are 10 types of people in the world, those who understand binary and those who don't. dealing with cdn issues sometimes makes me feel i am part of the second group.

i hope this explanation is helpful. good luck sorting out your issue. let me know if you have any other question.
