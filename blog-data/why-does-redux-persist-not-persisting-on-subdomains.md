---
title: "Why does Redux-persist not persisting on subdomains?"
date: "2024-12-14"
id: "why-does-redux-persist-not-persisting-on-subdomains"
---

here's the deal, i've been there, staring at the screen wondering why my redux state is vanishing into thin air on a subdomain. it's one of those experiences that makes you question everything you thought you knew about local storage, domain scopes, and persistence libraries. let's break it down, from my perspective, having spent way too many nights debugging this exact scenario.

the core issue, at least from what i've seen, isn't really a bug in redux-persist itself. it's more about how browsers handle storage in relation to domains and subdomains. think of it like this: local storage, the primary mechanism redux-persist often uses, is scoped to the *exact* domain or subdomain it’s created on. so, when you set something in `app.example.com` it is completely walled off from `www.example.com` or even `api.example.com`. each one of those is treated as a distinct origin by the browser. this behavior is a security feature, browsers don't want any cross-domain shenanigans with your data.

now, when we talk about redux-persist, it is attempting to read from, or write to, this storage. if your app starts on `app.example.com` and you've configured redux-persist there, then it’s working perfectly fine, storing your state in the local storage associated with this subdomain. but if you then navigate to `www.example.com`, a totally separate storage space kicks in, one that redux-persist knows nothing about, and there lies the problem. your state is not gone, it's just sitting pretty in the wrong storage location. this can be especially frustrating if you assume that local storage is one giant data blob shared across the entire `example.com` domain. it’s not. it’s like having multiple houses with same address but different postal codes.

over my years, i've seen this trip up a lot of people, myself included. i remember this one project that i inherited that had a user profile section on `profile.company.com` and the main app on `app.company.com`. data changes on the profile page weren’t appearing on the app page after a refresh. at first, it felt like a redux-persist config problem, i was chasing ghosts. i was checking serializer configurations, storage engine configurations and made sure that my reducers were pure. i felt like i was losing my mind! only after checking my browser network logs and carefully inspecting cookies, i found out that i was on different subdomains. after this specific incident, the solution felt obvious but until then, i felt very stupid.

so how do you fix it? well, you have a couple of options. the best practice is to move your application to a single domain with different paths, this avoids all storage issues. but i understand you can't always change the project architecture. another way you can manage it, is using a specific storage configuration which uses cookie storages. it's more work and can be a security issue as cookies have size limits and need specific security configurations, but it can be done. you can set the cookie path to `/` to share it across the whole domain, but i strongly suggest using another approach. or as a last option, you can use a centralized storage with your backend but that's outside the redux-persist scope.

here's a code snippet, using redux-persist's `persistStore`, showing a basic setup. this is fine if you stick to a single subdomain. notice that no domain or path are being configured here:

```javascript
import { createStore } from 'redux';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

const persistConfig = {
  key: 'root',
  storage,
};

const rootReducer = (state = { myData: null }, action) => {
  switch(action.type){
    case 'SET_DATA':
      return {...state, myData: action.payload}
    default:
      return state
  }
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = createStore(persistedReducer);
const persistor = persistStore(store);

export { store, persistor };
```

now, let's look at an example of how *not* to do it. imagine trying to shoehorn in some subdomain logic directly into the persist config. this wouldn't work as the storage is browser scoped:

```javascript
// this will not work correctly
import { createStore } from 'redux';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; // defaults to localStorage for web

const subdomain = window.location.hostname.split('.')[0] // not good enough

const persistConfig = {
  key: `root-${subdomain}`, // this will create separate storages
  storage,
};

const rootReducer = (state = { myData: null }, action) => {
  switch(action.type){
    case 'SET_DATA':
      return {...state, myData: action.payload}
    default:
      return state
  }
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = createStore(persistedReducer);
const persistor = persistStore(store);

export { store, persistor };
```

this approach doesn't solve the underlying problem of storage being separated by origin. every subdomain will get it's own storage. it doesn’t address the core issue of different subdomains having different local storage scopes. trying to hack it by appending the subdomain to the storage key just creates separate storage areas that can cause confusion when the application migrates across subdomains.

if you have to use subdomains then you may need to opt for a different storage solution. here’s a basic snippet with a cookie storage engine, though i caution against this approach without due consideration for security and size limits, it’s for demonstration purposes only:

```javascript
import { createStore } from 'redux';
import { persistStore, persistReducer } from 'redux-persist';
import createWebStorage from "redux-persist/lib/storage/createWebStorage";

const createNoopStorage = () => {
  return {
    getItem(_key) {
      return Promise.resolve(null);
    },
    setItem(_key, value) {
      return Promise.resolve(value);
    },
    removeItem(_key) {
      return Promise.resolve();
    },
  };
};

const storage = typeof window !== "undefined" ? createWebStorage("local") : createNoopStorage();

const cookieStorage = {
  getItem: async (key) => {
      const cookies = document.cookie.split('; ').reduce((acc, cookie) => {
          const [cookieKey, value] = cookie.split('=');
          acc[cookieKey] = decodeURIComponent(value);
          return acc;
      }, {});
      return cookies[key] || null;
  },
  setItem: async (key, value) => {
      document.cookie = `${key}=${encodeURIComponent(value)}; path=/`;
  },
  removeItem: async (key) => {
      document.cookie = `${key}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
  },
};

const persistConfig = {
  key: 'root',
  storage: cookieStorage, // use the cookie engine instead
};

const rootReducer = (state = { myData: null }, action) => {
    switch(action.type){
      case 'SET_DATA':
        return {...state, myData: action.payload}
      default:
        return state
    }
};

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = createStore(persistedReducer);
const persistor = persistStore(store);

export { store, persistor };
```

this last approach uses a simple implementation of a cookie-based storage. keep in mind that it may need extra security configurations and that storing large state objects in cookies is not usually recommended as they have size limitations.

the core takeaway is this: redux-persist isn't failing you, browser security is doing its job. be mindful of your domain configuration when setting up storage, especially if you’re using subdomains. single domain/path architectures are usually easier to manage but if that's not an option, consider alternative storage mechanisms and make sure that your cookies or other storage solution is properly configured. using subdomains is a complicated task as browser storage has some limitations that can lead to unexpected results. don’t blame redux-persist, blame your network! just joking. i know, it is a bad joke, i’ll show myself out.

if you want to read up on browser storage specifics, i'd recommend digging into the w3c specifications for web storage. those are your go-to resources for understanding the nitty-gritty details on domain and origin restrictions. or for more practical hands-on guide i recommend reading "programming javascript applications" from eric elliot which goes into the details on browser local storage and its limits in web applications. it’s a good read.

happy debugging!
