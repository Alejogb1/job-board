---
title: "When I go to the dashboard route, how do I show my hamburger icon?"
date: "2024-12-15"
id: "when-i-go-to-the-dashboard-route-how-do-i-show-my-hamburger-icon"
---

so you've got this hamburger icon problem on your dashboard route yeah I get it Been there done that more times than I care to admit It’s one of those deceptively simple things that can trip you up if you’re not thinking straight or you know having a bad day or something

Let me tell you about this one time back in my early days I was working on this project a single page application a SPA thing right It was supposed to be super sleek and responsive You know the whole nine yards Anyway we had this navigation bar that used to just kinda sit there all static But then of course the design team the ones who have never coded a day in their life decided it needed to be a sliding drawer menu on mobile I’m sure you can see where this is going And I was all like sure no problem I’ll just slap a hamburger icon on that bad boy

So I did I just stuck a div with some css to look like a hamburger and said good enough The thing was I only toggled the drawer on the root page It did work but then as soon as the user went to say /dashboard the hamburger completely vanished Like poof disappeared into the ether Turns out I was only checking for the root path for rendering the icon not every other route

The frustration I felt that day was real so let's get you sorted So here’s how we usually handle this situation you want to show your hamburger icon only on the dashboard route correct It's not rocket science it's just about how and where you render your hamburger component and how you control its visibility

First thing is first you need to understand your routing setup I'm assuming you are using something like React Router or Angular Router or Vue Router something in that same ballpark right Your routing library is what determines the current route the current path we are at and you need to hook into that to make the decision of displaying your hamburger or not

Here's a basic example using React Router to make this clear

```javascript
import React from 'react';
import { BrowserRouter as Router, Route, Switch, useLocation } from 'react-router-dom';

function HamburgerIcon() {
  return (
    <div style={{ cursor: 'pointer' }}>
      {/* Your hamburger icon elements go here svg or divs */}
      <div style={{ height: '3px', width: '25px', backgroundColor: 'black', margin: '3px 0' }}></div>
      <div style={{ height: '3px', width: '25px', backgroundColor: 'black', margin: '3px 0' }}></div>
      <div style={{ height: '3px', width: '25px', backgroundColor: 'black', margin: '3px 0' }}></div>
    </div>
  );
}

function Dashboard() {
  return <div><h1>Dashboard</h1></div>
}

function Home() {
  return <div><h1>Home</h1></div>
}

function App() {
  const location = useLocation();
  const showHamburger = location.pathname === '/dashboard';

  return (
    <Router>
      <div>
        {showHamburger && <HamburgerIcon />}
        <Switch>
          <Route path="/dashboard">
            <Dashboard />
          </Route>
          <Route path="/">
            <Home />
          </Route>
        </Switch>
      </div>
    </Router>
  );
}
export default App
```

In this example see the `useLocation` hook? that's how we access the current path and then we have a boolean `showHamburger` which is true only when the pathname is `/dashboard` simple straightforward right

Now in the earlier problem I described earlier I got tripped by that because I wasn’t thinking in terms of reusing my menu component I had a bunch of logic intertwined in my top level component which is a mess I learned the hard way not to do that again ever

Ok what about an angular version right here is some equivalent code for Angular

```typescript
import { Component } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  template: `
    <div *ngIf="showHamburger">
      <div style="cursor: pointer;">
      <div style="height: 3px; width: 25px; background-color: black; margin: 3px 0;"></div>
      <div style="height: 3px; width: 25px; background-color: black; margin: 3px 0;"></div>
      <div style="height: 3px; width: 25px; background-color: black; margin: 3px 0;"></div>
    </div>
    </div>
    <router-outlet></router-outlet>
  `,
})
export class AppComponent {
  showHamburger: boolean = false;

  constructor(private router: Router) {
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: any) => {
      this.showHamburger = event.url === '/dashboard';
    });
  }
}
```

Here I am using Angular’s Router’s event system I'm listening for NavigationEnd events and setting the `showHamburger` flag based on the url And of course there is a simple conditional display using the `ngIf` directive. So you are probably thinking why not use a more generic approach well thats great thinking and that is my favorite way of solving it

Consider this component I always use for displaying a common navigation

```javascript
import React from 'react';
import { useLocation } from 'react-router-dom';

function Navigation({ children, routesWithHamburger }) {
    const location = useLocation();
    const showHamburger = routesWithHamburger.includes(location.pathname);
  
    return (
      <nav>
        {showHamburger && <HamburgerIcon />}
        {children}
      </nav>
    );
  }
  
  export default Navigation;
```

This is a versatile component you can use in multiple locations with different routes it takes an array of route paths and that's it all you need to do is to specify the routes that need a hamburger This approach makes your code more maintainable and easier to read.

So that was my adventure with the disappearing hamburger icon It wasn’t pretty and I’m sure many other people had the same or worse experiences You just have to think of how the application renders to understand this stuff This is also something important you need to always remember

Now here are some resources that can help you understand routing better because its crucial for solving this type of problem

For a deep dive into React Router you can check out "React Router" by Remi Van Der Veen its super detailed and will cover a lot of routing scenarios and more advanced setup you might encounter later on

For Angular check out "Angular Router" from Deborah Kurata it's an old but gold resource for everything related to navigation in Angular projects and remember that most of this concepts are frameworks agnostic so it will help you in understanding core principles

Finally if you really want to solidify your knowledge of client-side routing in general you should read "Understanding Single-Page Applications" by Todd Motto this is a classic resource that covers architectural considerations that may help you later on

And oh here’s a tech joke for ya why did the programmer quit his job because he didn't get arrays He always felt like a zero

 bad joke I know but I had to get it out of the system Anyway make sure to use the concepts described in the code snippets above you will be fine you've got this If you have more questions feel free to ask and I'll be here and we all do our best to help each other
