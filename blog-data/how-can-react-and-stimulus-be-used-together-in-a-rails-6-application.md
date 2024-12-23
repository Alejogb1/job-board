---
title: "How can React and Stimulus be used together in a Rails 6 application?"
date: "2024-12-23"
id: "how-can-react-and-stimulus-be-used-together-in-a-rails-6-application"
---

,  I've seen a fair bit of frontend integration in my time, and the React-Stimulus combo within a Rails 6 environment is a pattern that, while initially seeming a little unusual, can actually become quite powerful when leveraged correctly. The key is understanding their distinct roles and how they can play off each other without stepping on each other’s toes. It's not about shoehorning them together, but rather about creating a harmonious balance.

My experience stems from a rather large project back in 2019 – a complex e-commerce platform where certain interactive components needed the full flexibility of React, but we weren't ready to completely abandon the Rails view rendering for the entire application, or commit entirely to a client-side rendering approach. We had legacy code using turbolinks and needed something more refined and manageable. This led us down this specific path of integration between the two.

Essentially, Stimulus acts as a lightweight, unobtrusive JavaScript framework that is perfect for encapsulating discrete, reusable behaviors within a server-rendered HTML page. Think of it as a bridge that connects your server-side rendered elements to the client-side interactivity. React, on the other hand, is much more suited for building complex, dynamic user interfaces where you need a high degree of control over the rendering process and state management.

When combining the two, I tend to view Stimulus as primarily acting on the server-rendered HTML, hooking up elements and providing basic interactions. Then, for specific areas needing more involved functionality, React components are introduced, managed and initialized via Stimulus within the context of their rendered HTML container. In essence, React components are bootstrapped through a Stimulus controller.

Let’s get into how I did that, practically, with the given scenario. The first thing you need to understand is how to bootstrap a React app from within a stimulus controller. This basically revolves around two crucial steps: ensuring React is loaded on the page and then utilizing `ReactDOM.render()` to insert the component inside the specified DOM node, handled via Stimulus.

Here is an example of how I did it:

```javascript
// app/javascript/controllers/react_controller.js
import { Controller } from "@hotwired/stimulus"
import React from 'react';
import ReactDOM from 'react-dom';
import MyReactComponent from '../components/my_react_component'; //Assume this exists

export default class extends Controller {
  static targets = [ "reactRoot" ]

  connect() {
    if(this.hasReactRootTarget){
      ReactDOM.render(<MyReactComponent />, this.reactRootTarget);
      // additional logic can go here, like setting up event listeners
    }
  }

  disconnect(){
      if(this.hasReactRootTarget) {
         ReactDOM.unmountComponentAtNode(this.reactRootTarget);
        // clean up event listeners
     }
  }
}
```

And your Rails view would look something like this:

```erb
  <!--  app/views/your_view.html.erb -->
  <div data-controller="react">
    <div data-react-target="reactRoot" >
    </div>
  </div>
```

Here, the Stimulus controller named `react_controller` is attached to a wrapping div. Inside, there is the essential target div that gets populated with your react component. Note that we need to manage the unmounting of the React app on disconnect, which can help prevent memory leaks if there is a lot of routing/page changes going on in the application.

You also might need to pass data from Rails to the React component. One practical way to do this is to use `data` attributes on the HTML element where the controller is connected. Then, in your Stimulus controller's `connect()` function, retrieve those attributes and pass them down to the React component.

Here is an example with data passing:

```javascript
// app/javascript/controllers/react_controller.js
import { Controller } from "@hotwired/stimulus"
import React from 'react';
import ReactDOM from 'react-dom';
import MyReactComponent from '../components/my_react_component'; //Assume this exists

export default class extends Controller {
    static targets = [ "reactRoot" ]

    connect() {
      if (this.hasReactRootTarget) {
          const initialData = this.element.dataset.initialData ? JSON.parse(this.element.dataset.initialData) : {};

         ReactDOM.render(<MyReactComponent initialData={initialData} />, this.reactRootTarget);
      }
    }

    disconnect(){
         if(this.hasReactRootTarget) {
              ReactDOM.unmountComponentAtNode(this.reactRootTarget);
          }
    }
}
```

And, the Rails view needs to send that data:

```erb
  <!--  app/views/your_view.html.erb -->
  <div data-controller="react" data-initial-data="<%= { name: 'User Name', id: 123 }.to_json %>" >
    <div data-react-target="reactRoot">
    </div>
  </div>
```

In this enhanced example, data is passed as a serialized JSON string to the div where the Stimulus controller lives. The controller retrieves this data, parses it back into an object, and then passes it to the `MyReactComponent` as a prop called `initialData`. This provides a neat way to pass server-side generated data into the React application.

Finally, dealing with more complex cases where you may have multiple react components on a single page, or the need to pass a more comprehensive set of configuration, becomes important. The following is an example on this:

```javascript
// app/javascript/controllers/react_app_controller.js
import { Controller } from "@hotwired/stimulus"
import React from 'react';
import ReactDOM from 'react-dom';

import App1 from '../components/app1';
import App2 from '../components/app2';


export default class extends Controller {
  static values = { appConfig: Object };
  static targets = ["appRoot"];

    connect() {
    if (this.hasAppRootTarget) {
        const {app, props} = this.appConfigValue;

        let reactApp;
        switch(app){
            case "app1":
                reactApp = <App1 {...props}/>;
                break;
            case "app2":
               reactApp = <App2 {...props}/>;
                break;
              default:
                console.error(`Unknown react app type ${app}`);
                return;
        }

      ReactDOM.render(reactApp, this.appRootTarget);
    }
  }

  disconnect(){
    if(this.hasAppRootTarget) {
      ReactDOM.unmountComponentAtNode(this.appRootTarget);
    }
  }
}

```

And the corresponding view would look similar to this:

```erb
  <!-- app/views/your_view.html.erb -->
  <div data-controller="react-app" data-react-app-app-config-value="<%= {app: 'app1', props: {someProp: 'hello'} }.to_json %>">
   <div data-react-app-target="appRoot"></div>
  </div>

  <div data-controller="react-app" data-react-app-app-config-value="<%= {app: 'app2', props: { otherProp: 'world'}}.to_json %>">
    <div data-react-app-target="appRoot"></div>
  </div>
```

This demonstrates how to configure the react application on initialization by utilizing stimulus values. We determine the type of the react application (`app`) and a set of custom properties `props`, and then based on that we initialize the proper application in the root target via `ReactDOM.render()`.

In terms of resources, I recommend looking at the official React documentation for best practices regarding component structures and state management (particularly when transitioning from server-rendered views). For Stimulus, the official documentation is excellent, and reading through issues and examples on the official GitHub repository can give you a solid grasp of its intended usage. Additionally, the book "Modern Front-End Development for Rails" by Noel Rappin is a good reference on how to handle multiple approaches and select the right one.

In summary, using React and Stimulus together in a Rails 6 application is not just possible but can be quite effective if you are strategic about how you divide the labor between the two. Stimulus handles basic behavior and the bootstrapping of React, while React deals with complex, dynamic ui elements. With careful planning and a solid understanding of their individual roles, you can create a maintainable and robust front-end experience that leverages the strengths of both. It’s about choosing the right tool for the job, and in this scenario, both tools can work together very well.
