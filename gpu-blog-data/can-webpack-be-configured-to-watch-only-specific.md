---
title: "Can webpack be configured to watch only specific entry points?"
date: "2025-01-30"
id: "can-webpack-be-configured-to-watch-only-specific"
---
Webpack's watch mode, while powerful, doesn't inherently offer a direct configuration to selectively monitor *only* specific entry points for changes. Instead, it primarily leverages the entire dependency graph originating from the defined entry points. This means that a modification in any file imported by, or importing, an entry point will trigger a recompile, even if the specific entry point itself remains untouched. However, through careful configuration of Webpack's features and some strategic file organization, a *behavior* akin to monitoring specific entry points can be achieved, enhancing development workflows in larger projects. My experience working on several micro-frontend projects where granular control over rebuild times was crucial, taught me the importance of understanding these nuances.

The core challenge lies in the fact that Webpack's watch mechanism is fundamentally based on changes within the dependency graph, not solely on the entry points themselves. An entry point serves as the root of the dependency tree that is traversed during bundling. When any file within this graph changes, Webpack triggers a new build. To emulate selective watching, we need to limit the scope of the dependency graph associated with each "watched" entry point. This can be accomplished using techniques focusing on minimizing shared module import dependencies between different entry points. We can achieve this isolation using techniques like file organization, using different output directories for different entries, and selective caching. This avoids triggering rebuilds in areas of the codebase we're not actively working on.

One critical approach is employing distinct folder structures mirroring the intended granularity. For example, consider a project with multiple micro-applications, each having its own entry point. Instead of placing all source code in a monolithic "src" folder, each micro-app is given its dedicated directory, such as `src/app1`, `src/app2`, and so on. Webpack's entry configuration should then directly point to these specific entry points within their designated folders. If the intention is to work primarily on `app1`, the build process will be confined to the files within `src/app1` and their respective dependencies. However, any shared components should be placed in a `shared` folder and their impact carefully considered to avoid unnecessary rebuilds. This separation limits the impact of a change to one specific area.

Let's consider the following example, illustrating a typical entry configuration for multiple applications, each with their respective source directories and a shared folder.

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
  mode: 'development',
  entry: {
    app1: './src/app1/index.js',
    app2: './src/app2/index.js',
    app3: './src/app3/index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name]/[name].bundle.js',
  },
  resolve: {
    modules: [
      path.resolve(__dirname, 'src'),
      'node_modules'
    ]
  },
  watchOptions: {
    ignored: /node_modules/,
    aggregateTimeout: 200, // Reduce debounce effect
  },
  // ...other configurations
};
```

In this configuration, each application has its unique entry point within its respective directory. The `output` configuration ensures that the generated bundles are placed in separate folders within the `dist` directory, avoiding potential conflicts or accidental shared scopes.  The `resolve.modules` setting allows import statements to start in the src directory, enabling importing from shared folder such as: `import { myComponent } from "shared/myComponent";`. Setting `watchOptions.ignored` will avoid needless checks in node\_modules which drastically speeds up the watcher. While a change to a file in `src/app1` would not trigger a rebuild of `src/app2`, a change to a shared module will recompile both `app1` and `app2` since they share a dependency.

To further illustrate how we can improve the granularity, consider a scenario with a shared component, say `ui-kit/Button.js`, that is frequently modified during development.  To address the need of a more specific build we can use a "code splitting strategy," for example, making components lazy loaded via dynamic import.  This requires using an import statement that signals to webpack to bundle it as separate chunk.

```javascript
// src/app1/index.js
import React from 'react';
import ReactDOM from 'react-dom';
const App = () => {

    const handleClick = () => {
       import(/* webpackChunkName: "ui-kit-button" */'ui-kit/Button').then(({ Button }) => {
         const btn = new Button()
         console.log(btn.render());
         // some logic
       });
    };


  return (
    <div>
        <button onClick={handleClick}>Click me</button>
    </div>
  );
};
ReactDOM.render(<App />, document.getElementById('root'));

```
In this scenario the 'ui-kit/Button' is loaded lazily and it's webpack output will create a separate chunk as `ui-kit-button.bundle.js`, ensuring minimal impact on other entry points during development. The first time the `handleClick` method is invoked it will cause the `ui-kit-button` chunk to load, this dynamic importing is not recommended for all components but for components that change often it can be beneficial to reduce bundle times and build impact.

For situations where granular control is critical, but lazy loading is not suitable, leveraging multiple webpack configurations can be an option. Each configuration can focus on a particular area of development. This would entail having different `webpack.config.js` files, one for `app1`, one for `app2`, and so on. This is not a single `watch` solution as envisioned, but it would produce different builds for each app that could be run separately and still provide fine-grained control. While it isn't the most efficient or elegant, it does allow us to focus the build process on what we are currently working on.

```javascript
// webpack.app1.config.js
const path = require('path');
module.exports = {
  mode: 'development',
  entry: {
    app1: './src/app1/index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'app1/app1.bundle.js',
  },
  resolve: {
    modules: [path.resolve(__dirname, 'src'),'node_modules'],
  },
    watchOptions: {
    ignored: /node_modules/,
    aggregateTimeout: 200, // Reduce debounce effect
  },
  // ...other configurations
};


// webpack.app2.config.js
const path = require('path');
module.exports = {
  mode: 'development',
  entry: {
    app2: './src/app2/index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'app2/app2.bundle.js',
  },
  resolve: {
    modules: [path.resolve(__dirname, 'src'),'node_modules'],
  },
  watchOptions: {
    ignored: /node_modules/,
    aggregateTimeout: 200, // Reduce debounce effect
  },
  // ...other configurations
};
```

To use these configurations you can use `webpack --config webpack.app1.config.js --watch` and `webpack --config webpack.app2.config.js --watch` to run each configuration separately, and in `watch` mode. This provides maximum isolation and control over the build process, but comes with the overhead of maintaining multiple configurations and running multiple build processes.

While Webpack does not provide direct support for watching specific entry points, we can achieve an effective level of control through careful project organization, code splitting techniques and potentially using different configurations. The key is to understand the underlying dependency graph and work to minimize the impact of code changes between different entry points. This allows for a focused and faster development experience, essential in larger and more complex projects.

For further study, I recommend exploring the official Webpack documentation regarding entry points, output management, code splitting, and caching. It's also beneficial to examine best practices for micro-frontend architectures, which often require similar granular control over build processes. Furthermore, delving into the concept of tree shaking, when applied strategically, can further minimize bundle sizes and improve performance during development and production.
