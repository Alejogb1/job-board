---
title: "How to set env variables for static methods during Gradle evaluation?"
date: "2024-12-23"
id: "how-to-set-env-variables-for-static-methods-during-gradle-evaluation"
---

Alright, let’s dive into this. I recall vividly a project some years ago, a large multi-module Android application, where we faced this exact hurdle. Managing environment variables, especially for static methods accessed during Gradle’s configuration phase, isn’t as straightforward as it might initially seem. It requires careful consideration of Gradle’s lifecycle and how it executes build scripts. Let me break down my experience and offer some concrete examples.

The crucial thing to understand is that Gradle's evaluation phase, where your `build.gradle` scripts are parsed and executed, runs before the actual tasks. Any static methods called at this stage, particularly those attempting to access external configurations, will not have direct access to runtime environment variables in the typical way you might in a running application. Direct calls to `System.getenv()` within a static initializer of a Java class, for example, will not pick up variables Gradle might expose later.

Instead of directly accessing environment variables at static initialization, you need to pass them into static methods after the evaluation phase. This is typically done during the configuration phase of Gradle. We can use Gradle's `ext` block or the `gradle.startParameter` to pass these variables.

Here’s the first scenario we encountered: Let's say we had a `ConfigurationManager` class with a static method that needs to read a `buildVersion` from an environment variable during Gradle evaluation to version the app. Here’s how we handled it:

**Example 1: Passing via Gradle’s `ext` Block**

First, we had the following static method in our hypothetical `ConfigurationManager` class (we made sure the static initializer did not attempt to access environment variables):

```java
// ConfigurationManager.java
package com.example;

public class ConfigurationManager {
    private static String buildVersion;

    public static void setBuildVersion(String version) {
         buildVersion = version;
    }

    public static String getBuildVersion() {
       return buildVersion;
    }
}
```

In our `build.gradle` file, we used the `ext` block to define the environment variable and then pass it:

```gradle
// build.gradle
import com.example.ConfigurationManager

ext {
  buildVersion = System.getenv("BUILD_VERSION") ?: "1.0.0-default" //default value
}


afterEvaluate {
    ConfigurationManager.setBuildVersion(buildVersion)
}

android {
    // other configs

    defaultConfig {
        applicationId "com.example.myapp"
        minSdkVersion 23
        targetSdkVersion 33

         versionName  ConfigurationManager.getBuildVersion() // Using the static method here

    }

}

```

Here, we fetch the `BUILD_VERSION` environment variable (or default to "1.0.0-default"). Importantly, we use the `afterEvaluate` block. This ensures that the static method `ConfigurationManager.setBuildVersion` is called *after* Gradle’s evaluation phase is completed, thereby avoiding any issues accessing the variable during Gradle’s startup. This pattern avoids direct static method access to environment variables.

**Example 2: Using `gradle.startParameter` to pass Variables**

Let's consider another case, where you are setting up a configuration based on different build types. Suppose that `ConfigurationManager` has another static method for a server URL that depends on the build environment. We can do something similar:

```java
// ConfigurationManager.java
package com.example;

public class ConfigurationManager {

    private static String serverUrl;

    public static void setServerUrl(String url){
        serverUrl = url;
    }

     public static String getServerUrl() {
        return serverUrl;
    }
}
```

In the `build.gradle`:

```gradle
// build.gradle
import com.example.ConfigurationManager

gradle.startParameter.projectProperties.each { key, value ->
    if(key == "environment"){
        if(value == "staging"){
           ConfigurationManager.setServerUrl("https://staging.api.com")
         } else if(value == "production"){
           ConfigurationManager.setServerUrl("https://production.api.com")
         }
    }

}

android {
    // other configs

    buildTypes {
      release {
          buildConfigField "String", "API_URL", "\"${ConfigurationManager.getServerUrl()}\""
          minifyEnabled true
          proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
      }
      debug {
          buildConfigField "String", "API_URL", "\"${ConfigurationManager.getServerUrl()}\""
      }

   }
}
```

This time, we’re using `gradle.startParameter.projectProperties`. When you execute your build command, you can pass a property called `environment` like this:
`./gradlew assembleRelease -Penvironment=staging` or `./gradlew assembleDebug -Penvironment=production`. Gradle will then set the URL according to the passed environment parameter. This method allows for dynamic environment configuration. The configuration manager's static method is called *after* evaluation during the `afterEvaluate` method.

**Example 3: Utilizing a Configuration Object**

For larger configuration sets, it’s often more manageable to create a specific configuration object, which helps keeps `build.gradle` file more manageable. Suppose `ConfigurationManager` now needs multiple parameters:

```java
// ConfigurationManager.java
package com.example;

public class ConfigurationManager {
    private static Config config;
    public static class Config {
        public String buildVersion;
        public String serverUrl;
        public Config(String buildVersion, String serverUrl){
           this.buildVersion = buildVersion;
           this.serverUrl = serverUrl;
        }
    }
    public static void setConfig(Config c) {
        config = c;
    }

    public static Config getConfig(){
        return config;
    }

}
```

Then, inside our `build.gradle`:

```gradle
// build.gradle
import com.example.ConfigurationManager
import com.example.ConfigurationManager.Config


def getEnvironmentUrl = { environment ->
    if(environment == "staging"){
      return "https://staging.api.com"
    } else if(environment == "production"){
      return "https://production.api.com"
    }
    return "https://localhost.api.com"
}


ext {
  buildVersion = System.getenv("BUILD_VERSION") ?: "1.0.0-default"
  environment = project.hasProperty('environment') ? project.getProperty('environment') : "local"
  apiUrl = getEnvironmentUrl(environment)
}

afterEvaluate {
    ConfigurationManager.setConfig(new Config(buildVersion, apiUrl))
}

android {
    // other configs
    defaultConfig {

      applicationId "com.example.myapp"
       minSdkVersion 23
        targetSdkVersion 33

        versionName ConfigurationManager.getConfig().buildVersion
        buildConfigField "String", "API_URL", "\"${ConfigurationManager.getConfig().serverUrl}\""
    }
}
```
Here, we build the entire config object and send it into the static method, and the static method stores the configuration. This is an even cleaner approach for more complex scenarios. It makes the configuration values explicit and improves readability.

In summary, avoiding direct static access to environment variables during Gradle's evaluation phase is critical. These examples demonstrate how to properly inject configuration values into static methods using Gradle's `ext`, `startParameter`, and configuration object patterns, making the entire process more robust and maintainable.

For a deeper understanding of Gradle's lifecycle and script execution, I highly recommend consulting "Gradle in Action" by Benjamin Muschko. It’s a great resource to further build your knowledge in this area. Also, the official Gradle documentation, specifically the sections detailing the configuration phase and task execution, is invaluable for this topic. Finally, looking at the source code of various gradle plugins, especially those that deal with configuration parameters, can provide further practical insights into how these problems are approached. These resources have been instrumental in my past work, and I hope they’ll be helpful to you as well.
