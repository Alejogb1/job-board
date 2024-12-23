---
title: "How to implement a custom Hibernate table naming strategy in Grails 5?"
date: "2024-12-23"
id: "how-to-implement-a-custom-hibernate-table-naming-strategy-in-grails-5"
---

Alright, let's tackle this one. I remember back in the early days of a project where we inherited a somewhat… *unique* database schema. The existing naming conventions were, shall we say, less than ideal, and forced our hand into implementing a custom Hibernate table naming strategy. It wasn't particularly difficult, but it definitely required diving into the nuances of both Grails and Hibernate configurations. Here’s a breakdown of how I've approached it, and how you can effectively implement your own in Grails 5.

First, let’s frame the problem. Hibernate, by default, uses a somewhat predictable algorithm to generate table names from your domain class names. This is usually fine, but sometimes you have legacy systems, specific organizational standards, or just plain eccentric requirements that demand something different. Grails, sitting atop Hibernate, provides mechanisms to tap into these customizations. The key lies in creating a custom `NamingStrategy` implementation. This class dictates how table and column names are generated.

Here’s how I usually go about it:

1. **Create your custom `NamingStrategy`:** This involves writing a Java (or Groovy, since you’re in Grails) class that implements `org.hibernate.boot.model.naming.PhysicalNamingStrategy`. You’ll need to override methods such as `toPhysicalTableName`, which converts an entity name to a table name. Other methods such as `toPhysicalColumnName`, `toPhysicalSequenceName` and `toPhysicalSchemaName` can also be overridden as needed. I'll focus on `toPhysicalTableName` for now as it's the most common for this case.

2. **Configure Hibernate:** Grails, through Spring Boot’s auto-configuration, typically picks up the `PhysicalNamingStrategy` by its class name if you declare it as a Spring component. But let's explicitly set it in `application.yml` for clarity.

Let's illustrate with a couple of code examples. In the first case, I'll demonstrate a simple strategy that prepends `tbl_` to every table name. This is a common enough request, as I recall.

```groovy
// src/main/groovy/com/example/MyCustomNamingStrategy.groovy

package com.example

import org.hibernate.boot.model.naming.Identifier
import org.hibernate.boot.model.naming.PhysicalNamingStrategy
import org.hibernate.engine.jdbc.env.spi.JdbcEnvironment

class MyCustomNamingStrategy implements PhysicalNamingStrategy {

   @Override
    Identifier toPhysicalTableName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        return Identifier.toIdentifier("tbl_" + name.getText())
    }

    @Override
    Identifier toPhysicalColumnName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        return name; // Default implementation
    }

    // Other toPhysical... methods would also follow the default if we don't override them.
}
```

Now, in `application.yml` (or `application.properties`):

```yaml
hibernate:
  naming:
    physical-strategy: com.example.MyCustomNamingStrategy
```

In this configuration, any domain class in your Grails project will have a corresponding database table prefixed with `tbl_`. For instance, a `Book` domain would result in a table called `tbl_book`.

Now let's consider a slightly more complex scenario, one I actually encountered. We had a requirement to pluralize table names while also converting them to snake case. Here’s how we can accomplish that using a bit of Groovy's dynamic nature and with the help of the `Inflector` class found in the `org.grails.orm.hibernate.support` package:

```groovy
// src/main/groovy/com/example/PluralSnakeCaseNamingStrategy.groovy

package com.example;

import org.hibernate.boot.model.naming.Identifier
import org.hibernate.boot.model.naming.PhysicalNamingStrategy
import org.hibernate.engine.jdbc.env.spi.JdbcEnvironment
import org.grails.orm.hibernate.support.Inflector;

class PluralSnakeCaseNamingStrategy implements PhysicalNamingStrategy {

   private final Inflector inflector = Inflector.getInstance()

    @Override
    Identifier toPhysicalTableName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        String tableName = inflector.pluralize(name.getText()).replaceAll("([a-z])([A-Z])", '$1_$2').toLowerCase();
        return Identifier.toIdentifier(tableName);
    }

    @Override
    Identifier toPhysicalColumnName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        return name; // Default implementation for columns.
    }


   // As before, the others could be defaulted or overridden.
}

```

And the corresponding `application.yml` entry is:

```yaml
hibernate:
  naming:
    physical-strategy: com.example.PluralSnakeCaseNamingStrategy
```

With this configuration, a `Book` domain class would produce a `books` table, and a domain class like `UserProfile` would map to `user_profiles`. It effectively pluralizes and converts camel case to snake case.

Finally, for a more challenging example, let’s pretend we need to prefix tables based on the package structure of the domain class. This pattern was part of an odd architectural choice in another project. This means we might have tables with names like `module_one_tbl_book` if `Book` was in the package `com.example.module.one`. Here's the implementation:

```groovy
// src/main/groovy/com/example/PackagePrefixNamingStrategy.groovy

package com.example

import org.hibernate.boot.model.naming.Identifier
import org.hibernate.boot.model.naming.PhysicalNamingStrategy
import org.hibernate.engine.jdbc.env.spi.JdbcEnvironment
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory
import org.springframework.context.ApplicationContext

class PackagePrefixNamingStrategy implements PhysicalNamingStrategy {

    private final ApplicationContext applicationContext;

    PackagePrefixNamingStrategy(ApplicationContext applicationContext){
        this.applicationContext = applicationContext;
    }

    @Override
    Identifier toPhysicalTableName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        ConfigurableListableBeanFactory beanFactory = (ConfigurableListableBeanFactory) applicationContext.getAutowireCapableBeanFactory();
        Class<?> domainClass = beanFactory.getType(name.getText());
        String packageName = domainClass.getPackage().getName();
        String prefix = packageName.replaceAll("\\.", "_").replace("com_example_","")
        return Identifier.toIdentifier( prefix + "_tbl_" +  name.getText().toLowerCase())
    }

    @Override
    Identifier toPhysicalColumnName(Identifier name, JdbcEnvironment jdbcEnvironment) {
        return name; // Default implementation for columns
    }


}

```

The `application.yml` remains similar, just with the new class:

```yaml
hibernate:
  naming:
    physical-strategy: com.example.PackagePrefixNamingStrategy
```

In this example, I’ve also injected the `ApplicationContext` to retrieve domain class information, which could be useful in more complex scenarios.

Now, a few important notes. First, remember that this strategy is applied to *all* entities within your application. It’s a global setting. Second, pay close attention to performance; simple string manipulations are generally , but avoid computationally intensive operations in these methods as they are executed very frequently. Third, ensure your chosen strategy remains consistent across your entire codebase to avoid confusion. When changing these strategies, it would be beneficial to run schema generation to reflect these changes in the database.

For additional reading, I strongly recommend the official Hibernate documentation, specifically the section on "Naming Strategies." The book "Java Persistence with Hibernate" by Christian Bauer and Gavin King provides a very thorough understanding of the underlying concepts. Furthermore, the Spring Boot documentation on configuration properties and autoconfiguration will illuminate how these settings interact within the Grails ecosystem. Don't solely rely on online tutorials; the primary documentation is often your most reliable guide.

Implementing a custom naming strategy in Grails 5 is definitely achievable with a little work. It provides a powerful method for bending Hibernate to your will, as it were, and resolving the problems arising from inconsistent or legacy database schemas. The key is understanding that you control how table names are generated at the Hibernate level, and it's relatively easy to plug in a custom implementation through Grails configuration. The examples I provided should equip you to get started, but remember to carefully consider your naming needs and thoroughly test any changes you make.
