---
title: "Where are Tomcat security constraints defined: web.xml or WEB-INF/web.xml?"
date: "2024-12-23"
id: "where-are-tomcat-security-constraints-defined-webxml-or-web-infwebxml"
---

, let's talk about Tomcat security constraints. I've had my share of late nights troubleshooting these, so I'll lay out what I know from practical experience, not just theoretical textbook concepts. The short answer – and I’m going to get into *why* this is – is that security constraints in a Tomcat web application are defined primarily within the `WEB-INF/web.xml` file, often referred to simply as the `web.xml`.

Let’s unpack that. You won't find security configurations, in the context of web applications, directly embedded in a top-level `web.xml` file alongside, say, the `catalina.sh` script in the Tomcat root. Those are configuration files for Tomcat itself, not for the deployed web applications. Instead, each web application, packaged as a WAR (Web Application Archive) file, carries its own `web.xml` located in the `WEB-INF` directory.

I remember vividly a project years ago where we had multiple applications running on the same Tomcat instance. Each needed its own authentication and authorization scheme, and things got pretty messy before we standardized. This highlighted exactly why Tomcat utilizes `WEB-INF/web.xml` for *application-specific* settings, including security constraints. It encapsulates the specifics of each application, keeping them independent and preventing conflicts.

Now, let’s drill into the structure. Within `web.xml`, the security configuration is handled via the `<security-constraint>` elements. These elements specify the following essential pieces:

*   **`<web-resource-collection>`:** Defines the resources (URLs or URL patterns) to which this security constraint applies. It uses the `<url-pattern>` tag.
*   **`<auth-constraint>`:** Defines the roles required to access those resources specified in `<web-resource-collection>`. It includes one or more `<role-name>` tags.
*   **`<user-data-constraint>`:** Specifies how data should be transmitted, particularly for sensitive information; mostly regarding whether to enforce HTTPS via `<transport-guarantee>` set to `CONFIDENTIAL`.

These are the foundational components. I've often found that neglecting any one of these can lead to gaping security holes. For instance, accidentally leaving out `<auth-constraint>` makes the associated resources publicly accessible. It's a common mistake that we learned to avoid after a few early security reviews.

To illustrate, let's look at some simple code examples. First, a basic example where we are requiring authentication to reach any URL under `/admin`:

```xml
<web-app>
  <security-constraint>
    <web-resource-collection>
      <web-resource-name>Admin Resources</web-resource-name>
      <url-pattern>/admin/*</url-pattern>
    </web-resource-collection>
    <auth-constraint>
      <role-name>administrator</role-name>
    </auth-constraint>
    <user-data-constraint>
        <transport-guarantee>CONFIDENTIAL</transport-guarantee>
    </user-data-constraint>
  </security-constraint>

  <login-config>
    <auth-method>FORM</auth-method>
    <form-login-config>
      <form-login-page>/login.jsp</form-login-page>
      <form-error-page>/login-error.jsp</form-error-page>
    </form-login-config>
  </login-config>

  <security-role>
    <role-name>administrator</role-name>
  </security-role>
</web-app>
```

In this configuration, all requests to URLs matching `/admin/*` require the user to have the `administrator` role. Crucially, this doesn't *implement* authentication. Tomcat itself will use the `login-config` to trigger an authentication process. The roles themselves are typically handled by a security realm set up in Tomcat’s server configuration (`server.xml`), or via an application-specific mechanism. The essential thing to note here is the clear declaration of the constraint in the web.xml.

Next, consider a scenario where you need different access requirements for different parts of your application. You can add multiple `<security-constraint>` elements. Let's say you want to restrict access to `/secure/*` to users with a `premium` role and also ensure https for this secure resource only.

```xml
<web-app>
  <!-- First Constraint for admin resource -->
  <security-constraint>
    <web-resource-collection>
      <web-resource-name>Admin Resources</web-resource-name>
      <url-pattern>/admin/*</url-pattern>
    </web-resource-collection>
    <auth-constraint>
      <role-name>administrator</role-name>
    </auth-constraint>
    <user-data-constraint>
        <transport-guarantee>CONFIDENTIAL</transport-guarantee>
    </user-data-constraint>
  </security-constraint>
  <!-- Second Constraint for premium resources -->
  <security-constraint>
    <web-resource-collection>
      <web-resource-name>Premium Resources</web-resource-name>
      <url-pattern>/secure/*</url-pattern>
    </web-resource-collection>
    <auth-constraint>
      <role-name>premium</role-name>
    </auth-constraint>
    <user-data-constraint>
         <transport-guarantee>CONFIDENTIAL</transport-guarantee>
    </user-data-constraint>
  </security-constraint>


  <login-config>
    <auth-method>FORM</auth-method>
    <form-login-config>
      <form-login-page>/login.jsp</form-login-page>
      <form-error-page>/login-error.jsp</form-error-page>
    </form-login-config>
  </login-config>

  <security-role>
    <role-name>administrator</role-name>
  </security-role>

  <security-role>
    <role-name>premium</role-name>
  </security-role>
</web-app>
```

This second example shows that security rules are evaluated top-to-bottom. If a user attempts to access `/secure/info.jsp`, Tomcat will enforce the second `<security-constraint>`. If the user doesn’t have the `premium` role, access will be denied. Conversely, if a user attempted to access `/admin/dashboard.jsp` the first `<security-constraint>` would be triggered.

Finally, if you want to explicitly allow unauthenticated access to certain resources (like static content), you would not include a `<security-constraint>` at all, or include a constraint that doesn't contain an `<auth-constraint>`. For example:

```xml
<web-app>

 <!--No constraint here allows access for everyone -->
 <security-constraint>
    <web-resource-collection>
      <web-resource-name>Public Content</web-resource-name>
      <url-pattern>/images/*</url-pattern>
      <url-pattern>/css/*</url-pattern>
    </web-resource-collection>
 </security-constraint>
 <!-- Normal constraint with auth requirements -->
  <security-constraint>
    <web-resource-collection>
      <web-resource-name>Admin Resources</web-resource-name>
      <url-pattern>/admin/*</url-pattern>
    </web-resource-collection>
    <auth-constraint>
      <role-name>administrator</role-name>
    </auth-constraint>
    <user-data-constraint>
        <transport-guarantee>CONFIDENTIAL</transport-guarantee>
    </user-data-constraint>
  </security-constraint>


  <login-config>
    <auth-method>FORM</auth-method>
    <form-login-config>
      <form-login-page>/login.jsp</form-login-page>
      <form-error-page>/login-error.jsp</form-error-page>
    </form-login-config>
  </login-config>

  <security-role>
    <role-name>administrator</role-name>
  </security-role>
</web-app>
```

In this scenario, content inside of `/images` and `/css` would be accessible to anyone without requiring login while resources under `/admin` remain protected with role restrictions.

For a deep dive into the Servlet Specification, which defines the functionality of `web.xml`, check out the official Oracle documentation for Java EE. Specifically, I recommend reading the section on the Web Application Deployment Descriptor within the Servlet API documentation, version applicable to your target Java EE/Jakarta EE specification. You’ll find in depth details of all the possibilities including nuances of transport guarantee and security roles. For a more general understanding of web application security, I recommend the OWASP (Open Web Application Security Project) resources, particularly their guides on authentication and authorization.

In summary, the `WEB-INF/web.xml` file is where you define the security constraints for a Tomcat web application. It’s the place to control access based on roles, and transport constraints. While Tomcat can handle some server-level configuration that’s generally separate from the application’s logic. It’s important to understand exactly where these application specific configuration resides and why. It’s crucial to understand the relationship between the `web.xml` and the Tomcat container, for effective security configuration.
