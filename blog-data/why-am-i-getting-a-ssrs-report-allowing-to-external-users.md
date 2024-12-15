---
title: "Why am I getting a SSRS report allowing to external users?"
date: "2024-12-15"
id: "why-am-i-getting-a-ssrs-report-allowing-to-external-users"
---

hey, so, dealing with ssrs reports getting out to external users, that's a situation i've tangled with before, and it's got a few common culprits. it's less about some magical setting and more about understanding how ssrs handles permissions and access, especially when you're not dealing with internal domain users.

let's talk through what likely happened, based on my past run-ins with similar stuff. first off, you have to consider that ssrs is built around the assumption that report consumers are typically on the same network, or at the very least, authenticated via a domain. when you're exposing reports to external users, that whole model breaks down, and you need to bridge the gap. this is almost always where people start experiencing headaches and hair loss. believe me, I can tell you stories about that from my past life, lol.

so, here's the breakdown of things you wanna be thinking about. generally, the easiest way to start tackling this is to think step by step. when you are trying to show a report to an external user the way they access, you should think about the layers of technology behind it. you have ssrs, which is the software that actually serves the reports. then you have the webserver that hosts it, usually iis, and finally the users's browsers and networks. any of these can be the problem, even your own network configurations.

authentication is the key piece. you're not going to be able to give ssrs a list of usernames and passwords like you can do with windows domain authentication. it was a big deal back in the day when I had a client who insisted we could reuse the company's passwords for every user. that was scary to even think about doing and the amount of emails I got after I shut that down were enormous. you really want to avoid that and also avoid setting up individual windows domain accounts for external users, it just doesnt scale and it becomes a headache to manage. that's what you should be looking at, ways of authentication. you should be aiming to use forms authentication or a more modern method like api keys if it makes sense.

here is how you would generally configure the basic forms authentication for ssrs:

```xml
<configuration>
    <system.web>
        <authentication mode="Forms">
            <forms loginUrl="login.aspx" name=".ASPXFORMSAUTH" timeout="30" />
        </authentication>
        <authorization>
            <deny users="?"/>
        </authorization>
    </system.web>
</configuration>
```

this configuration tells the webserver that it will use the forms authentication method. this specific example is very basic and is normally used with a login page with forms in a web application.

in terms of ssrs, in most cases, what happens is that you have the webserver or iis configured to use an authentication type, such as windows authentication or basic authentication. if you use windows authentication it only works for domain users. the problem is that if the end-user is outside the domain network, well they wont be able to authenticate. they wont be able to see the report, or anything at all. the same will happen with basic authentication which requires a username and password. the only advantage of basic is that it doesn't require a domain but its bad for security because it's basic. not recommended at all.

the other important thing is the url used to reach the reports, and how ssrs configures permissions for the report itself. if you are directly exposing the url to the report like `http://yourserver/reportserver/yourreport` and the security configuration of ssrs is configured to require the user to be authenticated in windows, you will see those 401 error or a prompt for login that fails.

once you have configured ssrs to accept a different type of authentication besides windows, like forms or even integrated authentication with a system like oauth, you will have to also configure the report permissions itself. if you expose a report to someone through the browser, ssrs, will normally ask for the user credentials based on its configured authentication. however, the report itself also has permissions, in a different configuration area.

let me show you a snippet that represents a report's access control configuration, using its xml structure. this snippet shows a security policy for the report:

```xml
<Security>
    <Policies>
        <Policy>
            <Group>Everyone</Group>
            <Role>Browser</Role>
        </Policy>
    </Policies>
</Security>
```

this snippet normally is part of the report xml file and it shows that the "everyone" group has "browser" access. if this is the case, that means that anyone who can authenticate to ssrs can then read the report. so authentication is the first step but permissions are also important, you should always consider permissions and authentication. this is not set in the web config file, you need to use the ssrs configuration manager to set these, or do it manually in the xml files or the ssrs web configuration page.

if you are using the default ssrs configuration, you are probably using windows authentication and the default ssrs url, which is similar to the url i showed before. what this implies is that only users in the domain can see the reports because the web server (iis) is only accepting authenticated users from the windows domain configured. that is why external users, even if they have accounts in your system, wont be able to log in. they just wont have the appropriate domain identity and authentication.

also, if you are accessing the reports with a url like `http://yourserver/reportserver/yourreport` you are probably accessing the report directly, which also is the default url in ssrs. that directly exposes ssrs to the public. the common thing to do is to have a web application that renders the reports instead of allowing users to directly access ssrs. this adds a layer of security and makes things a lot easier.

when I had to solve this for a company, i ended up creating a small web app that did the authentication and then used the ssrs web services to call and render the report. this was a common practice for many years. the url from the report then changed to `http://yourapp/reports/yourreport` instead of directly going to `reportserver`.

here is an example of how to programatically access a report from the ssrs web service:

```csharp
using System;
using System.Net;
using System.Web.Services.Protocols;
using ReportExecution; //ssrs web service dll

public class ReportServiceExample
{
   public static void Main()
    {
      ReportExecutionService rs = new ReportExecutionService();
      rs.Credentials = new NetworkCredential("username", "password", "domain");
      rs.Url = "http://yourserver/ReportServer/ReportExecution2005.asmx";

      try
      {
         byte[] result = rs.RenderReport("yourreport", "pdf", null, null);
          //save result in file
      }
      catch (SoapException ex)
      {
            //error handling
      }
    }
}
```

in this example, you would create a .net application to connect to ssrs using its web service and then render the report using its functions. this would require to authenticate with a username and password that has permissions in ssrs. also you can use other security methods besides username and password like api keys. you can also configure the rendering to be in different file formats, this example used the pdf format but there are other ways. this method adds an extra layer of security, besides having total control on how to display, or even filter the data before rendering.

in sum, start looking at your ssrs configuration and the web server configuration. make sure you have the correct authentication configured and you are not relying on default windows domain authentication. also check your report permissions and if you are directly exposing ssrs to the internet, you might need to build an extra security layer like I did. it's not an easy fix, but knowing where to look will save you some debugging time.

as far as resources go, for deep-dives into ssrs architecture and security, you might want to look into "microsoft sql server reporting services step by step" it was a great book that I used back in the day, it might be outdated but the main concepts are there. for a better understanding of authentication, i recommend the "oauth 2.0 in action" book. it explains how modern authentication systems work, and it is a recommended read for every developer. these resources should give you a more solid foundation for solving these kinds of ssrs problems.
