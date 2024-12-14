---
title: "Why is c# .net 6.0 is setting cookies on the wrong domain?"
date: "2024-12-14"
id: "why-is-c-net-60-is-setting-cookies-on-the-wrong-domain"
---

so, you’re seeing cookies go to the wrong domain in your .net 6.0 app. classic. i’ve been there, more times than i'd care to count. it’s rarely a problem with .net itself, more often it’s how the cookie is configured or how the requests are being handled, or the domain context it’s in, which leads to this. let's break this down from my experiences, focusing on those common pain points.

first, a bit of my history with cookie weirdness, because, well, we’ve all got those stories. years ago, i was working on this somewhat intricate microservices project. we had a frontend talking to a gateway, which would then pass requests to the actual services. everything worked fine locally, but when we deployed it to our dev environment, cookies would go to the gateway’s domain rather than the frontend’s. it took me a solid day of debugging and tcpdump to realize the gateway was setting the cookies itself after authenticating the users, as it was the authentication provider. my face was red for a week.

most of the time, a wrong domain for your cookies boils down to these causes:

1.  **domain mismatches in cookie settings:** this is the most frequent culprit. when setting a cookie, you can specify the `domain` attribute. if this attribute doesn't precisely match the domain of your application, the browser will either reject the cookie or not send it to the server when requested.

    let's say your app runs at `app.example.com`, and you set a cookie with the domain attribute as `.example.com`, that seems ok right? well... no, browsers see this as a different domain, and things get confused, and the cookies will not be sent to `app.example.com` in subsequent requests. the key is to be specific, or in some cases, it is necessary to make the cookie available to all subdomains with a wildcard `*.example.com`. remember the exact match is the safest and simplest configuration unless you are intentionally doing otherwise.

    in .net, this is usually configured when setting the cookie itself, often in the `httpcontext` or via `response.cookies`:

    ```csharp
    public void SetCookie(HttpContext context)
    {
        var cookieOptions = new CookieOptions
        {
            Domain = "app.example.com",
            HttpOnly = true,
            Secure = true, // always for production
            SameSite = SameSiteMode.Strict,
            Expires = DateTimeOffset.Now.AddHours(1)
         };

        context.Response.Cookies.Append("myCookie", "myValue", cookieOptions);
    }
    ```

    here, we're explicitly setting the `domain` to `app.example.com`. if this is different from where your app is running, expect weird behavior. this simple thing is one of the main mistakes.

2.  **incorrect request origin:** the browser security model plays a role too. if you are making a cross-origin request (for example, your frontend on `frontend.example.com` makes a request to `api.example.com`), cookies associated with `api.example.com` may not be automatically sent unless the appropriate cross-origin resource sharing (cors) headers are configured properly, or the domain attribute in the cookies is set up correctly. this can lead you to think the cookies are not there in the request, or in some cases, they are indeed set but they are not properly delivered to the correct place, that is your server.

    cors configurations can be a little verbose, especially if you need to allow multiple origins. you can use the options pattern in .net to configure cors. here is a common example, notice the origin being allowed:

    ```csharp
    // in startup.cs or program.cs

    builder.Services.AddCors(options =>
    {
      options.AddPolicy("MyCorsPolicy", policy =>
        {
           policy.WithOrigins("https://frontend.example.com") // your frontend
                 .AllowAnyHeader()
                 .AllowAnyMethod()
                 .AllowCredentials(); // allow cookies, very important
         });
    });

    // and, later, in a controller:
    [EnableCors("MyCorsPolicy")]
    public class MyController : ControllerBase { /* ... */ }
    ```
    remember that `.AllowCredentials()` is crucial when dealing with cookies and cross-origin requests. also, verify the allowed origin is the exactly expected one. i have lost quite some time figuring out that my front end was hitting the server with `www.frontend.example.com` instead of `https://frontend.example.com`. it’s silly, but it happens.

3.  **proxy and load balancer issues:** this one has burned me a few times. if you're using a reverse proxy (like nginx, apache or a cloud load balancer), the proxy might be intercepting and modifying headers including the `set-cookie` headers or request origins. it might even be setting its own cookies. this can completely throw off your application’s cookie management. also, these are the most complex scenarios to debug since you do not have total control over these infrastructure elements.

    when the application runs in a load-balanced scenario, and the request is redirected to different servers, cookie problems might appear since each app server might have its own setup, and might not be properly setting the cookies the same way. that’s why cookie configuration should be done consistently across all instances.

    here's an example of the problem with reverse proxies, sometimes a reverse proxy does the following:

    your app server does:

    ```http
    set-cookie: myCookie=value; domain=app.example.com;
    ```

    and your proxy, maybe for some reason modifies it to:

    ```http
    set-cookie: myCookie=value; domain=proxy.example.com;
    ```
     as you can imagine, the browser will deliver that cookie to the proxy, and not to your actual application domain. this is one of the hardest to debug, since you need to really go into the proxy logs to figure out this kind of behavior.

4.  **multiple cookie setters in the same http flow:** another common issue is inadvertently setting the same cookie more than once with slightly different configurations, often because of external dependencies or authentication modules. for instance, if you have an authentication middleware that also sets cookies, this might lead to conflict. browser behavior in these cases might be inconsistent, so the solution here is to track all cookie setters and figure out which is overwriting, and which is being overwritten.

5.  **browser specific rules:** each browser implements the cookie specification with some particularities. this means sometimes cookies behave differently in different browsers, some browser implement more aggressive cookie handling policies than others. it is a very deep topic that is not usually in the radar of the average software developer, and if you encounter an issue that is browser-specific, expect to delve into the specific details of your targeted browsers.

debugging this kind of issue can be a bit involved. what i usually do is:

*   **check the browser dev tools:** inspect the application tab or network tab to see the cookies being set and sent. pay close attention to the domain attribute, path and expires fields. look for inconsistencies. also the `samesite` attribute might be the key here.
*   **use a http debugging proxy:** tools like fiddler or charles can be very useful to inspect the actual headers (request and response) and trace where the cookies go. it might be difficult to isolate the problem just by the browser developer tools, and you might need to peek directly at the headers.
*   **review the configuration:** carefully examine all cookie-related configuration, like the `cookieoptions` object, cors policies, reverse proxy rules, and load balancers configuration. compare them across all environments.
*   **simplify the setup:** try to reproduce the issue in the simplest possible environment. sometimes you might need to isolate your application and run it in isolation from the load balancer or proxies. isolate the problem first and then bring back all the involved components one by one.
*   **enable logging:** logging the cookie setting and retrieval can be very helpful to track the origin and value of each cookie.

    ```csharp
     // simple cookie logging example
        public void SetCookie(HttpContext context)
    {
       var cookieOptions = new CookieOptions
       {
            Domain = "app.example.com",
            HttpOnly = true,
            Secure = true,
            SameSite = SameSiteMode.Strict,
            Expires = DateTimeOffset.Now.AddHours(1)
       };

        context.Response.Cookies.Append("myCookie", "myValue", cookieOptions);
        Console.WriteLine($"[cookie set] myCookie set to myValue, domain {cookieOptions.Domain}"); // use a proper logger instead
    }
    ```

as resources, i always recommend reading the rfc 6265, it’s the official cookie specification, it’s not exactly light reading, but it helps to understand the intricacies of the whole cookie concept. another extremely useful resource is the mdn documentation about cookies and http headers, it's a gold mine of information.

so, yeah, cookie issues, not really a day at the beach, but most of the time, the devil is in the details, pay close attention to the domain configuration and the request origin, and you’ll be back on track in no time. and remember when debugging, take small steps, it's the only way, sometimes, one must have a bit of patience, or otherwise you'll become... cookie monster.
