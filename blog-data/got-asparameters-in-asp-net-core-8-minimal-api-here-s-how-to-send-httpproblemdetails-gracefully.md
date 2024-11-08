---
title: "Got [AsParameters] in ASP.NET Core 8 Minimal API? Here's how to send HttpProblemDetails gracefully"
date: '2024-11-08'
id: 'got-asparameters-in-asp-net-core-8-minimal-api-here-s-how-to-send-httpproblemdetails-gracefully'
---

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<RouteHandlerOptions>(options =>
{
    options.ThrowOnBadRequest = true;
});

builder.Services.AddProblemDetails(opt => opt.CustomizeProblemDetails = context =>
{
    if (context.Exception is BadHttpRequestException ex)
    {
        context.HttpContext.Response.StatusCode = 400;
        context.ProblemDetails.Detail = ex.Message;
        context.ProblemDetails.Status = 400;
    }
});

var app = builder.Build();

app.UseExceptionHandler("/error");

app.Map("/error", (HttpContext context) =>
{
    var exception = context.Features.Get<IExceptionHandlerFeature>()?.Error;
    if (exception is BadHttpRequestException badRequestException)
    {
        return Results.BadRequest(new ErrorResponse
        {
            Error = "Invalid Parameter",
            Details = badRequestException.Message
        });
    }

    return Results.Problem("An unexpected error occurred.");
});

app.MapGet("/problem", Problem);

app.Run();

public class MyParameters
{
    public int Abc { get; set; }
}

private static Task<IResult> Problem([AsParameters] MyParameters p)
{
    return Task.FromResult(Results.Text("Hello, World!"));
}

public class ErrorResponse
{
    public string Error { get; set; }
    public string Details { get; set; }
}
```
