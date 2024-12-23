---
title: "system drawing common is not supported on this platform on windows vs2019?"
date: "2024-12-13"
id: "system-drawing-common-is-not-supported-on-this-platform-on-windows-vs2019"
---

 so you're hitting that "System Drawing Common not supported" wall on Windows using VS2019 right Been there totally understand the pain First off you are right System Drawing Common is a notorious pain point it has a complicated history especially between different .NET versions and operating system compatibility Before diving into the fixes lets just clarify a couple of things

I've seen this problem crop up more times than I care to admit Back in the early .NET Core days we were all migrating our projects some of them depended heavily on older WinForms stuff especially legacy image processing routines and guess what System Drawing Common was the first stumbling block

What you are essentially encountering is the fundamental shift in how .NET handles cross-platform libraries System Drawing Common is heavily tied to the GDI+ graphics library in Windows This makes it inherently OS dependent and it's not designed for cross platform execution and that was the big design problem of the old .NET framework architecture

So in newer .NET versions including the ones used by VS2019 which usually means .NET Core or .NET 5+ or later and not the old .NET Framework it's not included by default This is an intentional move to make .NET truly cross-platform which is great for many applications but it leaves some of us with legacy code stranded

Basically it means that the traditional way of just referencing `System.Drawing.Common.dll` isn't sufficient anymore because it is no longer guaranteed to be present or operational in all environments that your code is running on This is what your error is indicating

The solution is more or less as follows you need to pull it in as a NuGet package if you were relying on it previously for your graphics stuff or find a suitable alternative There are a few approaches we can try here let's start with the common cases I'll show some code examples too

**First Approach The NuGet Package Route**

The easiest approach for many is to add the package directly You'll want to use NuGet Package Manager In Visual Studio or using the `dotnet` CLI This is what I normally do

```csharp
// CLI example
dotnet add package System.Drawing.Common
```

Or in Visual Studio
1. Go to your project's Solution Explorer
2. Right-click on your project
3. Select Manage NuGet Packages
4. Browse for `System.Drawing.Common`
5. Install the latest stable version

This approach installs the appropriate version of the library for the current target framework you're using But that doesn't solve everything There is still one very important detail that we have to consider if we are using Linux as target in the future

**Second Approach Checking your Target Framework and Compatibility**

Here is the part where we check that your target framework and compatibility are aligned because that is important in all scenarios

Sometimes the problem isn't that you didn't install the NuGet package but that the target framework is wrong or you are trying to run a project targeting .NET Framework in a newer .NET Core context You can solve this as follows :

1. **Check your project's target framework:** In your project file (`.csproj`) make sure you're targeting the correct framework that supports the library. This can be done by editing your .csproj file, which is the default file for .NET projects and contains project settings
```xml
<TargetFramework>net6.0</TargetFramework>  <!-- Example for .NET 6 change to the correct version in your case -->
```
 or use VS interface for the same in project settings or in the properties

2. **Ensure compatibility:** Even with the NuGet package there might be minor differences across platforms make sure to test on different OSes and to use if required the OS-specific code if needed

3. **Be aware of non-Windows platforms:** System Drawing Common will work on some non-Windows platforms (like some Linux distributions with a few extra installs) but it's not its ideal use case. If you need a true cross platform solution you might want to look into a more general graphics library like ImageSharp or SkiaSharp both these are excellent choices as replacement for System Drawing for example If you want to stick to Microsoft solutions there is also Windows Imaging Component (WIC) but that is more complex to use directly

**Third Approach: Migration to a Cross-Platform Alternative**

For truly cross-platform applications relying on `System.Drawing.Common` is really a dead end. My suggestion is to switch to one of the alternatives I talked about. Let's give a simple example of using SkiaSharp to achieve some basic operations that usually are made with the older `System.Drawing` library. Let's assume you want to just load an image resize it and save it back. Here's what the code would look like:

```csharp
using SkiaSharp;

public class ImageOperations
{
    public static void ResizeAndSaveImage(string inputPath, string outputPath, int width, int height)
    {
        using (var originalBitmap = SKBitmap.Decode(inputPath))
        {
            if(originalBitmap == null)
            {
                Console.WriteLine("Error could not decode the image");
                return;
            }
            using (var resizedBitmap = originalBitmap.Resize(new SKImageInfo(width, height), SKFilterQuality.High))
            {
                using (var image = SKImage.FromBitmap(resizedBitmap))
                using (var data = image.Encode(SKEncodedImageFormat.Png, 100))
                using (var outputStream = File.OpenWrite(outputPath))
                {
                    data.SaveTo(outputStream);
                }
            }
        }
    }
}
// Usage
// ImageOperations.ResizeAndSaveImage("input.jpg", "output.png", 200, 150);
```
In this example we use the SKBitmap to load and resize the image from disk and save the result back after converting it to SKImage.

Remember to add the SkiaSharp NuGet package using the same method we used for System.Drawing.Common

```bash
dotnet add package SkiaSharp
```

If we try the previous example with `System.Drawing.Bitmap` we would fail as expected on Linux

```csharp
using System.Drawing;
using System.Drawing.Imaging;
public class ImageOperations
{
    public static void ResizeAndSaveImage(string inputPath, string outputPath, int width, int height)
    {
        using (var originalBitmap = new Bitmap(inputPath)) // This will crash in most non-Windows environments
        {
             using (var resizedBitmap = new Bitmap(width, height))
            {
                using (Graphics graphics = Graphics.FromImage(resizedBitmap))
                {
                    graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    graphics.DrawImage(originalBitmap, 0, 0, width, height);
                }

                 resizedBitmap.Save(outputPath, ImageFormat.Png);
            }

        }
    }
}
// Usage
// ImageOperations.ResizeAndSaveImage("input.jpg", "output.png", 200, 150);
```

**Resources and Further Learning**

Here are some good resources to better understand this issue and its history and the alternatives for the .NET platform:

*   **"Pro .NET 5" by Andrew Troelsen:** This book has an entire section about cross-platform development in .NET and how the old legacy components are handled and why are being replaced with other libraries or approaches.
*   **Microsoft's Official .NET Documentation:** Always consult the official docs because they have all the information about cross platform development and best practices with the newer versions of .NET
*   **SkiaSharp and ImageSharp documentation:** Check the official docs and tutorials. Both libraries have comprehensive documentation and community support.

So to summarize your issue you need to make sure the NuGet package is installed target framework matches the supported version and that is compatible with your operating systems and if you need cross-platform support then you really should consider an alternative library such as SkiaSharp or ImageSharp. This is the path for better architecture and less headaches in the long run.

And as a joke before finishing here is the old programmer joke why do Java programmers wear glasses? because they don't C#
I hope this helps and good luck with your project!
