---
title: "start process msiexec powershell scripting?"
date: "2024-12-13"
id: "start-process-msiexec-powershell-scripting"
---

Alright so you wanna kick off an msiexec process using PowerShell right I've been down that road more times than I care to remember trust me it's not always a smooth ride I remember my early days battling with MSI installers it felt like wrestling a greased pig sometimes

Okay so first things first why are you even trying this You've probably got a Windows application that needs to be installed or maybe you are doing some kind of unattended deployment thing and msiexec is your ticket to the show Fine I get it That's how it is sometimes

Now you're thinking PowerShell because well let's be honest it's the Swiss Army knife of Windows automation It's not as pretty as some other things but it gets the job done most of the time So let's break down how you do this thing effectively avoiding the common pitfalls along the way

The core of your operation is the `Start-Process` cmdlet That's your workhorse for launching any process including msiexec The basic syntax looks something like this

```powershell
Start-Process -FilePath "msiexec.exe" -ArgumentList "/i path\to\your.msi /qn" -Wait
```

Let's dissect this `FilePath` this tells PowerShell which program to start in this case it's `msiexec.exe` pretty straightforward Next we have `ArgumentList` this is where you pass in the command-line arguments that `msiexec` needs to do its job For example `/i` specifies that we are installing something followed by the path to the MSI file itself And then there is `/qn` which tells msiexec to run the install completely silently no dialog boxes no user interaction just straight to business

The `-Wait` parameter is optional but often necessary This tells PowerShell to wait for the `msiexec` process to finish before moving on to the next line of code This is especially important if your script depends on that installation completing otherwise you might get weird errors later on down the line trust me I've seen it all

Here is a quick one that also includes logging

```powershell
Start-Process -FilePath "msiexec.exe" -ArgumentList "/i path\to\your.msi /qn /l*v C:\install.log" -Wait
```

In this version we have added `/l*v` which tells msiexec to create a detailed log file called `install.log` in this case in the root of `C` drive Logging is your best friend when things go wrong I mean really it's like a bread crumb trail in a forest of error messages

Now let's talk about another thing that I had more than my fair share of time wasted on when you get this path to your MSI file be extra extra careful make sure it's correct because PowerShell can be a little unforgiving if you misspell something or forget the file extension trust me I have spent days trying to figure out a misspelled file location

Here is a more elaborated sample using variables because that's what we do to avoid hardcoding values

```powershell
$msiPath = "C:\path\to\your.msi"
$logPath = "C:\install.log"
$arguments = "/i `"$msiPath`" /qn /l*v `"$logPath`""

Start-Process -FilePath "msiexec.exe" -ArgumentList $arguments -Wait
```

Here we introduced a few variables `$msiPath` which holds the location of your MSI package `$logPath` keeps your log file location and then a `$arguments` where all your msiexec parameters are kept You can then directly use this `$arguments` variable in your `Start-Process` command This is better because when you need to change any settings you do it in one place instead of many

Let me tell you a little story when I first started automating this I was using the absolute simplest command just like the very first one and I thought I was king of the hill. One day the installers were just failing mysteriously I mean it was chaos turns out some MSI installers had custom actions and sometimes they'd hang forever if they didn't run from a path they liked or something. I had to go diving into the MSI internals and learn that I should sometimes specify properties and that sometimes the installation has a user context that is not the same as the script so yes. It was a hard lesson but now I learned and I can't forget it

And yeah don't even think about it if you need to do system level installs or things that require admin elevation you absolutely must run this PowerShell script with administrator privileges or you are just gonna have a bad time trust me I was there too. PowerShell is not magic it won't let you do stuff you are not allowed to do by the operating system

Also one more tip that will save you hours of debugging hell don't assume that the install will be successful always check the exit code of the msiexec process This will tell you whether the installation succeeded failed or encountered some other error

You can capture the exit code this way:

```powershell
$process = Start-Process -FilePath "msiexec.exe" -ArgumentList "/i path\to\your.msi /qn" -Wait -PassThru
$exitCode = $process.ExitCode

if ($exitCode -eq 0) {
    Write-Host "Installation successful"
} else {
    Write-Host "Installation failed with exit code: $exitCode"
}
```

Here we used `-PassThru` which allows us to get the process object and we can then extract the exit code and then we can check for exit code `0` to know if things went well or not If you see other numbers you must search them because sometimes it can be a system level issue you are facing

Now about the resources I won't just throw a bunch of links at you No instead I will suggest some proper material if you want to go deeper First off "Windows Installer XML Toolset (WiX) Tutorial" by Nick Ramirez it's a bit of an oldie but a goldie and if you really want to understand how MSI files actually work and how to create and work with them then you should start with that. Also have a look at "Windows Server Automation with PowerShell Cookbook" by Jonathan Noble It has a lot of practical examples and solutions for common automation scenarios including working with installers like `msiexec` and it will guide you to understand what I just said. There is also one more I can recall and it is "PowerShell in Action" by Bruce Payette It dives deep into the core concepts of PowerShell and helps you understand the underlying mechanism of what you are doing with PowerShell especially about the pipeline which is not the case here but it is necessary anyway for your future PowerShell endeavors

And oh there's a joke I have for you a programmer walks into a bar orders a beer and then asks what if it's not working because I am always expecting something to go wrong

So there you have it all my experience summarized Hopefully this will give you a good head start and keep you away from the land of msiexec nightmares it's not too difficult once you understand the basics and use the right tools and some decent logging. Just remember to be extra careful with file paths always double check your arguments and do not forget to check the exit codes and if you do these things you will be just fine. Happy automating and may your installers never fail because that would be a very sad story.
