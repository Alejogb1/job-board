---
title: "adding to time variable in windows cmd script?"
date: "2024-12-13"
id: "adding-to-time-variable-in-windows-cmd-script"
---

Alright so you're wrestling with time manipulation in Windows cmd scripts huh I've been there it's a rabbit hole trust me

Let's break this down I see your question as wanting to basically do addition to a time variable within a batch file Not a complex thing in theory but the batch scripting syntax being what it is well it can be a bit finicky I'm not going to sugarcoat it I've spent hours debugging what seemed like trivial time operations in batch files back in my early coding days

Okay first off Windows cmd itself isn’t exactly known for its amazing date-time handling capabilities It's definitely not a Python or a JavaScript when it comes to this kind of stuff you're stuck with what `date` and `time` throw at you plus some rather ugly workarounds We're gonna make it work though

The core problem is that `time` gives you strings not easily manipulated numbers So you can’t just `set newTime=%oldTime% + 1` because `time` strings like 10:30:45 are just treated as plain text by `set` not as numeric values

So what's the trick? You have to dissect the hours minutes and seconds into numeric parts then you do your addition and then you reform the time string It involves a bunch of variable parsing and math that batch is not great at I mean seriously have you ever tried doing complex floating point operations in batch? Don’t! You’ll only find pain and anguish

Here’s a basic way that I usually use that doesn't require external tools just plain old CMD commands:

```batch
@echo off
setlocal

:: Get current time
echo Current Time is %time%
set currentTime=%time%

:: Parse the time into hours minutes and seconds
for /f "tokens=1-3 delims=:" %%a in ("%currentTime%") do (
  set hours=%%a
  set minutes=%%b
  set seconds=%%c
)


:: Calculate the new time adding say 15 seconds
set /a seconds+=15

:: Handle seconds overflow
if %seconds% geq 60 (
    set /a minutes+=%seconds%/60
    set /a seconds=%%seconds%%60
)

:: Handle minutes overflow
if %minutes% geq 60 (
    set /a hours+=%minutes%/60
    set /a minutes=%%minutes%%60
)

:: Handle hours overflow (day rollover is a little more complex to handle correctly and is outside of scope of this)
if %hours% geq 24 (
	set /a hours=%%hours%%24
)

:: Format the output time string
set newHours=0%hours%
set newHours=%newHours:~-2%
set newMinutes=0%minutes%
set newMinutes=%newMinutes:~-2%
set newSeconds=0%seconds%
set newSeconds=%newSeconds:~-2%

set newTime=%newHours%:%newMinutes%:%newSeconds%
echo New time %newTime%

endlocal
```

This script will grab the current time parse it into components then add 15 seconds it'll handle the overflows for you (minutes to hours seconds to minutes) but it's not exactly the cleanest solution I know I've had to use this kind of code a million times in my first gig back in the day trying to automate server deployments with just batch files We didn’t have fancy PowerShell at that time haha I remember spending a weekend on a batch script that had to do server backups at specific times it was not pretty I mean not pretty at all

Now that I think of it handling overflows like that is tedious I've done so many time and date manipulations in various different projects I even wrote some stuff in C once to deal with those kinds of operations much more efficiently I can tell you if you have to handle more complex date time operations maybe you should look at using languages like that instead of batch scripts I mean for things that have to run on a windows machine and you just want the simplest solution cmd batch is fine but don't go too far with it if you have the possibility don't use this for critical systems you would be better off using more robust alternatives like you know Python or PowerShell

Here is another way using some mathematical trickery I did during that awful time I had to deal with this in batch file form It is less verbose I guess

```batch
@echo off
setlocal

:: Get current time
echo Current Time is %time%
set currentTime=%time%

:: Parse the time into total seconds
for /f "tokens=1-3 delims=:" %%a in ("%currentTime%") do (
  set /a totalSeconds=(%%a*3600) + (%%b*60) + %%c
)

:: Add seconds
set /a totalSeconds+=15


:: Calculate new hours minutes and seconds from total seconds
set /a newHours=totalSeconds/3600
set /a tempSeconds=totalSeconds%%3600
set /a newMinutes=tempSeconds/60
set /a newSeconds=tempSeconds%%60

:: Handle hours overflow
if %newHours% geq 24 (
	set /a newHours=%%newHours%%24
)


:: Format the output time string
set newHours=0%newHours%
set newHours=%newHours:~-2%
set newMinutes=0%newMinutes%
set newMinutes=%newMinutes:~-2%
set newSeconds=0%newSeconds%
set newSeconds=%newSeconds:~-2%

set newTime=%newHours%:%newMinutes%:%newSeconds%
echo New time %newTime%

endlocal
```

It still does not cover all the cases if you want to correctly handle dates across days using only cmd batch file you’ll need to add more parsing and more logic It can be done I mean I did it but I wouldn't wish that experience upon my worst enemy Seriously if you want to handle more advanced date operations or you have to consider other time zones or daylight saving time forget about this and use something that does these things easily and correctly for instance any language with a decent standard library will make it so much easier

But hey for a simple time addition these methods should work just fine I mean it is not super fast but it does the job it's like trying to use a screwdriver to hammer a nail it can be done but there are better tools for the job I’d say

Oh and by the way if you are thinking about using that as a reliable way for timing some processes think twice batch scripts aren’t known for their precision and its time is not very reliable for timing very short tasks but that's a topic for another question

And finally here is a example to show you another approach to achieve the same thing without using complex math operations it is more verbose though

```batch
@echo off
setlocal

:: Get the current time
echo Current time is %time%
set currentTime=%time%

:: Function to add seconds to a time string
:addSeconds
setlocal
    set timeString=%1
    set secondsToAdd=%2

    for /f "tokens=1-3 delims=:" %%a in ("%timeString%") do (
        set hours=%%a
        set minutes=%%b
        set seconds=%%c
    )
    set /a totalSeconds=(%hours%*3600) + (%minutes%*60) + %seconds%
    set /a newTotalSeconds=%totalSeconds% + %secondsToAdd%

    set /a newHours=newTotalSeconds/3600
    set /a tempSeconds=newTotalSeconds%%3600
    set /a newMinutes=tempSeconds/60
    set /a newSeconds=tempSeconds%%60

	:: Handle hours overflow
	if %newHours% geq 24 (
		set /a newHours=%%newHours%%24
	)

    set newHours=0%newHours%
    set newHours=%newHours:~-2%
    set newMinutes=0%newMinutes%
    set newMinutes=%newMinutes:~-2%
    set newSeconds=0%newSeconds%
    set newSeconds=%newSeconds:~-2%

    endlocal & set result=%newHours%:%newMinutes%:%newSeconds%
    exit /b 0

:: Call the function with the current time and 15 seconds
call :addSeconds "%currentTime%" 15
set newTime=%result%

echo New time is %newTime%

endlocal
```

This time it uses a function to make the code more readable it’s still essentially doing the same parsing and arithmetic but the function makes it a bit easier to call multiple times if you ever have to

Alright now about resources you asked for instead of links If you really need to dig into the inner workings of batch files and low level windows stuff I’d recommend a couple of oldies but goodies You can find them in used book stores or libraries

*   **Microsoft Windows Script Host Programmer's Reference**: This is an old book but it contains the bible on how the internals of Windows scripting work It will tell you all about how cmd scripts are processed and how the internals work not specifically for date and time but a must read if you want to deeply understand Windows scripting
*   **Advanced Batch Scripting**: You can find this title in a few places or in a couple of books with similar titles This will give you a more comprehensive guide to batch file tricks and techniques that you might find useful if you still need to use batch scripts It focuses more on the practical aspects of scripting and it will teach you some ways to cope with the limitations of batch programming

I mean let's be honest dealing with time in batch is like trying to make a rocket ship with duct tape and cardboard boxes You can kinda get it to work but it’s gonna be janky and unreliable and eventually it will explode in your face So keep it simple use something better when you can and that’s the end of my rant about time in batch If you get what I mean

Good luck and may your batch files never fail you unless you try to use them for something too complex then they will fail miserably
