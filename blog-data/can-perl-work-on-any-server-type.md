---
title: "can perl work on any server type?"
date: "2024-12-13"
id: "can-perl-work-on-any-server-type"
---

Okay so you're asking if Perl can run anywhere right like on any server. I've been kicking around servers and coding in Perl since the late 90s and let me tell you it's a yes and a no kinda answer it's complicated but not that complicated. 

Short answer is yes perl can run on basically any server that matters even the weird ones but it's always "it depends". Let's unpack that.

You see Perl's got this beauty called the Perl interpreter it's the engine that takes your Perl code and turns it into something a computer understands. This interpreter is written in C which is as close to the metal as you can generally get. C compilers exist for a vast number of platforms so perl can be compiled for them.

Back in the day before virtualization became a thing I had a client they were running this ancient Solaris server thing. It was this beast of a machine with custom everything. We were using Perl for their backend web stuff and it was a complete pain honestly. Getting libraries and modules to compile and play nice was a whole day job. We're talking about getting some freaky database connector for an old Sybase server to play nice and then we had to deal with the dependencies of all that too. It was painful yes it worked but the installation process was always a headache every time.

So yeah Perl is platform agnostic kinda but it's not magical it needs a C compiler and that means a certain level of compatibility with underlying OS.

Here's the deal though the vast majority of servers you'll run into today use operating systems like Linux and Windows. These are the two biggies in web servers cloud environments etc. Perl has super great support for both.

* **Linux:** Linux is where Perl feels right at home you can install it easily using the system's package manager like apt or yum or dnf. There are usually precompiled versions available and you are set for a quick install. If you don't want the system version you can compile it from source no problem at all.

* **Windows:** On windows it's a bit more clunky you usually end up using something like ActiveState Perl or Strawberry Perl which bundle the interpreter and a bunch of handy modules. I personally use Strawberry Perl on my Windows machines it works great for development and running small scripts. I had some issues with the command line interface working with long pathnames. But that was resolved with a proper installation. It can get tricky if you need to handle complex file IO with many special characters but that's more on windows than perl.

* **macOS:** macOS is really like a cousin of Linux at a terminal level so it also is very easy to install and run perl code.

Now let's talk about the "it depends" part. 

The core of Perl is very portable. You can write code that'll run mostly without changes across different operating systems. However when you start using external libraries or modules that depends on system calls or native libraries that's where you might run into problems.

Let's say you use the `Win32::OLE` module which is used to interact with Windows COM objects. That module will only work on Windows. Now that's obviously a Windows-specific use case so you would expect to have issues on other platforms.

Here is some sample Perl code that runs fine both in Linux and Windows.

```perl
#!/usr/bin/perl
use strict;
use warnings;

my $message = "Hello Stackoverflow";
print $message . "\n";

my $count = 5;
for (my $i = 0; $i < $count; $i++) {
    print "Iteration: " . ($i+1) . "\n";
}
```
This very simple code will print a message and a loop in the console. No surprises here. It's what you expect it's going to do no matter the machine or system.

However here is the other side of the coin something that might need tweaks or will just not work.

```perl
#!/usr/bin/perl
use strict;
use warnings;
use Win32::OLE;

my $excel = Win32::OLE->new('Excel.Application') or die "Could not start Excel";
$excel->{Visible} = 1;
my $workbook = $excel->Workbooks->Add();
my $worksheet = $workbook->Worksheets(1);
$worksheet->{Cells}(1,1) = "Hello Excel";
```

This code interacts directly with Windows and that will clearly not work on anything other than a Windows environment. You know the good old Windows things. This is the biggest issue when it comes to portability. It's the libraries not the core language.

The same applies to some system calls if your code interacts directly with low level system features like process management or network interfaces that might be different between systems you will have to write specific code.

I remember one time when dealing with specific network device drivers. I was trying to interface some custom network adapters in a embedded system using Perl and the network stack API was wildly different from the standard Linux stack. I had to write this custom module to directly use ioctl commands that was not even portable between different Linux kernels. I ended up abandoning perl for C in that particular project. That's the other side of the coin.

Another issue when people claim that Perl can run "everywhere" is the "where" part itself. Can Perl run on a toaster yes in theory. The problem comes when you need to compile the perl interpreter for the toaster processor. It's generally not worth it. So the "anywhere" usually means "any common server or PC environment".

One of the reasons why Perl was so good in the olden days when we didn't have all these cloud things is because it was a single script that you could copy and paste to a linux server and it would work. Most of the time that is. That kind of versatility is less common now when you have virtualized environments and docker and all sorts of containers.

The core language is still very useful in modern computing as a glue language or a quick scripting language to automate repetitive tasks.

If you need to check system info this is a snippet that uses perl system tools.

```perl
#!/usr/bin/perl
use strict;
use warnings;

my $os = `uname -s`;
chomp $os;
my $cpu = `cat /proc/cpuinfo | grep 'model name' | head -n 1`;
chomp $cpu;

print "OS: $os\n";
print "CPU: $cpu\n";
```
This code will print the operating system and a model of the processor however since `cat /proc/cpuinfo` is not valid in windows that part will clearly not work.

If you are targeting portability you should avoid such things.

**So how can you deal with portability issues?**

*   **Abstract your code:** Try to isolate system-specific parts into separate modules or functions. This makes it easier to make changes when you port to different operating systems.

*   **Use standard modules:** Whenever possible use modules that are portable like `File::Spec` for path manipulation instead of hard coding paths with forward or backslashes which are different depending on the OS. 

*  **Test thoroughly:** Always test your code on different operating systems to find potential issues early.

*   **Containerization:** Using containers like Docker you can package your Perl application and all its dependencies in an image. This will ensure it runs the same no matter the environment it is deployed to. There are also alternatives to containers like virtual machines that also can help to reproduce the environment in all machines.

*  **Modern Perl:** If possible use newer Perl versions. There have been many improvements in handling portability over the years. The older version of perl you use the more trouble you will have. I'm joking but really I've had to maintain some old perl code that only works on very specific perl versions and it's like pulling teeth every single time.

**Recommended Resources**

*   *Programming Perl* by Larry Wall the author of Perl. This book is like the Bible of Perl.
*   *Modern Perl* by chromatic. A more modern approach to Perl programming that emphasizes current best practices.
*   Perl documentation at `perldoc.perl.org`. Always a great source of truth for the official Perl documentation.

So to answer your question yes perl runs anywhere that matters but you have to be aware of the limitations of portability. You have to be smart and use the appropriate techniques. It's the same for any language that compiles for different operating systems. So go out there and write some awesome perl scripts.
