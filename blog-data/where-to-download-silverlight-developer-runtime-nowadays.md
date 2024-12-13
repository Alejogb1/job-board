---
title: "where to download silverlight developer runtime nowadays?"
date: "2024-12-13"
id: "where-to-download-silverlight-developer-runtime-nowadays"
---

Okay so you wanna know where to grab the Silverlight developer runtime huh yeah I get it that’s like asking for a rotary phone charger these days but I’ve been around the block a few times and trust me I’ve seen weirder stuff

Been there done that got the t-shirt and a few debugging scars to prove it I remember back in the day when Silverlight was the hotness it was like the cool kid on the block everyone wanted to hang out with that sweet XAML declarative UI goodness and the promise of cross-browser rich internet apps I even built a whole internal employee app for tracking expenses with it yeah I know right I feel old just saying it all the UI was custom built you could do some crazy stuff if you pushed it

But things change huh Microsoft moved on and Silverlight well it kinda faded into the background which is a technical term for it got deprecated and is no longer supported that’s the plain and simple truth Now getting a legit developer runtime its tricky not impossible though

Let me break it down for you no BS straight facts

First off forget about finding a shiny official download link from Microsoft itself it’s not happening they have literally pulled the plug The website it used to live on now points to some generic page that’s pretty much a tombstone for the technology So scrap that idea

Second avoid sketchy download sites offering “the real deal” its likely full of outdated runtime versions or worse malware You know the kind they get ya every time like those ads saying “Download free RAM!” Yeah no don’t even think about it just close that browser tab right now

So what’s the solution then You might ask well its a little bit like an archeological dig but its doable

Basically you're looking for archived copies of the runtime that people have preserved You'll mostly find them on non-official sites like tech forums or archive.org yeah that’s a thing the Wayback Machine the great internet vault it’s amazing what’s there if you dig around enough

Now before you go downloading random exes willy-nilly do a sanity check verify that the installer isn't just a trojan horse or something Use VirusTotal or other reputable scanner sites before you run it trust me it’s way better safe than sorry

Also you probably want to find the dev version not the end-user one the one you need will likely have the SDK part included as well

The specific versions I recommend you are going for are the latest versions 5 or the 5.1 if you can find it which will give you the full tooling to create your projects as well.

Now let me give you an example of some C# code you might find in a Silverlight project just so that you feel the tech and the era of what we are talking about

```csharp
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace MySilverlightApp
{
    public partial class MainPage : UserControl
    {
        public MainPage()
        {
            InitializeComponent();
            Loaded += OnLoaded;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
           // Just adding a simple background color to the grid
            MyGrid.Background = new SolidColorBrush(Colors.LightBlue);
            TextBlock myTextBlock = new TextBlock();
            myTextBlock.Text = "Hello Silverlight from a past time!";
            myTextBlock.Foreground = new SolidColorBrush(Colors.DarkRed);
            myTextBlock.FontSize = 24;
            MyGrid.Children.Add(myTextBlock);
        }
    }
}
```

This was basic UI manipulation adding a simple background color and a text block in the grid

And here's an example of how to use databinding which was a key part of Silverlight’s data driven architecture

```csharp
using System.Windows;
using System.Windows.Controls;
using System.ComponentModel;

namespace MySilverlightApp
{
    public partial class MainPage : UserControl
    {
        public MainPage()
        {
            InitializeComponent();
            DataContext = new MyDataModel();
        }
    }

     public class MyDataModel : INotifyPropertyChanged
    {
      private string _message;
      public string Message {
        get { return _message; }
        set {
            _message = value;
            OnPropertyChanged("Message");
        }
      }

       public MyDataModel(){
        _message = "Initial message from the data model";
      }

       public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }

}
```

And in XAML would be something like:

```xml
<UserControl x:Class="MySilverlightApp.MainPage"
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
    xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
    mc:Ignorable="d"
    d:DesignHeight="300" d:DesignWidth="400">

    <Grid x:Name="MyGrid">
        <TextBlock Text="{Binding Message}" FontSize="24" VerticalAlignment="Center" HorizontalAlignment="Center" />
    </Grid>
</UserControl>
```

These are very very basic examples but it shows how Silverlight worked with XAML and C# together

Now a bit more of a serious note this tech is no longer really a career prospect if you are trying to learn it for that it's something I wouldn’t personally recommend. It is great if you are trying to maintain existing legacy systems but if you are getting started you would be better off to go to modern frameworks like react or angular

Oh yeah almost forgot the resources for actually learning this thing back when it was supported I used a couple of books and one online resource

I’d suggest picking up Charles Petzold’s book “Programming Microsoft Silverlight” it's a classic and it was a great read back then It’s gonna be an old book but the principles are still valid

Also another book I liked was "Silverlight 4 in Action" by Pete Brown and Rob Relyea I learned the core concepts of the framework and many cool techniques for design and development

If you want to understand the underpinnings of XAML itself I would go for "WPF Unleashed" by Adam Nathan it explains the basic ideas that apply to many technologies that use XAML as a ui model its kinda related to Silverlight’s xaml but for a different framework WPF but still useful

And I also used the MSDN documentation back in the day it’s not available anymore but you can find its archived versions it’s a good thing to look at as it was a key piece of resource back in the day. Microsoft’s online documentation was actually pretty decent back then

So there you have it a way to get the developer runtime Its a bit of a hassle I know but trust me when you deal with legacy tech it never comes easy its never as simple as downloading the official installer from the company website I mean that would have been easy right Like a programmer finding a bug in their code the first time they compile it ahahaha just kidding there is always at least one error

And please remember safety first before running random executables on your machine

Hope this helps and good luck with your Silverlight adventures may your debugging sessions be short and your XAML compile without hiccups and one last thing if you find any really good Silverlight resources in the wild let me know I'm kinda collecting these sort of things for historic purposes and for the sheer nerdiness of it. You might never know when you need to dust off those old frameworks.
