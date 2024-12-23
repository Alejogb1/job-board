---
title: "wix embeddedchainer examples?"
date: "2024-12-13"
id: "wix-embeddedchainer-examples"
---

 so you're digging into Wix's EmbeddedChainer right I get it I've been there done that bought the t-shirt and probably accidentally deleted the source code a few times let's talk about that mess

First off if you're asking for *examples* you're probably hitting a wall with the official documentation it's like a treasure map with half the landmarks missing Yeah the concept is pretty straightforward right you want to create a chain of actions within an installer usually related to custom actions or sequencing but actually doing it especially with Wix's EmbeddedChainer is like herding cats blindfolded

My first tango with EmbeddedChainer was back in 2015 I think or was it 2016 time blurs when you spend too much time battling installer technologies I was tasked to build a custom installer for an internal tool that needed a very specific sequence of configurations to be done before the app itself installed and during uninstall the opposite dance We tried using just custom actions at first but that turned into spaghetti code real quick

I remember one particularly dark afternoon with 8 or so failed builds because some custom action was firing out of order and corrupting some registry key We needed finer grained control we needed a chain we needed EmbeddedChainer and we needed more coffee I'm not sure which one helped the most to be honest

The big catch with EmbeddedChainer is this it's about embedding a sequence of actions inside the main MSI not just triggering stuff it works off these *chained packages* which are just MSI files that are sequenced by the main installer it's a kind of nested installer situation

Let’s start simple here’s a minimal example of defining the embedded chainer in a wix project file you can see that i am using the heat harvested project for simplicity in a real world scenario this would need to be your existing project

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Product Id="*" Name="MyProduct" Language="1033" Version="1.0.0.0" Manufacturer="MyCompany" UpgradeCode="PUT-YOUR-GUID-HERE">
        <Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine" />

        <MediaTemplate EmbedCab="yes" />
	<Directory Id="TARGETDIR" Name="SourceDir">
	    <Directory Id="INSTALLFOLDER" Name="MyProduct">
		<Component Guid="PUT-YOUR-GUID-HERE-FOR-COMPONENT">
			<File Id="ExampleFile" Name="example.txt" Source="example.txt" />
		</Component>
	    </Directory>
	</Directory>
		<ComponentGroup Id="ProductComponents">
			<ComponentRef Id="MyProduct.Component"/>
		</ComponentGroup>

        <Feature Id="Complete" Title="Complete Feature" Level="1">
            <ComponentGroupRef Id="ProductComponents" />
		</Feature>
	<EmbeddedChainer>
	    <MsiPackage SourceFile="YourChainedPackage1.msi"/>
        <MsiPackage SourceFile="YourChainedPackage2.msi"/>
	</EmbeddedChainer>
    </Product>
</Wix>
```

The key parts here the `<EmbeddedChainer>` element and `<MsiPackage>` elements This tells Wix look I have some MSI files they should be run in this order and embed them inside the main MSI The `SourceFile` attribute should be the relative or absolute path to your chained MSI file
And do not forget to add a dummy file to be able to compile correctly the project

here is an example of an example file to be used for compiling the project above

```
I am an example file
```

Now for the crucial part the chained packages themselves These aren't just random MSIs you need to be thinking about what *they* do in relation to the main package often these are the custom action parts that need careful sequencing

So let's say you want to configure some settings or perform some tasks before the actual main app gets installed a good example is a third party dependency installer

Here's a very basic example of an MSI that is the chained package it does absolutely nothing but serves to demonstrate how it should be created:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
	<Product Id="*" Name="MyChainedPackage1" Language="1033" Version="1.0.0.0" Manufacturer="MyCompany" UpgradeCode="PUT-YOUR-GUID-HERE">
		<Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine" />

		<MediaTemplate EmbedCab="yes" />
	    	<Directory Id="TARGETDIR" Name="SourceDir">
	   	     <Directory Id="INSTALLFOLDER" Name="MyChainedPackage1">
			<Component Guid="PUT-YOUR-GUID-HERE-FOR-COMPONENT">
	 	         <File Id="ChainedExampleFile" Name="chainedexample.txt" Source="chainedexample.txt" />
			</Component>
	    		</Directory>
		</Directory>
		<ComponentGroup Id="ProductComponents">
		    <ComponentRef Id="MyChainedPackage1.Component"/>
		</ComponentGroup>

		<Feature Id="Complete" Title="Complete Feature" Level="1">
		    <ComponentGroupRef Id="ProductComponents" />
		</Feature>
	</Product>
</Wix>
```
And here is an example file for this chained msi package:

```
I am the chained file example
```

Notice a few things each chained package is itself a complete MSI it has its own Product tag its own Directory structure etc This is important because it means they can be installed and uninstalled independently if needed that's the whole point
The main MSI installer will trigger an install of the chained package which can trigger install of any component in that MSI package.
One of the first mistakes I made with these chainer packages was trying to add the components directly to the main MSI which led to a very bad experience of confusion. Do not do that.

When it comes to more complex scenarios involving custom actions within the chained packages you might need to add a custom action entry point in the main MSI's `InstallExecuteSequence` and `UninstallExecuteSequence` tables and use the `EmbeddedUI` property to control the UI of the chained packages here is an example of this:

```xml
<InstallExecuteSequence>
<Custom Action="EmbeddedUI_ChainedPackage1" Before="InstallFinalize">NOT Installed</Custom>
</InstallExecuteSequence>

<UninstallExecuteSequence>
	<Custom Action="EmbeddedUI_ChainedPackage1" Before="RemoveFiles">NOT Installed</Custom>
</UninstallExecuteSequence>
```

The `EmbeddedUI_ChainedPackage1` is defined inside the MsiPackage tag in the main MSI using `EmbeddedUI` property this instructs the main MSI to call the embedded MSI

Now this one here is a bit of a rabbit hole because you can configure all sorts of things like conditional installation for these nested MSIs but let's stick with the basics for now

What is also important is how to debug it? Well the truth is the debugging process is not optimal The best way is to create detailed logging for all MSIs both the main one and the chained one If you can see each custom action and its result this becomes easier.
Also you can enable verbose logging for the main installer and try to see what is going wrong in the chained packages installation process
Keep in mind that sometimes a failing of a chained package installation is going to break the entire installation this is not a bad thing per se since it is important to correctly configure everything for an ideal install

As for resources I didn't really use books that were specific to *Wix EmbeddedChainer* when I first started working with it to be honest It was more about combing through the WiX documentation which can be found in the official wix documentation site. There are also various blog posts and articles scattered around the web that go deeper into specific use cases and the wix mailing list is also a must to subscribe and follow for future changes in the tech landscape of wix and installer technologies.

One final tip avoid the temptation of using it as the solution to everything it might not be the right tool in some cases. Do not over use custom actions in the main msi and try to use the embedded chainer as the solution to complex workflows because it might also not be the correct tool. Sometimes a simple custom action with a sequence might be all that you need

Oh and one last thing did you hear about the programmer who couldn't debug his code? He kept saying "I haven't got a clue!"

I hope this helps you navigate the sometimes confusing world of embedded chainers it definitely gave me headaches back in the day and hopefully some of that is useful
