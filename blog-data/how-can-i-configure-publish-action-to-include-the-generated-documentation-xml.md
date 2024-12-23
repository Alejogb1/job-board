---
title: "How can I configure 'Publish' action to include the generated documentation XML?"
date: "2024-12-15"
id: "how-can-i-configure-publish-action-to-include-the-generated-documentation-xml"
---

alright,  you're after getting your generated documentation xml files included when you hit 'publish', specifically when using a visual studio project, i'm assuming. this isn't uncommon, i've banged my head against this more than once. it's one of those things that's frustratingly simple once you get the hang of it, but feels like a brick wall when you're starting out.

i’ve been doing this sort of development for close to two decades now, and i remember my first real encounter with this. it was back in the days of .net framework 2.0, i think. i was working on a rather large internal library and documentation was, well, let's say not a priority for the rest of the team. i wanted to make it easier for people to actually use it, and auto-generated xml docs were my weapon of choice. but hitting publish and not seeing them included, that was a problem. i spent a whole afternoon going through msbuild files before finally getting it to work. it was a good learning experience. you tend to remember these things.

the core issue is that visual studio's publish mechanism, by default, doesn't know that those xml files are something you care about. it's not magical; it just copies over what it's configured to copy. these generated xml files are not marked as "content" in the project and msbuild doesn't pick them up. so we need to tell the build process to include them as content or similar.

there are a couple of ways you could handle this.

**the "easiest" (or most common) method is to modify your csproj file directly.**

the project file, which typically ends in `.csproj` is an xml file and can be a bit daunting at first glance, but it's where msbuild gets its marching orders. you can do this either in a text editor or even better, use the visual studio unload project and edit option in visual studio itself. it provides a pretty clean way to edit the file with syntax highlighting and intellisense for the xml.

here's a snippet of what you'd likely need to add inside the project tag:

```xml
    <PropertyGroup>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    </PropertyGroup>
    <ItemGroup>
        <Content Include="$(OutputPath)\*.xml">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>true</Visible>
        </Content>
    </ItemGroup>
```

let's break this down a bit:

*   `<GenerateDocumentationFile>true</GenerateDocumentationFile>`: this is more about enabling the xml doc generation in the first place, if you don’t already have it on. you must have this one otherwise there won't be documentation files generated.
*   `<ItemGroup><Content Include="$(OutputPath)\*.xml">`: this tells msbuild: "hey, everything with a `.xml` extension in the output directory, i’m interested in it as content". output path variable normally resolves to `bin\debug` or `bin\release` depending on the build configuration.
*   `<CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>`: this copies the file only if its newer that the one in the destination path.
*   `<Visible>true</Visible>` this makes sure this file is visible in the visual studio project explorer. useful for debug if needed.

this approach is pretty straight forward and has served me well many times. it simply tells msbuild to treat those xml files as "content" to be copied into the publish directory during the publish.

**another, maybe slightly more robust way, is to leverage the msbuild `target` mechanism.**

this is a bit more involved but gives you more control and keeps your csproj file a bit cleaner if you have more complex scenarios. it can also be handy if you want to do some transformations on the xml files or to add a folder or change the name during the publishing.

```xml
   <Target Name="IncludeDocumentationXml" AfterTargets="AfterBuild">
      <ItemGroup>
           <DocumentationXmlFiles Include="$(OutputPath)\*.xml" />
      </ItemGroup>
      <Copy SourceFiles="@(DocumentationXmlFiles)" DestinationFolder="$(PublishDir)" />
   </Target>
```

let's unpack this too:

*   `<Target Name="IncludeDocumentationXml" AfterTargets="AfterBuild">`: this defines a new target called "includedocumentationxml" that runs after the standard "afterbuild" target. msbuild targets are the building blocks of the build process.
*   `<ItemGroup><DocumentationXmlFiles Include="$(OutputPath)\*.xml" />`: creates a named itemgroup that selects every xml file present in the output directory.
*  `<Copy SourceFiles="@(DocumentationXmlFiles)" DestinationFolder="$(PublishDir)" />`: copies all files selected by the "DocumentationXmlFiles" itemgroup into the publish output directory.

the key difference is that this is explicit about copying the files only as part of the publishing process.

in my experience, both of these methods work perfectly well, but the first one is easier to implement and maintain on small projects. the second gives you more power if you are handling lots of files or different locations and formats.

**a third option, useful if you are using a modern dotnet sdk project (dotnet core/dotnet 5+) and your documentation xml is produced in a different path. It is often the case with more complex projects where each component creates its own xml doc:**

```xml
  <ItemGroup>
    <PublishDocumentation Include="$(ProjectDir)docs\*.xml" />
  </ItemGroup>
  <Target Name="CopyDocumentationXml" AfterTargets="Publish">
    <Copy SourceFiles="@(PublishDocumentation)" DestinationFolder="$(PublishDir)/docs" />
  </Target>
```

the major difference here is that instead of targeting the `OutputPath`, this assumes that all of your docs are located in the `/docs` folder relative to your project directory. and copies them in a `docs` folder inside the publish folder.

using a custom target as well is a good pattern because gives you granular control, and a lot of flexibility. it also avoids polluting your content definitions when the project grows up.

now, which one should you pick? well, it depends on what feels more convenient for your situation. i would say start with the first method if you are starting out. it’s simpler and easier to grasp. the target approach is better when you feel you need more control over the publishing process. this method is more verbose. the third one is usefull to document code located in different paths (different projects in a monorepo for example).

a word of caution: after making these changes, sometimes visual studio might not pick up the changes instantly, you might need to manually close and reopen the project. and always remember to back up your project or version control it before meddling with csproj files.

now about documentation, beyond just reading the official microsoft msbuild documentation, i've found some more materials very useful over the years. "msbuild unleashed" by ian griffiths is a classic if you want to truly grasp msbuild. it's a bit of a tome, but worth its weight in gold. and the "inside msbuild" by sayed hashimi provides also great detail on the build process. for a more concise intro, "essential msbuild" by andrew hunter is a good start. these books really helped me understand the intricacies of how msbuild works, more than anything i found online.

one thing i found out the hard way is that sometimes the debugging process for msbuild is… a bit of a pain. it would sometimes not copy the files but it wouldn't show a warning or an error, at all, so i had to add a small trace using `<message text="found file %{file.name}" />` in my target just to see what was happening. but don't worry, msbuild debugging isn't rocket science or brain surgery, it's more like… a very peculiar type of accounting. (that's my attempt at humor for today. let me know if you need more help).

i hope this is useful, let me know if you have more questions.
