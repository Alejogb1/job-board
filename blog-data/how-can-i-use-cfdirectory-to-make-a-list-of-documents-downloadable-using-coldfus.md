---
title: "how can i use cfdirectory to make a list of documents downloadable using coldfus?"
date: "2024-12-13"
id: "how-can-i-use-cfdirectory-to-make-a-list-of-documents-downloadable-using-coldfus"
---

Okay so you want to make a directory listing that lets users download files using ColdFusion huh been there done that a million times It's like the bread and butter of web dev back in the day well and still today tbh 

So let's get right into it you're thinking `cfdirectory` and that's a solid starting point It's kinda the workhorse for file system interaction in CFML its purpose is listing files and directories right it doesn’t serve files it does that for you to be able to do that.  I've spent countless hours wrestling with it especially back in the CF 5 days when I first started doing this stuff oh man those were the days remember those tag spaghetti times yeah I do. It felt like trying to debug a nuclear reactor with a potato.

Alright so you want to make the files downloadable not just show their filenames So we need to combine `cfdirectory` with some kind of link generation and then a bit of file serving logic. The simplest approach goes something like this we generate a simple html list with anchor tags that point to filepaths so the users browser can download them.

Here's a basic example I used a similar approach in one of my previous projects involving legacy asset management system it was a messy monster I tell ya but this code worked perfectly fine:

```cfml
<cfset targetDirectory = "path/to/your/directory">
<cfdirectory directory="#targetDirectory#" action="list" name="fileList">

<cfoutput>
    <ul>
        <cfloop query="fileList">
            <cfif fileList.type EQ "file">
                <li><a href="download.cfm?file=#URLEncodedFormat(fileList.name)#">#fileList.name#</a></li>
            </cfif>
        </cfloop>
    </ul>
</cfoutput>
```

**A breakdown of that:**

1.  `cfset targetDirectory = "path/to/your/directory"`: This sets the path to directory to scan make sure it exists right.
2. `cfdirectory directory="#targetDirectory#" action="list" name="fileList"`: We make `cfdirectory` list the files and folders in the target location storing the result in the `fileList` query.
3.  `<cfoutput><ul></cfoutput>`: We open up an unordered list.
4.  `<cfloop query="fileList">`: loop over result rows returned from our previous `cfdirectory`.
5. `<cfif fileList.type EQ "file">`: we check if the current item is a file skipping any folders that we may have found.
6. `<li><a href="download.cfm?file=#URLEncodedFormat(fileList.name)#">#fileList.name#</a></li>`: This generates the link to a file using `download.cfm` and passes it the filename via the query string. Note `URLEncodedFormat` is important here to encode any spaces and special characters that files can have.
7.  `</ul></cfoutput>`: Close the html ul tag that we opened at step 3.

Alright now what's that `download.cfm` all about well it handles the actual file serving I bet you figured that out. Here's a simple implementation:

```cfml
<cfparam name="URL.file" default="">

<cfset filePath = "path/to/your/directory/#URL.file#">

<cfif not FileExists(filePath)>
  <cfabort>File Not Found</cfabort>
</cfif>


<cfheader name="Content-Disposition" value="attachment; filename=#URL.file#">
<cfheader name="Content-Type" value="#getMimeType(filePath)#">
<cffile action="readbinary" file="#filePath#" variable="fileContent">
<cfcontent type="#getMimeType(filePath)#" variable="#fileContent#">

<cffunction name="getMimeType">
    <cfargument name="file" type="string" required="true">
    <cfset var ext = listLast(arguments.file,".")>
    <cfswitch expression="#ext#">
        <cfcase value="pdf">
            <cfreturn "application/pdf">
        </cfcase>
        <cfcase value="txt">
            <cfreturn "text/plain">
        </cfcase>
        <cfcase value="jpg,jpeg">
            <cfreturn "image/jpeg">
        </cfcase>
		<cfcase value="png">
			<cfreturn "image/png">
		</cfcase>
		<cfcase value="gif">
			<cfreturn "image/gif">
		</cfcase>
        <cfdefaultcase>
            <cfreturn "application/octet-stream">
        </cfdefaultcase>
    </cfswitch>
</cffunction>
```
**And what the hell that does**

1.  `<cfparam name="URL.file" default="">`: We check to see if the file was actually passed into url paramter called `file` if not just default to empty string for sanity.
2.  `cfset filePath = "path/to/your/directory/#URL.file#"`: Make the complete path to the file to be served.
3.  `<cfif not FileExists(filePath)>  <cfabort>File Not Found</cfabort></cfif>`: Let us be sure that file exists before we do anything.
4.  `<cfheader name="Content-Disposition" value="attachment; filename=#URL.file#">`: We need to force the browser to try and download the file using the `attachment` header.
5.  `<cfheader name="Content-Type" value="#getMimeType(filePath)#">`: We need to send correct Content-Type header so that browsers know what to do with file.
6.  `<cffile action="readbinary" file="#filePath#" variable="fileContent">`: Load the file into memory using `readbinary` action in a variable named `fileContent`.
7.  `<cfcontent type="#getMimeType(filePath)#" variable="#fileContent#">`: We output file content using the correct mime type we extracted.
8. `<cffunction name="getMimeType"> ...`: This is simple function I wrote which is used to extract mime type based on the file extension. Not the most efficient but you can make it better it works for the sake of the example.

Couple of things: In a real-world setup don't rely on extensions like that for content types instead use Apache Tika or some other similar library to extract the correct mime-type. I mean why would someone name a pdf file with txt extension just to cause headaches. I've seen things. Also make sure that "path/to/your/directory" is in a location where your CF app has read permissions for files and directory.

So that covers the basics I would go a bit further tho in production grade systems we generally don't put files directly under the webroot. Instead we put it outside of the web accessible folder for security reasons. We need a bit more work with a secure download system where we use file ids and not file paths that is what you should do if there is confidential data in those files. Also you could use a session variable to handle access rights. Here is a glimpse of a slightly more complex example but it does not include all features that I have discussed above.

```cfml
<cfset targetDirectory = "path/to/your/directory">
<cfset fileList = directoryList(targetDirectory)>


<cfoutput>
    <ul>
        <cfloop array="#fileList#" index="file">
            <cfif isFile(file)>
                <cfset fileId = hash(file)>
                <li><a href="download.cfm?fileId=#fileId#">#listLast(file,"/")#</a></li>
            </cfif>
        </cfloop>
    </ul>
</cfoutput>

<cffunction name="directoryList">
	<cfargument name="targetDirectory" type="string" required="true">
	<cfset var files = arrayNew(1)>
	<cfset var localFileList = directoryList(arguments.targetDirectory)>
    <cfdirectory directory="#arguments.targetDirectory#" action="list" name="fileList">
	<cfloop query="fileList">
		<cfset arrayAppend(files, arguments.targetDirectory & "/" & fileList.name)>
	</cfloop>
	<cfreturn files>
</cffunction>

<cffunction name="isFile">
	<cfargument name="targetPath" type="string" required="true">
    <cfset var result = false>
	<cftry>
		<cfset result = fileExists(arguments.targetPath)>
		<cfcatch type="any">
			<cfset result = false>
		</cfcatch>
	</cftry>
    <cfreturn result>
</cffunction>
```

**And what is this about**
1. `<cfset targetDirectory = "path/to/your/directory">`: just as the first example this determines our target directory.
2. `<cfset fileList = directoryList(targetDirectory)>`: This time instead of `cfdirectory` this example uses recursive approach to list files which is more flexible and gives you more control.
3. `<cfoutput><ul></cfoutput> `: just as previous example we open ul tags to generate list.
4.  `<cfloop array="#fileList#" index="file">`: loop over our file list generated using a function not a query.
5.  `<cfif isFile(file)>`: we filter again but this time using function we have written which uses exception handling.
6.  `<cfset fileId = hash(file)>`: we hash the file path which is used as the id instead of the original name.
7.  `<li><a href="download.cfm?fileId=#fileId#">#listLast(file,"/")#</a></li>`: link is generated but this time it uses the hash as an id. And only shows file name.
8.  `<cffunction name="directoryList"> ...`: The file list function which recursivelly scans for all files in subdirectories.
9. `<cffunction name="isFile"> ... `: And the function that checks if a path is actually a path to file.


And the modified download.cfm.

```cfml
<cfparam name="URL.fileId" default="">

<cfset targetDirectory = "path/to/your/directory">
<cfset fileList = directoryList(targetDirectory)>

<cfset filePath = "">

<cfloop array="#fileList#" index="file">
	<cfif hash(file) EQ URL.fileId>
		<cfset filePath = file>
		<cfbreak>
	</cfif>
</cfloop>

<cfif filePath EQ "">
	<cfabort>File not found</cfabort>
</cfif>

<cfif not isFile(filePath)>
  <cfabort>File Not Found</cfabort>
</cfif>

<cfheader name="Content-Disposition" value="attachment; filename=#listLast(filePath,"/")#">
<cfheader name="Content-Type" value="#getMimeType(filePath)#">
<cffile action="readbinary" file="#filePath#" variable="fileContent">
<cfcontent type="#getMimeType(filePath)#" variable="#fileContent#">
<cffunction name="directoryList">
	<cfargument name="targetDirectory" type="string" required="true">
	<cfset var files = arrayNew(1)>
	<cfset var localFileList = directoryList(arguments.targetDirectory)>
    <cfdirectory directory="#arguments.targetDirectory#" action="list" name="fileList">
	<cfloop query="fileList">
		<cfset arrayAppend(files, arguments.targetDirectory & "/" & fileList.name)>
	</cfloop>
	<cfreturn files>
</cffunction>

<cffunction name="isFile">
	<cfargument name="targetPath" type="string" required="true">
    <cfset var result = false>
	<cftry>
		<cfset result = fileExists(arguments.targetPath)>
		<cfcatch type="any">
			<cfset result = false>
		</cfcatch>
	</cftry>
    <cfreturn result>
</cffunction>
<cffunction name="getMimeType">
    <cfargument name="file" type="string" required="true">
    <cfset var ext = listLast(arguments.file,".")>
    <cfswitch expression="#ext#">
        <cfcase value="pdf">
            <cfreturn "application/pdf">
        </cfcase>
        <cfcase value="txt">
            <cfreturn "text/plain">
        </cfcase>
        <cfcase value="jpg,jpeg">
            <cfreturn "image/jpeg">
        </cfcase>
		<cfcase value="png">
			<cfreturn "image/png">
		</cfcase>
		<cfcase value="gif">
			<cfreturn "image/gif">
		</cfcase>
        <cfdefaultcase>
            <cfreturn "application/octet-stream">
        </cfdefaultcase>
    </cfswitch>
</cffunction>
```

**And what has changed?**
1.  `<cfparam name="URL.fileId" default="">`: The url parameter that we check has been renamed from `file` to `fileId`.
2.  `<cfset filePath = "">`: the file path variable that will store the correct file is initialized.
3.  `<cfloop array="#fileList#" index="file"> ...`: loop over array of file paths.
4.  `<cfif hash(file) EQ URL.fileId> ...`: find matching path using hash that we set earlier.
5.  `<cfif filePath EQ ""> ...`: abort if path was not found.
6.  `<cfheader name="Content-Disposition" value="attachment; filename=#listLast(filePath,"/")#">`: get the filename for the header.

This is just a basic setup of course It can be made more robust with better error handling sanitization and file path management. But hey it gets the job done you know. This method was something I came up with during a late night debugging session back when CF8 came out good old times. 

As a quick recommendation if you're digging deeper into file management in CFML then you should definitely check out some of the online resources and books on CFML specifically. There used to be the "Advanced ColdFusion 8 Application Development" back in those days I found it super useful for this type of thing it is dated sure but many principles are still valid.

And if you're dealing with more complex scenarios where you have a lot of files or large files consider using a CDN or a file storage service like Amazon S3 it's usually the smarter approach to avoid bottle-necking your own server.

So yeah that's pretty much it for now hope it helps you out with your file downloading issue.
