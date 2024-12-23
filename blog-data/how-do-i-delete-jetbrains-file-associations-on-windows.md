---
title: "How do I delete JetBrains file associations on Windows?"
date: "2024-12-23"
id: "how-do-i-delete-jetbrains-file-associations-on-windows"
---

,  I've seen this particular issue rear its head countless times over my years in development. It's one of those seemingly minor annoyances that can snowball into a significant productivity drain if not addressed correctly. You're trying to remove file type associations that were established by a JetBrains IDE—say, IntelliJ IDEA, PyCharm, or WebStorm—on a Windows system. It's a common scenario, particularly after a fresh install, upgrade, or if you've been experimenting with various IDE configurations. The challenge comes from the fact that these associations aren't always neatly stored in one place, and sometimes the standard Windows interface falls short.

My typical approach involves a blend of methods, starting with the more conventional and then escalating as required. It's always best to start with the least invasive technique. First off, let's try the built-in Windows settings. I recall one instance involving a particularly persistent `.sql` file association that refused to yield through the regular settings menu. We had to go digging deeper, but let's not get ahead of ourselves.

The initial step you should take is to navigate to the "Settings" app in Windows. Specifically, you'll want to go to "Apps" and then "Default Apps." From there, you'll find an option to "Choose default apps by file type." This displays a list of file extensions and the applications currently associated with them. You can scroll through the list, find the file extension causing you grief (e.g., `.java`, `.py`, `.html`, etc.), and change the associated application. If JetBrains is still listed as an option, you should be able to select "Choose an app" and either select a different application or select "Look for an app in the Microsoft Store," which often clears the current association if no suitable application is found. This is generally the simplest and most direct path.

However, and this is a big *however*, sometimes the associations are more stubbornly rooted. This is where the registry comes into play. Before diving into this, it's essential to understand the risks. The Windows Registry is a sensitive area, and incorrect modifications can lead to system instability. I *strongly* advise creating a system restore point before making any changes there. Consider it akin to a safety net before walking a tightrope – a practical safeguard.

The registry keys related to file associations are typically found under `HKEY_CLASSES_ROOT`. Within this hive, you'll find keys corresponding to file extensions (starting with a dot, like `.txt`, `.py`, etc.) and keys representing file type handlers. You'll be looking for entries associated with your JetBrains product. Let’s illustrate this with a couple of specific examples via the `reg` command in the command prompt, and after that, I'll present a PowerShell equivalent to give us flexibility. The command prompt method is simple, but the PowerShell version allows for more automation if needed.

**Example 1: Deleting a File Association using `reg` command (Command Prompt)**

Suppose you want to remove the association for `.myproj` files from IntelliJ. You'd first have to identify the exact file type handler. Often, this is something with the name containing the IDE, for example `IntelliJ.myproj`.

```batch
rem Identify the file type handler for the extension
reg query "HKCR\.myproj"

rem Output will look something like:
rem   (Default) REG_SZ IntelliJ.myproj

rem Delete the file type handler association for the file extension .myproj
reg delete "HKCR\IntelliJ.myproj" /f

rem Remove association from .myproj itself, if necessary.
reg delete "HKCR\.myproj" /f
```

In this example, `reg query` first retrieves the default entry for the specified file type. If the result contains `IntelliJ.myproj` (or a similar identifier that points to the desired IDE), you then use `reg delete` with the `/f` flag to force the deletion. After this step, you might also want to delete the extension's entry itself if the system doesn't reassign it automatically.

**Example 2: Deleting a file association using `reg` command (Command Prompt) with confirmation (Safer approach)**

Sometimes you'd rather not delete without confirmation. This example shows a similar approach but with an extra check before deleting.

```batch
@echo off
setlocal

set "ext=.myproj"

reg query "HKCR\%ext%" > nul 2>&1
if errorlevel 1 (
    echo Extension "%ext%" not found.
    exit /b 1
)

for /f "tokens=2*" %%a in ('reg query "HKCR\%ext%" ^| findstr "(Default)"') do set "handler=%%b"

echo.
echo Current Handler: %handler%
echo.

if "%handler%"=="" (
  echo No default handler found for %ext%
  exit /b 1
)


echo Do you want to remove handler %handler%? (y/n)
set /p "choice="
if /i "%choice%"=="y" (
  echo Deleting %handler%...
  reg delete "HKCR\%handler%" /f >nul 2>&1
    if errorlevel 1 (
    echo Deletion failed for handler %handler%
    exit /b 1
  )

  echo.
  echo Removing association for  %ext%
  reg delete "HKCR\%ext%" /f  >nul 2>&1
  if errorlevel 1 (
    echo Deletion failed for extension %ext%
    exit /b 1
  )
  echo.
  echo Associations deleted successfully.
) else (
    echo Deletion cancelled.
)

endlocal
```

This example first checks if the extension is valid using `reg query`. Then, it gets the associated handler. Finally, it asks for user confirmation before executing the delete.

**Example 3: Using PowerShell for the same task**

PowerShell provides a more streamlined and flexible way to handle registry interactions. Here’s how you can use it:

```powershell
# Specify the file extension
$fileExtension = ".myproj"

# Get the associated registry key name
$regKey = Get-ItemProperty "HKCR:\$fileExtension" -ErrorAction SilentlyContinue | Select-Object "(default)"
if (!$regKey) {
    Write-Host "No association found for $($fileExtension)"
    return
}

$handler = $regKey.'(default)'

if(!$handler){
    Write-Host "No handler for $($fileExtension)"
    return
}


Write-Host "Current handler for $($fileExtension): $($handler)"

$Confirm = Read-Host -Prompt "Do you want to remove this handler and association? (y/n)"

if($Confirm -eq "y"){

    # Remove the handler key
    try {
      Remove-Item -Path "HKCR:\$handler" -ErrorAction Stop
      Write-Host "Handler key '$handler' removed successfully."
    } catch {
       Write-Host "Error removing handler key: $_"
    }

    # Remove the file extension association itself
     try {
        Remove-Item -Path "HKCR:\$fileExtension" -ErrorAction Stop
        Write-Host "Association for '$fileExtension' removed successfully."
    } catch {
      Write-Host "Error removing association: $_"
    }
}
else {
  Write-Host "Operation Cancelled."
}

```

This script first uses `Get-ItemProperty` to retrieve the registry key related to the file extension. Then, it attempts to delete both the handler and the extension's key using `Remove-Item`. The `-ErrorAction Stop` option here is helpful for catching and reporting errors during registry operations. This gives you more control during operation compared to the batch file version.

After these operations, you'll want to close any open file explorer windows and potentially even restart explorer.exe (using the task manager) for the changes to take full effect. You may need a full reboot of the system as well depending on the complexity of your setup.

For further exploration, I'd suggest consulting "Windows Internals, Part 1" by Mark Russinovich, David Solomon, and Alex Ionescu. This book provides an in-depth understanding of how the Windows operating system, including its registry, functions. For specific details on file associations, you might find the relevant sections on the Microsoft documentation helpful, specifically the articles and resources related to "File Associations" in the official Windows Development area, and specifically, the documentation related to the `reg` command and the `Remove-Item` PowerShell command.

In summary, removing file associations isn't always straightforward, requiring a methodical approach that balances the ease of built-in tools with the more precise control afforded by registry manipulation. Remember to proceed cautiously when interacting with the registry, and take proper backups before making any changes.
