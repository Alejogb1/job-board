---
title: "Why isn't HAXM installing?"
date: "2024-12-23"
id: "why-isnt-haxm-installing"
---

,  I’ve seen this HAXM installation hurdle plenty of times, and it usually boils down to a few common culprits. Rather than jumping straight to a list of potential fixes, I'd prefer to walk you through the diagnostic thought process I typically employ. We’re aiming for a systematic solution, not just a shot in the dark.

First, let’s clarify what HAXM *is*. Hardware Accelerated Execution Manager (HAXM) is Intel's virtualization engine—a crucial component if you're working with emulators, particularly in Android development. It leverages your processor's virtualization capabilities to dramatically speed up emulation. So, when it's not installing, the first thing that pops into my head is: are those virtualization features actually available and enabled on the machine?

I remember a project years ago where we were trying to set up a new build server, and we spent hours pulling our hair out trying to get the android emulator to run at any reasonable speed. Turns out, a setting was buried deep in the server’s bios. It's a painful lesson that highlights the fundamental pre-requisite for HAXM: the *processor must support virtualization*, and *it needs to be enabled in the BIOS/UEFI settings*. This isn’t something the software installer can just circumvent.

Before diving into specific code or system checks, let's address a common misconception. HAXM installation issues often get blamed on some random conflict, but the actual cause usually falls within these three primary categories: *disabled virtualization*, *conflicting hypervisors*, and *incompatible system state*. Let’s break each of these down:

**1. Disabled Virtualization:** This is the most common culprit. The virtualization extensions (VT-x for Intel, AMD-V for AMD) need to be explicitly enabled. They are often disabled by default in the BIOS/UEFI firmware to reduce power consumption. To check, boot into your BIOS/UEFI setup (the exact key varies by manufacturer, often it’s del, f2, or f12). Look for settings related to ‘virtualization,’ ‘VT-x,’ ‘SVM’ (Secure Virtual Machine, for AMD), or similar terms. Enable it, save the changes, and reboot. I’ve frequently had to guide new engineers through this specific step, and it always amazes me how many leave it disabled. If you do have virtualization enabled in the BIOS, and HAXM still fails, it is worthwhile to go in and disable and re-enable the virtualization to verify the setting was properly applied.

**2. Conflicting Hypervisors:** Multiple hypervisors trying to co-exist on one system will result in nothing but problems. If, for example, Hyper-V or WSL2's hypervisor is active, it will conflict with HAXM. HAXM is a type-2 hypervisor. These systems cannot run concurrently with other hypervisors, and that is the root of the problem when people experience installation failures. This is a slightly less obvious issue for those who don’t interact often with virtual machine technologies. When I was working with docker for our development environment, I initially had problems with HAXM until I disabled Hyper-V on the system. To check this on windows, go to "Turn Windows features on or off" in Control Panel and check if Hyper-V is enabled. Consider disabling it temporarily to see if it resolves the HAXM installation issue, and remember that WSL2 requires hypervisor services enabled, so it may need to be temporarily disabled. Linux based users should be sure that the KVM module is not currently in use.

**3. Incompatible System State:** Occasionally, outdated Windows system files or incomplete driver installations can interfere. The installer requires the windows version of .net 4.7, so that is something that should always be verified if the first two steps fail. This is far rarer than the first two, but it's worth considering. Also make sure the Hyper-V driver is fully removed before attempting HAXM installation, even if the feature is disabled. A good way to do this is by removing the feature in the windows features, rebooting, and verifying the Hyper-V driver has been uninstalled.

, now for some code examples, focusing on the most common checks and solutions, not just for install. I’ve used a mix of Powershell and command-line commands, which should be adaptable for most environments:

**Snippet 1: Verifying VT-x/AMD-V is enabled from the Operating System (Windows):**

```powershell
# Check for VT-x (Intel)
Get-WmiObject -Class Win32_ComputerSystem | Select-Object HypervisorPresent | Format-List

# Check for Hyper-V and Virtualization (requires Administrator)
Get-WindowsOptionalFeature -FeatureName Microsoft-Hyper-V-Hypervisor -Online | Select-Object State

# This command provides more info in the event the features are enabled, but not working.
systeminfo | findstr /I "hypervisor"
```

This Powershell code first checks if a hypervisor is present according to windows system configurations. Then, it uses `Get-WindowsOptionalFeature` to check if the Hyper-V Hypervisor is installed and enabled. Finally, `systeminfo` provides a lot of detail and is frequently useful to verify the status of the hardware virtualization features of the system. If "hypervisor running" is set to "No" and virtualization is enabled in the BIOS, there might be other things interfering.

**Snippet 2: Attempting a HAXM Uninstall/Reinstall via Command Line (Windows):**

```cmd
REM Navigate to the HAXM installation directory. Usually in program files.
cd "C:\Program Files\Intel\HAXM"

REM Uninstall HAXM.
silent_uninstall.exe

REM Force uninstall if the silent uninstall fails.
msiexec /x haxm.msi /qn

REM (Once uninstalled), Try re-installing HAXM. This path depends on your installation.
haxm-7.8.3-setup.exe
```

This command-line sequence attempts a clean uninstall of HAXM, followed by a re-install. Sometimes, a corrupted installation is the root cause, and this approach can fix the problem. It starts by changing the working directory to the HAXM installation directory, assuming the default one is correct. If the silent uninstall does not work, then the `msiexec` command is used to forcefully remove the HAXM package. Finally, the re-installation of HAXM is started. The version may vary, so be sure to replace the version number as appropriate.

**Snippet 3: Verifying the Hyper-V Driver is Removed.**

```powershell
#check for hyper-v services
Get-Service | Where-Object {$_.Name -like "hv*"} | Select-Object Name, Status

#Check for hyper-v drivers
Get-WindowsDriver -Online | Where-Object {$_.ProviderName -like "Microsoft"} | Select-Object Classname, FriendlyName | Where-Object {$_.Classname -like "System"} | Where-Object {$_.FriendlyName -like "*hyper-v*"}

#Force the removal of hyper-v drivers from powershell.
pnputil /delete-driver oem*.inf /force
```

This code starts by checking to see if any services containing the word "hv" are active. This will show any hyper-v related services still running on the machine. Next, it shows all the system class drivers on the system related to "hyper-v." If drivers remain after disabling the hyper-v windows feature, they can be force removed with the `pnputil` command. It is important to re-install HAXM after these drivers have been properly removed.

For further reading and a more in-depth look at system level virtualization, I would recommend *Intel® 64 and IA-32 Architectures Software Developer's Manual*, volume 3, which goes into the details of the Virtual Machine Extension (VMX) instructions, if you really want to understand the technology at a deeper level. For a practical understanding of how virtualization is used for emulation and development specifically on android, look for information regarding the Android Open Source Project (AOSP) documentation, as well as the specific documentation provided for HAXM from Intel. For a more general overview of virtualization technologies in general, the book *Virtual Machines: Versatile Platforms for Systems and Processes* by Jim Smith and Ravi Nair provides a good introduction to many related concepts, as well as some of the more advanced techniques that may be relevant for those interested in taking a deeper dive into virtualization technologies.

In my experience, a methodical check through these areas usually identifies the source of the HAXM installation issues. It rarely has been a software bug in HAXM, but rather, a misconfiguration or conflicting system state. The key takeaway is to verify the hardware virtualization settings, check for hypervisor conflicts, and understand that these are the primary causes of installation difficulties. By methodically eliminating these, the issues often become readily apparent, rather than chasing phantom problems.
