---
title: "Why is homebrew installation failing on macOS Catalina?"
date: "2025-01-30"
id: "why-is-homebrew-installation-failing-on-macos-catalina"
---
Homebrew installation failures on macOS Catalina often stem from significant shifts in the operating system’s security model and its system directory structure, specifically concerning permissions and the separation of the operating system from user-installed software. Prior to Catalina, users could generally install Homebrew, along with other third-party tools, with fewer restrictions. Post-Catalina, however, the system volume is mounted as read-only, a change directly impeding Homebrew’s typical operation which involves writing to directories within `/usr/local`. This fundamental alteration requires a deeper understanding of how the installer and the user need to adapt to maintain functionality.

The primary cause of these failures isn't a bug in Homebrew itself, but rather a direct conflict with Catalina’s System Integrity Protection (SIP) and the read-only nature of the system volume. When Homebrew attempts to write to directories traditionally located within `/usr/local` or creates necessary directories within this space, it is now met with a refusal because these locations are part of the sealed system volume. This leads to permission errors, and the installation process stalls or aborts. The installer needs to be adapted to function properly within the new framework, often involving moving the installation prefix to a location outside of the protected system space.

The typical error message encountered during installation often reflects this issue with variations of “permission denied” or “cannot create directory” within the `/usr/local` hierarchy. I’ve personally encountered this across several machines during upgrades and deployments, experiencing these frustrating situations firsthand. A common misconception is that disabling SIP will resolve the issue. While it *may* allow the installation to proceed to the traditional `/usr/local` location, disabling SIP is not a recommended or secure practice and it's best to avoid it for long-term system stability. The more sustainable approach is to utilize the Homebrew installation options to accommodate Catalina’s security changes and use a writeable installation prefix. This often means modifying the default installation location and ensuring permissions are correctly set for the new directory.

Let me illustrate with a few examples. A failed installation often looks like this in the terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
...
mkdir: /usr/local/Cellar: Read-only file system
Error: Failed to create /usr/local/Cellar
```
This snippet highlights the explicit permission error encountered when Homebrew's installer attempts to create a crucial directory within the now read-only `/usr/local` structure. The error message `Read-only file system` is a direct symptom of Catalina's system-volume protection. This is often the first indication that the standard installation procedure will fail under Catalina, highlighting the system's refusal to allow modifications within this protected area. Attempts to force creation will lead to more similar errors further down the installation process.

Here is a corrected installation command that uses the `/opt/homebrew` prefix, addressing the Catalina issue:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
export HOMEBREW_PREFIX="/opt/homebrew"
eval "$(/opt/homebrew/bin/brew shellenv)"
```

This code example demonstrates the correct approach for installing Homebrew on Catalina (and later versions of macOS). Setting `HOMEBREW_PREFIX` before executing the install script instructs the installer to utilize the `/opt/homebrew` directory instead of the protected `/usr/local`. The `eval` command then ensures the necessary environment variables are set up for Homebrew to function correctly post-installation. This is not a workaround, but rather the standard, supported installation method since the introduction of the read-only system volume. By adopting this approach, the system's security measures are honored, and Homebrew functions as designed.

Finally, a common setup step I always employ after installation (using the correct prefix) is to check its operation. Here's a simple `brew doctor` check:
```bash
/opt/homebrew/bin/brew doctor
```

This command executes the `brew doctor` utility, which scans the Homebrew environment and reports any potential problems or conflicts. After a successful installation to `/opt/homebrew` (or another user-specified prefix), this should report a clean bill of health. If any issues are found, the report will often contain helpful hints towards resolution. I’ve used `brew doctor` countless times to troubleshoot issues from misconfigured paths to missing dependencies. It’s the first thing I run after a new installation and it is crucial to confirm the integrity of the install and the environment. The output of this command provides a clear indication of the success or failure of the previous steps. Any failures here would indicate something amiss with either the prefix setup or the permissions at `/opt/homebrew`, requiring further investigation.

In my experience, the key to successfully installing Homebrew on Catalina (and later macOS versions) lies in a precise understanding of the operating system's security architecture and using the correct installation prefix. Ignoring this reality will invariably result in persistent permission failures. The use of a prefix outside the system volume ensures compatibility with the read-only system, preventing conflicts with SIP, which is enabled by default on Catalina.

Furthermore, understanding basic shell environment concepts is essential, particularly regarding PATH variables. If one were to use the `/opt/homebrew` prefix, failing to adjust the shell’s path to include `/opt/homebrew/bin` will mean commands such as `brew` will not be recognized by the shell. The installation script sets these paths for you, but in cases where a manual setup has been done, the shell's path must be updated explicitly in the shell initialization script.

For further learning and detailed explanations, I'd recommend consulting the official Homebrew documentation, which is typically very comprehensive and up-to-date. The macOS security documentation available through Apple's developer resources also provides considerable background information about SIP and related changes to the operating system. There are numerous guides and tutorials on the web, but it's paramount to prioritize official documentation to understand the underlying concepts and not just rely on copy-pasting commands. Finally, community forums on websites like Reddit or Stack Exchange can offer insights into specific scenarios, troubleshooting tips, and different perspectives. The combination of official documentation, operating system background and community engagement will enable you to handle Homebrew installations and the errors you encounter with confidence.
