---
title: "su must be suid work root?"
date: "2024-12-13"
id: "su-must-be-suid-work-root"
---

 so you're asking why `su` which is the switch user command needs the setuid bit set and also needs to be owned by root right Got it I've been down this rabbit hole more times than I care to admit lets unpack this thing

So first off lets talk about what `su` even does in simple terms it lets you change your current user context to another user Usually its used to become root so you can do system level administration stuff but you can also use it to switch to other regular users

Now you might be thinking well why cant I just run a program as any user I want like I own the computer right Wrong operating systems do not work like that They have a concept called user ids or UIDs each user has a unique numerical UID and processes are tagged with the UID of the user who started them This is essential for file permissions and security in general If you could just switch UIDs willy nilly things would be chaos So to run a program with a different UID the system needs a mechanism

And that's where the setuid bit comes in The setuid bit is a special file permission bit when set on an executable it tells the kernel to run that executable with the UID of the *owner* of the executable not the user who executed it This is very different so keep that in mind

Now lets look at `su` specifically. If you do an `ls -l /bin/su` or `/usr/bin/su` which is the common place you'll find it you would probably see something like this

```bash
-rwsr-xr-x 1 root root 47048 Oct 17 2022 /bin/su
```
See that `s` in the user permissions spot not `x` That means the setuid bit is set this is critical

Also the file is owned by root as you saw which is also critical. If the file is not owned by root or if the setuid bit is not set `su` simply will not work as you expect it or at all.

Now imagine you are a regular user with UID 1000 lets call you user bob. You type `su` the shell now executes `/bin/su`. Because the setuid bit is set the program `/bin/su` executes as root or UID 0 instead of UID 1000. Think of it like this the `/bin/su` program temporarily borrows roots power

The `su` program now does its thing asking you for the root password checking it and then spawning a new shell process but this shell process now has the UID of root. This is how you become root.

 so why does `su` absolutely positively need root ownership. Well the setuid bit only works if the executable is owned by the *user* whose privileges you are trying to inherit in our case its the all powerful root UID 0. It also needs to be owned by root to make sure that only an admin can modify the su program.

If you tried setting the setuid bit and the owner was not root then it would just mean that the process will temporarily have the user id of say bob or any other user and not root or what ever the owner of the `/bin/su` file is

If it were set to another regular user it'd be pointless. I had an instance where I copied the `/bin/su` and set the ownership and the setuid bit to my user, thinking I was a genius but all I did was allow myself to run a root program as my own user which obviously does nothing. Learned a lot that day.

There are some other things like the sticky bit and setgid but these are not strictly part of this question and would be too off topic to go deeper so I'll skip these but those are good ones to look into if you want to learn more on linux file permissions.

Now lets say you had a buggy `su` implementation like I did once where you could bypass it by creating a file in the same directory called `/bin/su` because it had a higher precedence in the search path and it had the setuid bit set but it was not owned by root then your whole security model falls apart that's why ownership matters a lot.

Here is some pseudo code that demonstrates the core idea but its a dangerous program dont run this code or compile this.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    // Check if the setuid bit is set correctly and owned by root
    // This is a dangerous example and should never be run as it bypasses all security
    // This is only here to make the point of why su need to be owned by root and have the bit set
    // DO NOT RUN
    if(getuid() == 0) {
       printf("I have root access because the setuid bit is set and owned by root, not your user.\n");
       // Code here that can do dangerous things
    } else {
        printf("You are not root sorry can't do what you want to do with no root privileges\n");
    }
    return 0;
}
```
The `getuid()` system call is how you check who the current user is in any program running. Here the idea is that when we run the program we check for uid to be 0 if yes that means the process is executing with uid 0 not the uid of the user who ran the program which is what would happen without the setuid bit being set.

The whole concept is very important to understand when doing security on linux systems. Lets move to more practical stuff now.

If you want to see it in action you can actually try this in a controlled environment like a virtual machine dont do this on a system you care about.

```bash
# copy su to a temp location
cp /bin/su /tmp/my_su

# set the owner to a non-root user in this case it is the current user, which is not root
sudo chown $USER:$USER /tmp/my_su

# try to run it like normal su
/tmp/my_su

# try to make it work by setting the setuid bit
chmod u+s /tmp/my_su

# try to run it again
/tmp/my_su
```
You will notice that it won't work it wont give you root because it's not owned by root It doesn't matter if you give the file the permission it will still check to see the owner of the program. This demonstrates the importance of root ownership with the setuid bit.

Now I'll add another pseudo code that is very close to the idea of what `su` does in essence just for a better understanding again do not run this as this is insecure and only for educational purposes.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <pwd.h>

// Danger zone program this is for educational purposes only.
int main(int argc, char *argv[]) {
    //check if the user is root
    if(getuid() != 0) {
       printf("This program must be run as root\n");
       return 1;
    }

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <username>\n", argv[0]);
        return 1;
    }

    char *target_username = argv[1];
    struct passwd *target_pw = getpwnam(target_username);
    if (target_pw == NULL) {
        fprintf(stderr, "User not found: %s\n", target_username);
        return 1;
    }

    uid_t target_uid = target_pw->pw_uid;
    
    // This is a very dangerous operation only for educational purposes. Do not use this
    if (setuid(target_uid) != 0) {
        perror("setuid failed");
        return 1;
    }
     printf("Successfully switched to user %s (uid: %d)\n",target_username, target_uid);
     char *const shell_args[] = { "/bin/bash", NULL };
    execv("/bin/bash", shell_args);

     perror("execv failed");

    return 0;
}
```

This code is a simplification of what `su` does. `getpwnam()` is used to get the user information given the username from the system users database and then it sets the uid of the process using `setuid()` system call if it is sucessful it then executes a new shell with the new user privileges. In this case it would work because we ran this program with root permissions and it allowed `setuid()` to operate correctly. You can only execute this code when the process is running as root.

Again do not use this in real life systems it is just for demonstration purposes. Now if I have to pick a single paper or book to recommend to you its probably "Operating System Concepts" by Abraham Silberschatz which covers these things very deeply. Also maybe look into papers on capabilities if you're digging really deep they are good resources to go a little bit deeper.

Oh hey I have a funny story for you when I first started I once tried to use `sudo` to fix a permissions issue on `su` I was like super meta it did not work it turns out you just need to chmod not sudo the root owned file haha but hey we all started somewhere right.

So to wrap it up yeah `su` needs to be owned by root and have the setuid bit set that's how it works its not a bug it's a critical feature for system administration on linux type systems or the systems that follow this type of permission system. Hopefully this helps clear things up let me know if you have any other questions.
