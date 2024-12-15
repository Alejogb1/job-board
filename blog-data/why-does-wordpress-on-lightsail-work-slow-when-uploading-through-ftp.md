---
title: "Why does Wordpress on Lightsail work slow when uploading through FTP?"
date: "2024-12-15"
id: "why-does-wordpress-on-lightsail-work-slow-when-uploading-through-ftp"
---

hey,

so you’re seeing wordpress on lightsail getting sluggish with ftp uploads, huh? i've been down that rabbit hole more times than i care to remember. it's a classic case of "works locally but then... bam!". let me tell you, this isn't some weird wordpress quirk. it's almost always a combination of factors, and i've got some experience breaking them down.

first, let's talk about the obvious: network latency. lightsail isn't some magic bullet, it's still a vps running in a data center somewhere. if your connection to that data center isn't great, ftp is going to feel like you're moving molasses. now, this isn't just about your download speed. it's also about packet loss and jitter. think of it like trying to fill a bathtub with a hose that's got a few holes in it and sometimes the water pressure drops. you'll get water in the tub eventually, but it will take forever. i recall this one time back in 2015 when i was using a server located in some place in ohio (i was in san francisco) and i spent two days trying to figure out why file transfers took so long before realizing that it wasn't the server configuration but rather my internet connection. i switched my connection and problem solved. you can use tools like `mtr` or `traceroute` from your local machine to your lightsail instance to get some data on latency to see if that is the case. look for packet loss or jumps in latency. if you see consistent latency over 100ms, or lots of packet loss, that's where your main bottleneck is and fixing that is beyond server configuration issues. usually a problem with your internet provider or your physical connection.

second, and this is a big one for wordpress, is disk i/o. lightsail gives you ssd storage, but it's shared. so, if other lightsail instances on the same physical hardware are thrashing their disks, your i/o can suffer. what does this mean for ftp? well, when you're uploading a bunch of files, the webserver (likely apache or nginx) is writing to disk, wordpress is probably doing some things, and that all adds up. that's disk i/o. if your disk is slow, or heavily utilized, ftp is going to be painfully slow, even for small files. i once had a project where we thought we were experiencing a network issue, but it was actually our database struggling under load that it was causing that bottleneck. we eventually moved to a managed database solution and the issue disappeared. use a tool like `iotop` to watch real-time disk i/o on your lightsail instance. it gives you a good idea of what's going on under the hood.

third, and this one is also important, is ftp itself. ftp, while simple, isn't exactly known for performance, and the default ftp implementation on most servers is rarely optimized. it's an old protocol, it's not really intended for the speed we expect nowadays. in fact, ftp connections and file transfers are single-threaded. a single transfer uses a single connection, one connection to upload one file, even if you're using an ftp client that has multiple connections. so if you're uploading many small files, each one has to go through the whole handshake, transfer, and disconnect process which is time consuming. i remember back in 2010 i tried to upload some 500 small files one by one to a shared hosting server via ftp. it took ages. it took me half a day just to upload and i had to restart the upload process three times due to network issues. that was also when i discovered rsync and never looked back. consider using `sftp` (ssh file transfer protocol) which is encrypted and is known to have better performance overall, specially when used with compression. or even better, `rsync` over `ssh` is far superior when uploading multiple files, because it can transfer files incrementally.

now, onto wordpress itself: a badly configured wordpress installation can also slow things down. if you have a ton of plugins and themes installed, each one of them will add overhead which means they add complexity to your system. every request to the system, even when uploading a file, the theme and plugins might be loaded, which means it can slow down the process, and make it take longer for the file to complete the upload process. i had a project where a client wanted every single plugin under the sun installed, even if they weren’t being used. the performance of the whole server was just terrible and we had to prune them to get any sort of performance. also, wordpress does have limitations on the size of files that can be uploaded. so, if you’re uploading a large file, that can be a problem as well, and could make the upload appear as if it's extremely slow, when it is actually just doing the upload but it's taking a longer time because of the file size itself. you may need to configure some php settings to allow bigger files to be uploaded, things like `upload_max_filesize` or `post_max_size`. also make sure the `php` memory limit is high enough. usually in the `php.ini` file.

here are a few code snippets that can help you test the issue:

first, checking for disk i/o with `iotop`. install iotop if you don't have it:

```bash
sudo apt update && sudo apt install iotop
```

then, run it and see what the usage looks like:

```bash
sudo iotop -ao
```

this will show you real time disk i/o use, and will help you identify if disk i/o is a problem.

second, a basic `rsync` command to upload a folder using ssh and compression. this will significantly improve speed over ftp in most cases:

```bash
rsync -avz -e ssh /path/to/local/folder user@your_lightsail_ip:/path/to/remote/folder
```

breakdown:
`a`: archive mode, which preserves all the file attributes and timestamps.
`v`: verbose, which prints the files that are being transferred.
`z`: compression, which helps speed up the transfer if you have a bottleneck with internet speed.
`-e ssh`: use ssh for the transfer.

this will upload the local folder content to the remote folder. if the folder does not exists, then it will be created automatically.

third, a sample php configuration values in `php.ini` that you can tweak. these values may vary depending on your specific php version and server setup:

```ini
upload_max_filesize = 64M
post_max_size = 64M
memory_limit = 256M
max_execution_time = 300
```
make sure to restart the webserver after editing this file.

about resources, instead of links i can recommend some really good books:
*   “unix system programming” by kay a. robbins and steven robbins. this book will give you a better grasp of the basics of linux systems. it's a deep technical book that covers many aspects of the operating system, not only disk i/o, but also processes, memory, and signals. a must read if you're dealing with the linux ecosystem.
*   “understanding the linux kernel” by daniel p. bovet and marco cesati. it goes deep on how the kernel works, and why some things are the way they are. it's very helpful to understand why disk i/o can be a problem with linux in general and will give you a general understanding on how most things actually work in any linux system.
*   “high performance mysql” by barron schwartz, peter zaïtsev, vadim tkachenko. while this book focuses on mysql, it’s a really solid reference for performance tuning in general and provides lots of information that can be applied to web servers in general such as load balancing, monitoring, disk i/o, query optimization, and other really useful details.

and finally a joke: why was the ftp server always invited to parties? because it was good at *transferring* the good vibes!

anyways, hope this helps, let me know if you have other specific questions or details. good luck with your server.
