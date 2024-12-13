---
title: "how to use remote access iot over internet windows?"
date: "2024-12-13"
id: "how-to-use-remote-access-iot-over-internet-windows"
---

Alright so you wanna get your IoT thingamajig accessible from anywhere using Windows over the internet huh Been there done that got the scars to prove it. It’s a classic problem a rite of passage really. Been wrestling with this kinda stuff since dial-up was a thing. Back in the day I even used a literal serial port and modem to access my custom built weather station from the library computer yeah those were the days… No fancy cloud platforms or APIs just pure grit and a very slow connection.

Okay let's break this down. Forget the smoke and mirrors you need a solid plan. You got your IoT device which I'm assuming runs some sort of web server or can be controlled via a network protocol and you got your Windows machine which you want to use to poke it remotely.

First thing first networking basics. Your IoT device needs an IP address on the same network as your router and the router needs to be able to forward a specific port to that IP. We are talking port forwarding folks classic stuff you need to dive into your router settings its different for every router manufacturer but the basic gist is you create a rule something like:

*   **External Port:** 8080
*   **Internal Port:** 80 (assuming your IoT device is on port 80)
*   **IP Address:** the internal IP of your IoT gizmo typically something like 192.168.1.xxx

Don't forget to save and reboot your router if that's a thing for your model.

Okay that's the simple part now comes the Windows bit. Assuming you're not behind a double NAT (which would be a headache for another day) you need your public IP address. Go to any website that shows your IP "what is my ip" on Google works just fine.

Now you've got your public IP and your forwarded port. In theory you should now be able to type in your public ip followed by a colon and then that forwarded port number into any browser from anywhere with internet connection that is not blocked by other firewall or network policy. So like `http://your.public.ip:8080`.

If it doesn't work check your firewall settings on both the router and your IoT device itself some devices have a built-in firewall or security software that needs some love and care to allow external access.

Okay moving on let's talk about actual methods for remote access beyond just basic HTTP. Here’s a couple of different approaches I've used in the past with code snippets cause I know you like that.

**Option 1: SSH tunneling**

If your IoT thingamajig is something running Linux or something that can run an SSH server this is your safest best friend. It’s secure you can forward multiple ports and it’s generally reliable. First you need an SSH server on your IoT device. Install `openssh-server` if its a linux device like a Raspberry Pi usually its already installed but lets cover all the angles:

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

Make sure SSH is running properly also you should not do this without setting up a strong password for your user and preferrably configuring your server to use key based login. Now you need an SSH client on your Windows machine. PuTTY is a good free choice.

Here's the trick you can use PuTTY to create a tunnel that forwards a local port on your Windows machine to the internal port on your IoT device. You connect to your IoT device via SSH and then specify the local port for the tunnel.

Let's say your IoT device has a web server on port 80 and you want to access it from your Windows machine using local port 9000. In PuTTY after setting the hostname to your public IP and port 22 you'd go to the Connection > SSH > Tunnels settings and configure as such:

*   **Source port:** 9000
*   **Destination:** 127.0.0.1:80 (it doesnt have to be that IP you can try other IPs but local loopback is better security)

Then click Add make sure the checkmark in the Local section is ticked then click open and you will now have an SSH connection and the tunnel will be ready for your needs. Now type in your browser on Windows `http://localhost:9000` and you should see your IoT device's web interface.

**Option 2: VPN**

If you are not a fan of port forwarding and SSH tunneling VPNs are a good middle ground. A VPN creates a secure encrypted tunnel between your Windows machine and your home network. This way your IoT device thinks you are on the local network and you don't have to expose ports directly to the internet.

There are many VPN solutions out there. WireGuard is a good one as its relatively simple. You can set up a WireGuard server on your router if it supports it or on another machine inside your home network.

Here is an example config for your Windows client configuration file its very basic to get a quick working setup. You should adjust some of these options based on your network environment and needs.

```
[Interface]
PrivateKey = <your private key for the windows client>
Address = 10.8.0.2/24
DNS = 1.1.1.1

[Peer]
PublicKey = <your server public key>
AllowedIPs = 0.0.0.0/0
Endpoint = your.public.ip:51820
PersistentKeepalive = 25
```

And then the example config for your server:

```
[Interface]
PrivateKey = <your server private key>
Address = 10.8.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <your client public key>
AllowedIPs = 10.8.0.2/32
```

Now when you connect to your VPN you are effectively "inside" your network you can access your IoT device directly using its internal IP.

**Option 3: Remote Desktop or RDP**

If you really want to be controlling the operating system on the IoT device instead of just using the webserver or an API you could use remote desktop. However in most cases its not desirable as it consumes resources and its more intensive as it shares an entire screen and graphical user interface. Usually IoT devices are meant to be headless which means without a monitor attached. Remote Desktop also can expose another attack vector so be careful when using RDP for IoT devices that are exposed to internet. On most versions of windows you can use the `mstsc.exe` command to connect to a remote machine by IP and port 3389 if it is enabled on the target IoT machine. If you have a linux based IoT device you would need to use a third party client with a VNC server on the device that can be connected over RDP like Remmina or similar.

Now I know someone will bring it up. Yes there are other fancier solutions like cloud platforms or specific IoT management software they have their own trade offs and also usually incur a cost. These solutions are usually overkill for many projects. I prefer the simpler approach that gives me full control and its mostly based on well established protocols.

For learning more about network security specifically on IoT its very important to read good literature. Look into Bruce Schneier's "Applied Cryptography" for a good deep dive on protocols. Also you need to understand network protocols in detail there is a great series of books by W. Richard Stevens "TCP/IP Illustrated" which are very detailed and explain the underlying mechanisms of network communication.

Okay so to recap. You need to:

1.  Forward a port on your router to your IoT device.
2.  Know your public IP.
3.  Choose your access method SSH tunneling VPN or direct port access.
4.  Always configure your network and device with strong passwords and consider encryption.

Remember this is not a plug and play solution. There's a learning curve and a whole lot of debugging along the way.

Oh and a funny story from my past I once spent 3 hours debugging a network issue only to realize the network cable was not fully plugged in its these small things that drive us crazy. Just a friendly reminder to always check the basics. So yeah happy coding and good luck with your IoT adventures. Feel free to reply if you have some specific questions and I'll do my best.
