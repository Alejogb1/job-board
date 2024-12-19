---
title: "Why am I unable to decrypt backup files of Mail-in-a-box stored in S3 or locally?"
date: "2024-12-15"
id: "why-am-i-unable-to-decrypt-backup-files-of-mail-in-a-box-stored-in-s3-or-locally"
---

Okay so you're having trouble decrypting your Mail-in-a-box backups right Been there done that tons of times especially when I was first messing around with self-hosted email back in '16 before I even properly understood symmetric encryption versus asymmetric

Let’s break this down without getting too deep into fancy terms. Mail-in-a-box uses encryption for backups that’s a fact you can see it right there in the code and it's not a complicated idea once you actually get it but it can be a pain to debug if you haven’t done it before. This isn't your typical "oops I forgot the password" kind of problem though it definitely could be that I've done that a few times myself if you use some password manager make sure the passwords match I recommend bitwarden seriously

So the core issue is that these backups are not just zip files or tarballs; they’re encrypted using symmetric key encryption. In plain speak one key is used to encrypt the data and the same key is used to decrypt it which also means if your encryption key is lost or unavailable so is your mail content or well almost everything. Usually Mail-in-a-box defaults to using AES-256 it's pretty solid stuff and I mean you don't mess around with security especially with sensitive data like emails right

You can find that info too in their repo somewhere I think it's in the docs folder or a script or function name it's been a while to pinpoint it. They do a good job documenting but sometimes you just have to dig deeper and also do some real-world debugging especially with stuff like backup recovery.

I’m going to guess you’re not getting an error like "Invalid password" or something so let’s go through the common reasons why decryption might fail:

First and most important you need the correct encryption key. Mail-in-a-box generates this key when you set it up and it’s stored in the `/home/mailinabox/mailinabox.conf` file as the `BACKUP_PASSPHRASE` variable. This is not the web admin panel password nor any other user password. It's a unique randomly generated passphrase used only for backups. I'm pretty sure that thing is like 32 bytes of random data you can easily check its size in linux using cat pipe to wc command.

If you’re using S3 you’re probably also confused at this point like “but I’m not passing a password anywhere!” Mail-in-a-box handles the encryption locally before sending the data to S3. This is a security best practice you don’t send decrypted data to a third party unless you really trust them like your own NAS in your house but then again you should encrypt there too

So here's what we need to do to actually decrypt a backup. The basic approach is to use `openssl` which is pretty much the industry standard for crypto stuff if you aren't using that yet you definitely need to get familiar with it.

Let's say the backup file is named `backup_20240120.tar.gz.enc`. This is how you'd decrypt it assuming your encryption key is stored in a variable named `$BACKUP_KEY`:

```bash
BACKUP_KEY="your_backup_passphrase_from_mailinabox_conf"
openssl aes-256-cbc -d -in backup_20240120.tar.gz.enc -out backup_20240120.tar.gz -pass pass:"$BACKUP_KEY"
```

This command decrypts `backup_20240120.tar.gz.enc` using AES-256-CBC mode using the key you specified and the output is a decrypted file called `backup_20240120.tar.gz`. The `-d` flag specifies we're decrypting also `-in` and `-out` are the input and output file paths pretty standard for linux apps that handle files. You’ll need to replace your\_backup\_passphrase\_from\_mailinabox\_conf with the actual passphrase from `/home/mailinabox/mailinabox.conf`. Be super sure that you copy that passphrase correctly even a single character wrong and you're done.

Another common problem especially for large backups is the `-pass` argument. On some Linux distros or if you have a weird configuration of bash shell it might have trouble with the variable substitution. Instead you can try piping the key into the openssl command like this:

```bash
echo "your_backup_passphrase_from_mailinabox_conf" | openssl aes-256-cbc -d -in backup_20240120.tar.gz.enc -out backup_20240120.tar.gz -pass stdin
```

This command does the same but the key is being passed via stdin rather than an argument which is a bit more reliable at times. It’s always good to have a few tricks up your sleeve especially when debugging something like this you know.

Now after you have a decrypted tar archive you can extract it using `tar -xzvf backup_20240120.tar.gz`. If you get an error here it means that something went wrong with the decryption or that the backup file was corrupted during upload/download but most likely it was the key you typed incorrectly. Always double check the key. I had one situation a few years back where I was struggling with decrypting for hours only to realize that the issue was that the key was saved on my password manager with a trailing space character at the end and after all those years I still check that space always. The joke's on me really isn't it?

One other thing that can go wrong is if you change the Mail-in-a-box configuration after the backup was made. Specifically the encryption key. If you've changed the `BACKUP_PASSPHRASE` after the backups were taken the old backups will be encrypted with the old key and the new backups with the new key. This is a very common mistake since people try to use a new stronger password or change it for some reason but they forgot to first decrypt all the backups before changing the passphrase it's like trying to open a door with the wrong key every single time.

Now there are other less likely scenarios. If you made any modifications to the backup process itself or the crypto libraries then you could also have some issues. Always make sure you're working on the most stable version and before doing that read the release notes carefully. This should be obvious but people do overlook that.

If for some reason you're still getting errors like `bad decrypt` it’s possible the backup itself is corrupted or you are missing an `iv` (Initialization Vector) which in that case you need to look deeper and try to understand how Mail-in-a-box does its backups. You need to read the source code in such cases this is the only way to debug very specific things. If you go that route familiarize yourself with cryptography the book "Applied Cryptography" by Bruce Schneier is a must read if you want to be proficient in this field, or at least be able to debug crypto issues. Also "Understanding Cryptography" by Christof Paar and Jan Pelzl is another good resource with some math and implementation examples.

Remember encryption and decryption are very precise operations even a small mistake will cause it to fail so be extremely careful when dealing with them.

To summarize:

1.  Get the `BACKUP_PASSPHRASE` from `/home/mailinabox/mailinabox.conf`.
2.  Use `openssl` with the correct key to decrypt the backup file.
3.  Double check the key, you can't check it enough times.
4.  If you have multiple backups make sure you're using the key that corresponds to the specific backup.
5.  If it still doesn’t work go read the Mail-in-a-box scripts and code.

If you have tried all of this and are still having issues provide some more info like the exact error messages you're getting or any changes you've made to the setup it will make troubleshooting easier. Also verify with the `openssl` command that the output file is not a zero-byte file or something since that also means that the key is not correct. In my debugging experience I have seen people getting empty files all the time and they think that they are doing something wrong with the commands when in fact it is just an incorrect decryption key.

And that’s pretty much everything you need to know at the moment. If you find any more problems feel free to ask. I have seen a lot of strange things regarding crypto and backups over the years. Also learn some more cryptography it is always a good skill to have.
