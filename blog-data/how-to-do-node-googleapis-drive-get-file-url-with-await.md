---
title: "How to do Node Googleapis Drive Get File Url With Await?"
date: "2024-12-14"
id: "how-to-do-node-googleapis-drive-get-file-url-with-await"
---

alright, so you're looking to grab a file's url from google drive using node.js and the googleapis library, specifically using async/await, got it. this is a classic, and i’ve definitely been down this rabbit hole more than once. it can feel a little clunky at first if you're not used to the api's quirks but stick with it, you will get it.

i remember back when i was first messing with this. it was like 2017, i think. i was trying to build this little script that would automatically back up my project files to drive. i had the whole authentication dance down, but getting those darned file urls was a pain. i spent like a whole evening banging my head against google’s docs, and i swear, they weren't as clear back then. turns out, the key is understanding how the `drive.files.get` method and specifically the `webViewLink` property work.

let’s get to it though. the primary thing is making sure that you’ve got your googleapis setup and your auth process correct. i am not going over that, i will assume you already have that working and you have your authentication setup with the credentials json file and the token saved. but if not read the google’s documentation.

the core of this involves making an asynchronous function and using `await` to manage the promise returned by the `drive.files.get` call. the important thing here is also making sure you are specifying the fields you need, like `webViewLink`. if you don't specify the fields the request will return you a lot of information that is totally useless to get the url.

here is the gist, in code, of how to do it:

```javascript
const { google } = require('googleapis');

async function getFileUrl(fileId, drive) {
  try {
    const res = await drive.files.get({
      fileId: fileId,
      fields: 'webViewLink',
    });

    return res.data.webViewLink;
  } catch (err) {
    console.error('error getting file url:', err);
    return null;
  }
}

async function main(){
   // your authentication setup here
    const auth = new google.auth.GoogleAuth({
         keyFile: "credentials.json",
         scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });
    const drive = google.drive({ version: 'v3', auth });

    const fileId = 'your_file_id_here'; // replace with your file id
    const fileUrl = await getFileUrl(fileId, drive);

     if(fileUrl){
      console.log('file url:', fileUrl);
    } else {
        console.log("could not get file url");
    }
}

main();
```

ok, let me break this down for you bit by bit.

1.  **import `google`:** we pull in the `googleapis` library so we can actually interact with google drive. it is required to execute anything on google drive, the code will fail without it.
2.  **`async function getFileUrl(fileId, drive)`:** this declares an async function called `getFileUrl` that will take the file id and a drive object as input.
3.  **`try...catch`:** this is a basic error handler, we try to get the file url and if any error happens like if the file id doesn't exist we capture it and return `null`. very good practice to use this all the time.
4.  **`await drive.files.get(...)`:** this is where the magic happens. we use the `drive.files.get` method and the `await` keyword to wait for the promise to return. we tell the api we just want the 'webViewLink' by specifying the `fields`. this is key.
5.  **`return res.data.webViewLink`:** if everything goes fine the `webViewLink` is then returned. the `res` object is a huge javascript object with all the info of the requested resource so it is crucial to know the structure of this data in order to navigate to what we want.
6. **`main` function:** this is how we would normally call the function, notice the authentication setup here with the required scopes for read only access, you can modify this according to your needs.

now, let’s say you want to download the file too, not only get the url. this is a little more involved but totally doable. you’ll use the `drive.files.get` method again but with a different approach. you'll request the `res` object as a stream and then pipe it to a file.

here's how you'd do that:

```javascript
const { google } = require('googleapis');
const fs = require('fs');
const path = require('path');

async function downloadFile(fileId, drive, destinationPath) {
  try {
    const res = await drive.files.get({
      fileId: fileId,
      alt: 'media'
    }, { responseType: 'stream' });

      const dest = fs.createWriteStream(destinationPath);

      res.data.pipe(dest);

      return new Promise((resolve, reject) => {
        dest.on('finish', resolve);
        dest.on('error', reject);
      });

    } catch (err) {
    console.error('error downloading file:', err);
      return null;
    }
}

async function main(){
    // your authentication setup here
    const auth = new google.auth.GoogleAuth({
        keyFile: "credentials.json",
        scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });
    const drive = google.drive({ version: 'v3', auth });
    const fileId = 'your_file_id_here'; // replace with your file id
    const filename = 'downloaded_file.pdf';
    const destPath = path.join(__dirname, filename);

    const downloadStatus = await downloadFile(fileId, drive, destPath);

    if(downloadStatus === null){
        console.log("could not download file");
    } else {
        console.log("file downloaded successfully to: ", destPath)
    }
}


main();
```

this example is a bit more involved, check this out:

1.  **`fs` and `path`:** here we import `fs` and `path` modules to help deal with files and filepaths.
2.  **`alt: 'media'`:** this crucial parameter tells the api that we want to get the raw data of the file. it is essential to getting the file's content
3.  **`responseType: 'stream'`:** by specifying stream we can get a readable stream that we can then pipe into a writable stream using `fs`.
4. **`res.data.pipe(dest)`:** the key point here where we pipe the readable stream into our destination writable stream
5. **`return new Promise(...)`**: this is required to handle the `finish` and `error` events of the stream so we can tell when the file has finished downloading or if it failed to download. this is necessary because the `pipe` function is asynchronous.
6. **`destinationPath`**: this will tell where to store the downloaded file.

a couple of things to consider, google drive has different kinds of file representations, for example a google docs file does not have a direct binary representation. google will convert the files you request for download to the specified mime type if available, check the docs for more info about this. so if the file is not a common file format like a pdf or a image, you may need to handle this conversion process.

lastly, if you are dealing with a large number of files, you might want to look into using batch requests to improve performance. the googleapis library allows you to group requests and send them at once, reducing network overhead and this will increase your script performance. i am not going to go over it in this answer but here is how you would do it with this previous examples in the simplest terms:

```javascript
const { google } = require('googleapis');
const fs = require('fs');
const path = require('path');

async function batchDownloadFiles(fileIds, drive, destinationFolder) {
    try {
        const batch = drive.newBatch();

        fileIds.forEach(fileId => {
            batch.add({
                method: 'GET',
                url: `/drive/v3/files/${fileId}`,
                alt: 'media',
                responseType: 'stream'
                }, (err, res) => {
                   if (err) {
                        console.error(`error getting file ${fileId}:`, err);
                        return;
                    }
                    const filename = `${fileId}.pdf`;
                    const destPath = path.join(destinationFolder, filename);
                    const dest = fs.createWriteStream(destPath);
                    res.data.pipe(dest);
                    dest.on('finish', () => console.log(`file ${fileId} downloaded successfully.`));
                    dest.on('error', (err) => console.log(`file ${fileId} failed to download. ${err}`));
            })
        })
       await batch.execute();
    } catch (err){
        console.error('error downloading files:', err);
    }
}

async function main(){
    // your authentication setup here
    const auth = new google.auth.GoogleAuth({
         keyFile: "credentials.json",
         scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });
    const drive = google.drive({ version: 'v3', auth });
    const fileIds = ['your_file_id_here_1', 'your_file_id_here_2', 'your_file_id_here_3'];
    const destFolder = path.join(__dirname, "downloads");
    fs.mkdirSync(destFolder, {recursive: true});

    await batchDownloadFiles(fileIds, drive, destFolder);
}

main();
```

a small joke if you will. i once wrote a script that deleted all my google drive files instead of just backing them up, thank goodness for the trash bin. you have to be really careful when using these apis.

as for further reading. for deep dive in promises in javascript you cannot skip "You Don't Know JS: Async & Performance" by kyle simpson, it's very good to truly understand how async works in javascript. and for everything google api related check "google apis documentation" it can be tedious at first but eventually you will get used to it, and you will be able to navigate it easily.

that's pretty much it, hope it helps and good luck with your code!
