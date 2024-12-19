---
title: "How to Get the first frame or thumbnail for a video on a phone storage (gallery) or link?"
date: "2024-12-15"
id: "how-to-get-the-first-frame-or-thumbnail-for-a-video-on-a-phone-storage-gallery-or-link"
---

alright, so you're asking about grabbing that initial frame, the thumbnail, from a video, whether it's chilling in the phone's gallery or hanging out on a link somewhere, right? i've been down this rabbit hole more times than i care to count, and trust me, there are some gotchas. let's break it down, shall we?

first off, the approach varies a bit depending on where the video's coming from. if it's a file on the phone, we can use native api's to get to it directly. but if it’s a url, things get a bit more… involved.

let's tackle the local file first. assuming you're on android, we're going to leverage the `mediaMetadataRetriever` class. this guy is pretty much a swiss army knife for media info. it can give us durations, codec details, and, of course, thumbnail frames. here's a snippet in kotlin, because frankly, java feels like using a rotary phone these days:

```kotlin
import android.media.MediaMetadataRetriever
import android.graphics.Bitmap
import java.io.File

fun getThumbnailFromLocalFile(filePath: String): Bitmap? {
    val retriever = MediaMetadataRetriever()
    try {
        retriever.setDataSource(filePath)
        return retriever.frameAtTime
    } catch (e: Exception) {
        // log the exception here instead of swallowing it like a bad pill
        e.printStackTrace()
        return null
    } finally {
         retriever.release()
    }

}
```

this is pretty straightforward. we instantiate the `mediametadataretriever`, give it the filepath, ask it to snag the frame at time zero (the first frame), and then clean up. it is absolutely critical to release that resource in the `finally` block, otherwise you end up with zombie retrievers chewing up your memory. i had a horrible memory leak issue once that this caused me and it took me days to track down. i swear the app was eating more ram that my actual os. i think the code in my case back then was java, which didn't really make me want to switch back to it.

now, about that `filepath`. on android, it's usually something you get from the `contentResolver` and `uri` obtained from the user's gallery. dealing with that is another can of worms. you should look at the `contentResolver.query` method and the `mediaStore.video.media` columns, specifically the `mediaStore.mediaColumns.data` column. there's no real magic here, just good old android-api-fu. you might also run into permission issues. make sure you request the `read_external_storage` permission or something equivalent. android likes to change things from release to release.

let's move on to the web, the world of urls, this gets trickier. you cant just go to any url and ask for a thumbnail. a lot of video streaming sites use adaptive bitrate streaming, which means that the raw video files are not readily accessible and sometimes they dont even show you the video in the html. you need to parse it, if you want to stream a video from there. sometimes, depending on how the site was built, it is easier than other times.

when it comes to urls, things arent straight forward, you cant just feed a link to the `mediaMetadataRetriever` and expect results. you generally have to download the video, at least partially. this is because media metadata is usually located in the beginning of the video file. downloading it takes some time, and this is not ideal. its usually better to check if there are a pre generated thumbnail available via the html, rather than actually grabbing a part of the video file. so lets explore other options: if its a youtube video, for example, they usually generate thumbnails in different resolutions and they make them available via predictable links, you can easily construct the link by knowing the video id.

but lets assume that you dont have a thumbnail. and that you have the url, so we are going to grab a fragment of the video, just enough to extract the metadata we need. we will use `okhttp` for downloading. why `okhttp` and not the standard java url connection? mostly because i hate the standard java connection, its more cumbersome, and okhttp is a more mature and a better api in my opinion.

```kotlin
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.InputStream
import android.media.MediaMetadataRetriever
import android.graphics.Bitmap
import java.io.ByteArrayInputStream

fun getThumbnailFromUrl(url: String): Bitmap? {
    val client = OkHttpClient()
    val request = Request.Builder().url(url).build()

    try {
        val response: Response = client.newCall(request).execute()
        if (!response.isSuccessful) {
             // something went wrong
            return null
        }
        val inputStream: InputStream? = response.body?.byteStream()
        if(inputStream == null){
            return null
        }

        val partialData = inputStream.readNBytes(1024 * 1024 * 2) //2 mb, just to grab the start.
        val mediaRetriever = MediaMetadataRetriever()
        val stream = ByteArrayInputStream(partialData)
        mediaRetriever.setDataSource(stream.fd)
        val frame = mediaRetriever.frameAtTime
        mediaRetriever.release()
        inputStream.close()
        return frame

    } catch (e: Exception) {
          // log it
        e.printStackTrace()
        return null

    }
}
```

here's what's happening. we create an `okhttpclient`, build a `request`, make that request to the `url`, and get the response. after checking if the response is ok, we open an inputstream on the body of the response and read the first 2mb, to be sure. this is very optimistic but should work for most cases, as metadata is usually at the begining of the file, we create a bytearrayinputstream, and finally, we do the same we did for the local case using `mediaMetadataRetriever` and releasing the resource. i had issues where i was downloading more than i needed, which was unnecessary, so i started to download the first 2mb.

the `1024 * 1024 * 2` bit means 2 megabytes. tweak it as needed, but keep in mind you don't need to download the entire video. downloading too little may result in the `mediaMetadataRetriever` not finding what it needs, and then it will not generate a thumbnail. you'll need a decent sized part of the video file for the metadata to be available.

now, be cautious: network operations should never be done on the main thread, do not make this a synchronous operation, use `coroutines` or `async tasks` for this. i had a case where i completely crashed the app because the ui thread was stuck waiting for a big download to complete. dont do this.

and here's the final piece, dealing with android's `uri`. you will have to get the file `uri` from android's gallery, you will get the file uri from something like a content resolver. and there are many edge cases, like some weird phones storing things in a certain way that is hard to parse. but i think that for most of your needs you will get the `uri`, extract a file path using a content resolver and then just feed that to the first method. the code is roughly this:

```kotlin
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import java.io.File

fun getPathFromUri(context: Context, uri: Uri): String? {
    val projection = arrayOf(MediaStore.MediaColumns.DATA)
    val cursor = context.contentResolver.query(uri, projection, null, null, null)
    cursor?.use {
        if (it.moveToFirst()) {
            val columnIndex = it.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA)
             val filePath = it.getString(columnIndex)
            return filePath
        }
    }
    return null
}
```

here we create a `cursor` to query the database, we look for the `data` field and return the value, as a string, that will be our `filepath` then we can use the method that i mentioned before. and dont forget to request the permission to read the external storage.

now, i know that there are lots of libraries that do all of these. i am not saying not to use them, but i usually try to avoid them because i prefer to have as few dependencies as possible. plus sometimes the libraries hide the complexity under layers of abstractions, and then when they fail, you end up with an exception that you have no clue where it came from, it is hard to debug. i have been there. debugging some third party obscure bug was very difficult in the past, but that may be just my opinion. but in my experience you are usually much better off using the standard api and writing your own stuff, and it's more flexible.

if you want to really dive into the media stuff, you should check out the official android documentation, of course, but there's also some older papers on video encoding and media containers that could be helpful if you are interested. if you're into understanding why things work the way they do, look for resources about mpeg-4 file formats, that should keep you occupied for a while. one classic is "understanding mpeg-4" by ikeomi. the iso/iec 14496-14 standard is available online also, but its a bit dense. and if you want to go down the web streaming way, you can read about mpeg dash or hls streaming, there is plenty of material available online about this.

one last thing, dont forget to test your code in different phones because some manufacturers may have specific implementations. i had a case where a random brand was using a nonstandard media format that was not supported by `mediametadataretriever` out of the box. and that was not fun. good luck!
