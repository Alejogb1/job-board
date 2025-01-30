---
title: "How can I efficiently create videos from a series of Cairo canvases in C?"
date: "2025-01-30"
id: "how-can-i-efficiently-create-videos-from-a"
---
Generating video from Cairo canvases within a C environment necessitates a careful orchestration of pixel data, image encoding, and video container formats. My experience in high-performance graphics rendering has shown that treating this as a pipeline with distinct stages yields the most maintainable and efficient results. Direct manipulation of raw pixel data followed by library-based encoding provides a solid foundation.

The process fundamentally breaks down into three key areas: capturing the Cairo surface, encoding the frame, and multiplexing into a video file. Cairo, at its core, renders to surfaces, which are effectively in-memory representations of pixel data. This data, by default, is laid out in a manner compatible with various image formats, specifically ARGB32, which represents Red, Green, Blue, and Alpha channels with 8 bits allocated to each. While Cairo itself doesn't provide video encoding functionalities, these raw pixel buffers are the perfect input for external libraries.

Frame encoding, the next stage, is typically accomplished with libraries like libavcodec from the FFmpeg project. Libavcodec offers a wide range of encoding options, including popular codecs like H.264 (AVC), H.265 (HEVC), and VP9, each optimized for different performance trade-offs. Each codec has unique requirements regarding pixel format and encoding parameters; for example, H.264 commonly employs YUV color spaces, demanding a color space conversion from Cairo's ARGB32 representation before encoding.

Finally, the encoded frames need to be multiplexed, or packaged, into a container format like MP4 or MKV. This involves defining the video stream metadata and organizing the encoded frames in a manner that adheres to the chosen container's specifications. Libavformat, also part of the FFmpeg project, is crucial for handling this multiplexing phase, providing the mechanisms to create, manage, and write streams to various container file formats.

Let's examine some practical C code examples to solidify this concept:

**Example 1: Capturing a Cairo Surface**

This example demonstrates how to draw a simple red circle on a Cairo surface and then extract its pixel data for further use.

```c
#include <cairo.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480

unsigned char* capture_cairo_frame(cairo_surface_t *surface) {
    int stride;
    unsigned char *data;
    int width, height;

    width = cairo_image_surface_get_width(surface);
    height = cairo_image_surface_get_height(surface);
    stride = cairo_image_surface_get_stride(surface);
    data = cairo_image_surface_get_data(surface);

    if (!data) {
        fprintf(stderr, "Error retrieving surface data.\n");
        return NULL;
    }

    // Allocate memory to store a copy of the data
    unsigned char *frame_data = malloc(width * height * 4);
    if (!frame_data) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
     
    // Copy data from the surface to the allocated memory.
    for (int y = 0; y < height; y++) {
        memcpy(frame_data + y * width * 4, data + y * stride, width * 4);
    }

    return frame_data;
}


int main() {
    cairo_surface_t *surface;
    cairo_t *cr;

    surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, WIDTH, HEIGHT);
    if (!surface) {
        fprintf(stderr, "Error creating Cairo surface.\n");
        return 1;
    }
    cr = cairo_create(surface);
    
    // Draw a red circle
    cairo_set_source_rgb(cr, 1, 0, 0); // Red
    cairo_arc(cr, WIDTH / 2, HEIGHT / 2, 100, 0, 2 * M_PI);
    cairo_fill(cr);

    unsigned char *frame_data = capture_cairo_frame(surface);

    if(frame_data) {
       //Frame data is now accessible for encoding
      printf("Frame data captured successfully.\n");
       free(frame_data); 
    }
    
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    
    return 0;
}
```
**Commentary:**
*   `cairo_image_surface_create` initializes a Cairo surface with a specified format and dimensions. ARGB32 provides the layout we'll use for the pixels.
*   `cairo_create` creates a context for drawing onto the surface.
*   `cairo_set_source_rgb`, `cairo_arc`, and `cairo_fill` draw a basic red circle onto the surface.
*   `capture_cairo_frame` retrieves the underlying pixel data from the Cairo surface using `cairo_image_surface_get_data`, accounting for the surface stride, and returns a copied block of the raw pixel data. The stride may be different from `width * 4`, so copying with the `stride` parameter is crucial for correctness.
*   The returned `frame_data` pointer contains the pixel information in ARGB format. It requires proper handling including memory deallocation using `free` after use.

**Example 2: Encoding with Libavcodec (Simplified)**

This simplified example sketches the structure of encoding a single frame using libavcodec. It omits many details for clarity, such as handling encoder initialization, parameter setting, and complete error handling.

```c
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>

#define CODEC_ID AV_CODEC_ID_H264
#define WIDTH 640
#define HEIGHT 480

int encode_frame(unsigned char* frame_data, AVCodecContext *codec_context, AVPacket *pkt, AVFrame *frame) {
    int ret;

    // Allocate the frame buffer if needed. This should be done once
    if (!frame->data[0]) {
        ret = av_frame_get_buffer(frame, 0);
        if (ret < 0) {
            fprintf(stderr, "Error allocating frame buffer\n");
            return ret;
        }
    }

    // Copy the frame data into the AVFrame, converting pixel format if needed
    av_image_fill_arrays(frame->data, frame->linesize, frame_data, AV_PIX_FMT_RGBA, WIDTH, HEIGHT, 1);

    frame->pts++;

    ret = avcodec_send_frame(codec_context, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending frame to encoder\n");
        return ret;
    }

    ret = avcodec_receive_packet(codec_context, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
       return 0; // Needs more input or done
    }
    else if (ret < 0) {
      fprintf(stderr, "Error receiving packet from encoder\n");
      return ret;
    }


    return 1; // Frame encoded
}

int main() {
    AVCodec *codec;
    AVCodecContext *codec_context;
    AVPacket *pkt = NULL;
    AVFrame *frame = NULL;

    av_register_all();

    codec = avcodec_find_encoder(CODEC_ID);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        return 1;
    }
   
    codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return 1;
    }

    codec_context->bit_rate = 400000;
    codec_context->width = WIDTH;
    codec_context->height = HEIGHT;
    codec_context->time_base = (AVRational){1, 30};
    codec_context->framerate = (AVRational){30, 1};
    codec_context->gop_size = 10;
    codec_context->max_b_frames = 0;
    codec_context->pix_fmt = AV_PIX_FMT_YUV420P; // H.264 usually wants YUV420P
    
     // Open codec
    if (avcodec_open2(codec_context, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return 1;
    }

    pkt = av_packet_alloc();
    if(!pkt) {
      fprintf(stderr, "Error allocating packet.\n");
      return 1;
    }
   
   frame = av_frame_alloc();
   if(!frame) {
      fprintf(stderr, "Error allocating frame.\n");
      return 1;
   }
    frame->format = codec_context->pix_fmt;
    frame->width = codec_context->width;
    frame->height = codec_context->height;

    // Assume `frame_data` is the result from `capture_cairo_frame` function
    unsigned char *frame_data; //Populated elsewhere, e.g. from example 1
    
    // Assume `frame_data` was populated elsewhere from cairo
     frame_data = malloc(WIDTH * HEIGHT * 4); // Example only
    if(frame_data == NULL) {
      fprintf(stderr, "Memory allocation error.\n");
      return 1;
    }

    memset(frame_data, 100, WIDTH * HEIGHT * 4); // Simulate frame data


    if(encode_frame(frame_data, codec_context, pkt, frame) > 0 ) {
        // Process the encoded packet (write to file, etc.)
       printf("Packet data size: %d\n", pkt->size);
       av_packet_unref(pkt);
    }


    free(frame_data);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_context);


    return 0;
}
```

**Commentary:**

*   `avcodec_find_encoder` locates an appropriate encoder for a given codec (H.264 in this example).
*   `avcodec_alloc_context3` creates a context to hold parameters needed for encoding.
*   Encoder parameters, such as bitrate, frame rate, and pixel format, are set within the codec context. Notably, the pixel format must be compatible with the selected encoder; H.264 often requires `AV_PIX_FMT_YUV420P`.
*   `avcodec_send_frame` passes a `AVFrame` to the encoder, and `avcodec_receive_packet` retrieves the encoded packet. The frame data is copied using `av_image_fill_arrays`.
*   The encoded packet is unreferenced using `av_packet_unref`.
*   Important note: this is an oversimplified example. The loop of sending and receiving should also be checked for cases when `avcodec_receive_packet` returns `EAGAIN` or `EOF` which implies that we need to send more data.
**Example 3: Multiplexing with Libavformat (Simplified)**
```c
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <stdio.h>
#include <stdlib.h>

#define CODEC_ID AV_CODEC_ID_H264
#define WIDTH 640
#define HEIGHT 480


int main() {
    AVFormatContext *output_format_context = NULL;
    AVOutputFormat *output_format = NULL;
    AVStream *video_stream = NULL;
    AVCodec *codec = NULL;
    AVCodecContext *codec_context = NULL;
    AVPacket pkt;
    AVFrame *frame = NULL;
    int ret;


     av_register_all();

    // Select the output container format (MP4 in this example)
    output_format = av_guess_format(NULL, "output.mp4", NULL);
    if (!output_format) {
        fprintf(stderr, "Could not guess output format.\n");
        return 1;
    }
  
    // Allocate output format context
    output_format_context = avformat_alloc_context();
    if (!output_format_context) {
        fprintf(stderr, "Could not allocate output context.\n");
        return 1;
    }

    output_format_context->oformat = output_format;
    
    // Open the output file
    if(avio_open(&output_format_context->pb, "output.mp4", AVIO_FLAG_WRITE) < 0)
    {
      fprintf(stderr, "Could not open output file.\n");
      return 1;
    }


    // Find the encoder
     codec = avcodec_find_encoder(CODEC_ID);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        return 1;
    }

    // Allocate the codec context
    codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return 1;
    }

    codec_context->bit_rate = 400000;
    codec_context->width = WIDTH;
    codec_context->height = HEIGHT;
    codec_context->time_base = (AVRational){1, 30};
    codec_context->framerate = (AVRational){30, 1};
    codec_context->gop_size = 10;
    codec_context->max_b_frames = 0;
    codec_context->pix_fmt = AV_PIX_FMT_YUV420P; // H.264 usually wants YUV420P
    
    if (avcodec_open2(codec_context, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return 1;
    }

    // Create a new stream
    video_stream = avformat_new_stream(output_format_context, NULL);
    if (!video_stream) {
        fprintf(stderr, "Failed to create video stream.\n");
        return 1;
    }

     // Copy codec parameters to the video stream
    if(avcodec_parameters_from_context(video_stream->codecpar, codec_context) < 0) {
      fprintf(stderr, "Error copying codec parameters to stream.\n");
      return 1;
    }

     // Write the header for the file.
     if(avformat_write_header(output_format_context, NULL) < 0) {
         fprintf(stderr, "Error writing output file header.\n");
         return 1;
     }

    // Example of sending data to the stream (This would be in a loop)
    frame = av_frame_alloc();
    if(!frame) {
      fprintf(stderr, "Error allocating frame\n");
      return 1;
    }

    frame->format = codec_context->pix_fmt;
    frame->width = codec_context->width;
    frame->height = codec_context->height;


    unsigned char *frame_data;
    frame_data = malloc(WIDTH * HEIGHT * 4);
    if(frame_data == NULL) {
      fprintf(stderr, "Memory allocation error.\n");
      return 1;
    }

    memset(frame_data, 100, WIDTH * HEIGHT * 4);

    pkt.data = NULL;
    pkt.size = 0;

    int frame_encoded = 0;
    // Example: Encode and write a single frame.
    if((frame_encoded = encode_frame(frame_data, codec_context, &pkt, frame)) > 0 )
    {
      
        pkt.stream_index = video_stream->index; // Tell which stream
         if(av_interleaved_write_frame(output_format_context, &pkt) < 0) {
            fprintf(stderr, "Error writing frame.\n");
         }

        av_packet_unref(&pkt);
    }
    
    free(frame_data);
    av_frame_free(&frame);

     // Write the end of file.
     if(av_write_trailer(output_format_context) < 0 ) {
        fprintf(stderr, "Error writing the end of the output file.\n");
     }

     avio_close(output_format_context->pb);
    avformat_free_context(output_format_context);
    avcodec_free_context(&codec_context);

    return 0;
}
```

**Commentary:**
*   `av_guess_format` infers the desired output format based on the filename.
*   `avformat_alloc_context` allocates a new context to hold the parameters for the output file.
*   `avio_open` opens the output file in write mode.
*   A new video stream is added to the format context via `avformat_new_stream`.
*   Codec parameters are copied to the new video stream using `avcodec_parameters_from_context`.
*   `avformat_write_header` writes the file's header with stream data.
*   The `encode_frame` function from the previous example is used to encode a frame into a packet.
*   `av_interleaved_write_frame` writes the encoded packet to the output file.
*   `av_write_trailer` writes the end of the output file.

For further in-depth understanding, I strongly recommend exploring these resources:
*   The official FFmpeg documentation, which is comprehensive and covers all aspects of encoding and multiplexing.
*   Cairo’s documentation, which elucidates the functionalities of drawing on surfaces.
*   Various video processing blog posts and articles, which provide practical insights into the domain.
*   Online forums for FFmpeg and graphics programming for community-based knowledge sharing.

These references, when combined with persistent experimentation and hands-on practice, will equip you with the expertise to develop an efficient and robust video generation pipeline based on Cairo canvases in C.
