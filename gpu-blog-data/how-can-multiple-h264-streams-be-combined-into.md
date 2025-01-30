---
title: "How can multiple H.264 streams be combined into a single H.264 stream?"
date: "2025-01-30"
id: "how-can-multiple-h264-streams-be-combined-into"
---
A common challenge in video processing involves merging several independent H.264 encoded video streams into a single, coherent H.264 stream. My experience working with multi-camera surveillance systems and video conferencing applications has highlighted the intricacies of this process. Simply concatenating the encoded bitstreams is not a viable solution because H.264, like most video codecs, relies on sequence headers and inter-frame dependencies; merging raw bitstreams would result in a corrupted and unplayable output stream. Therefore, a more structured approach is required that involves decoding, arranging, and re-encoding the streams.

The core concept revolves around the following steps: first, each input H.264 stream must be decoded into raw video frames, typically in YUV color space. Second, these decoded frames are composited into a single frame, arranging them according to the desired layout, such as tiled, picture-in-picture, or side-by-side. Lastly, the composited frames are then re-encoded into a new H.264 stream. This is not a trivial task; it necessitates careful handling of timing, frame rates, and spatial resolutions.

Let's consider the first stage: decoding. This requires leveraging an H.264 decoder library. I've primarily worked with FFmpeg's `libavcodec` for this purpose due to its flexibility and robustness. The process typically involves feeding raw H.264 encoded data (Nalu units) into the decoder and receiving raw YUV pixel data as output. Since each stream will likely have its own sequence headers, care must be taken to properly initialize the decoder for each stream before attempting to decode. After the decoding stage, each frame will need to be represented as a buffer in memory.

The next crucial stage is composition. This is where the visual layout of the combined video is determined. I've often implemented composition using techniques involving direct pixel manipulation in YUV color space because it tends to be computationally more efficient. For instance, if we aim for a side-by-side view, each frame from the input streams will be scaled down to half of the width of the final frame. The pixels of each input frame are then placed adjacent to each other in the output buffer representing the composite frame. It's worth noting that maintaining correct aspect ratios and dealing with differing spatial resolutions of input streams is often required.

Finally, the composited frames are encoded back into an H.264 stream. This process is also handled by an H.264 encoder, typically provided by a library such as `libx264` or `nvenc`. I have used both extensively depending on whether hardware encoding is available and beneficial. Here, proper configuration of the encoder is important as it defines the quality, bit rate, and profile of the resulting output stream. It’s critical to manage frame rates carefully, ensuring that the encoded stream matches the desired output characteristics. The resulting encoded packets can then be output to file or passed to a streaming protocol.

The code examples below provide a conceptual view of these processes, focusing on the core ideas. In a real-world system, significant additional complexity arises including error handling, proper buffer management, synchronization between input streams, and ensuring real-time performance.

**Example 1: Decoding H.264 and Obtaining YUV Frames**

```c
// Assume we have a function to read NALU's from a file or socket
AVPacket* readNalu(FILE* input_file) {
    AVPacket *pkt = av_packet_alloc();
    // ... Code to read a NALU and put it into the pkt->data and pkt->size
    if (read_error) {
      av_packet_free(&pkt);
      return NULL;
    }
    return pkt;
}

void decode_h264(AVCodecContext* codecContext, AVPacket* pkt, AVFrame** frame) {
  int ret;
  // Send the packet to the decoder.
  ret = avcodec_send_packet(codecContext, pkt);
  if (ret < 0) {
    fprintf(stderr, "Error sending packet for decoding \n");
    return;
  }

  while (ret >= 0) {
     ret = avcodec_receive_frame(codecContext, *frame);
     if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return;
      }
     else if (ret < 0) {
         fprintf(stderr, "Error during decoding \n");
         return;
     }
    // At this point *frame contains the decoded frame
      break;  // Assume one frame per call
  }
}

int main() {
    AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext *codecContext = avcodec_alloc_context3(codec);

    // ... Configure the codec context, error handling omitted.
    avcodec_open2(codecContext, codec, NULL);
    AVPacket* packet;
    AVFrame *frame = av_frame_alloc();

    FILE* inputFile = fopen("input.h264", "rb");
    while((packet = readNalu(inputFile)) != NULL) {
        decode_h264(codecContext, packet, &frame);

        // 'frame' now contains a decoded YUV frame
        // At this point further processing or compositing could occur
       av_packet_unref(packet);
    }
    av_frame_free(&frame);
    avcodec_close(codecContext);
    avcodec_free_context(&codecContext);
    fclose(inputFile);
    return 0;
}
```
*Commentary:* This code snippet highlights the core decoding process. It reads NAL units, sends them to the H.264 decoder, and retrieves decoded frames in YUV format. The actual logic for allocating codec contexts, parsing parameters from sequence header, error checking, and handling of end-of-stream are excluded for brevity, but are critical in a real implementation.

**Example 2: Simple Side-by-Side Composition**

```c
void compose_sidebyside(uint8_t* output_buffer, int output_width, int output_height,
                       uint8_t* input_buffer1, int input_width1, int input_height1,
                       uint8_t* input_buffer2, int input_width2, int input_height2) {
    int half_output_width = output_width / 2;
    // Assume YUV420p format. Planar buffer layout: Y, U, V planes
    int y_size_output = output_width * output_height;
    int u_size_output = y_size_output/4;
    int v_size_output = y_size_output/4;
    int y_size_input1 = input_width1 * input_height1;
    int u_size_input1 = y_size_input1/4;
    int v_size_input1 = y_size_input1/4;
    int y_size_input2 = input_width2 * input_height2;
    int u_size_input2 = y_size_input2/4;
    int v_size_input2 = y_size_input2/4;
    // Simplistic scaling assuming same aspect ratios for simplicity
    // In practice, a more robust scaling algorithm should be used
    for (int y = 0; y < output_height; ++y) {
         int source_y1 = (int)((double)y/output_height * input_height1);
         int source_y2 = (int)((double)y/output_height * input_height2);
        for (int x = 0; x < half_output_width; ++x) {
            int source_x1 = (int)((double)x/half_output_width * input_width1);
            int output_index = y * output_width + x;
            int source_index1 = source_y1 * input_width1 + source_x1;

             output_buffer[output_index] = input_buffer1[source_index1];

        }

        for (int x = half_output_width; x < output_width; ++x) {
            int source_x2 = (int)((double)(x-half_output_width)/half_output_width * input_width2);
           int output_index = y * output_width + x;
           int source_index2 = source_y2 * input_width2 + source_x2;

           output_buffer[output_index] = input_buffer2[source_index2];
        }
    }

      for (int y = 0; y < output_height/2; ++y) {
         int source_y1 = (int)((double)y/(output_height/2) * (input_height1/2));
         int source_y2 = (int)((double)y/(output_height/2) * (input_height2/2));
        for (int x = 0; x < half_output_width/2; ++x) {
            int source_x1 = (int)((double)x/(half_output_width/2) * (input_width1/2));
            int output_index = y * (output_width/2) + x + y_size_output;
             int source_index1 = source_y1 * (input_width1/2) + source_x1 + y_size_input1;
             output_buffer[output_index] = input_buffer1[source_index1];

         }
           for (int x = half_output_width/2; x < output_width/2; ++x) {
               int source_x2 = (int)((double)(x-(half_output_width/2))/(half_output_width/2) * (input_width2/2));
             int output_index = y * (output_width/2) + x + y_size_output;
            int source_index2 = source_y2 * (input_width2/2) + source_x2 + y_size_input2;
             output_buffer[output_index] = input_buffer2[source_index2];
         }
    }

      for (int y = 0; y < output_height/2; ++y) {
          int source_y1 = (int)((double)y/(output_height/2) * (input_height1/2));
          int source_y2 = (int)((double)y/(output_height/2) * (input_height2/2));
        for (int x = 0; x < half_output_width/2; ++x) {
            int source_x1 = (int)((double)x/(half_output_width/2) * (input_width1/2));
            int output_index = y * (output_width/2) + x + y_size_output+u_size_output;
            int source_index1 = source_y1 * (input_width1/2) + source_x1 + y_size_input1+u_size_input1;

            output_buffer[output_index] = input_buffer1[source_index1];
         }
            for (int x = half_output_width/2; x < output_width/2; ++x) {
                int source_x2 = (int)((double)(x-(half_output_width/2))/(half_output_width/2) * (input_width2/2));
             int output_index = y * (output_width/2) + x + y_size_output+u_size_output;
             int source_index2 = source_y2 * (input_width2/2) + source_x2 + y_size_input2+u_size_input2;
            output_buffer[output_index] = input_buffer2[source_index2];
          }
      }
}
```
*Commentary:* This function illustrates a simplistic implementation of side-by-side composition. It copies the pixels of the input frames to their respective locations within the output frame's buffer. A true implementation would handle scaling and format conversions more carefully and efficiently using libraries. The buffer layout in memory also includes UV plane for chroma data.

**Example 3: Encoding H.264 from YUV Frames**

```c
void encode_h264(AVCodecContext* codecContext, AVFrame* frame, AVPacket* pkt) {
    int ret;
    ret = avcodec_send_frame(codecContext, frame);
    if(ret < 0)
    {
        fprintf(stderr,"Error sending the frame for encoding\n");
        return;
    }
    while(ret >= 0) {
      ret = avcodec_receive_packet(codecContext, pkt);
      if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
           return;
      else if (ret < 0) {
        fprintf(stderr,"Error receiving packet during encoding\n");
         return;
      }
     // pkt->data and pkt->size now contain the encoded NALU data
        // Can write pkt->data to output file or stream
         break; // Assume one packet per call
    }
}


int main(){
    AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    AVCodecContext *codecContext = avcodec_alloc_context3(codec);
     // ... Configure the codec context, error handling omitted.
    avcodec_open2(codecContext, codec, NULL);

    AVFrame *frame = av_frame_alloc();
    // Assuming frame buffer is filled with output of composition logic
     //frame->data, frame->linesize, width, height need to be set properly
    AVPacket *packet = av_packet_alloc();
     encode_h264(codecContext, frame, packet);
   //output packet to file or network
   av_packet_unref(packet);
   av_packet_free(&packet);

   av_frame_free(&frame);
    avcodec_close(codecContext);
    avcodec_free_context(&codecContext);
    return 0;
}
```
*Commentary:* This code demonstrates the H.264 encoding process, accepting a raw frame as input and producing encoded H.264 NALU packets as output. Again, actual codec configuration, error handling, and output logic is not included for clarity.

For further learning I recommend exploring resources focusing on multimedia processing, specifically the documentation for libraries like FFmpeg, and libx264, along with literature on video encoding standards. Studying the YUV color space and video frame structures will also significantly aid in understanding the manipulation of raw video data. Specific books on video compression would be useful, as well as online courses that address video processing topics. These resources, combined with practical implementation, are invaluable for mastering the complex process of combining multiple video streams.
