---
title: "How can large files be efficiently uploaded to a Rails app from a React/Electron front-end?"
date: "2024-12-23"
id: "how-can-large-files-be-efficiently-uploaded-to-a-rails-app-from-a-reactelectron-front-end"
---

Alright, let's tackle this. I remember a project a few years back – a large-scale document management system – where we faced a similar challenge: efficiently uploading hefty files from our Electron-based desktop client to our Rails backend. Handling that efficiently involved a mix of techniques, each playing a crucial role. We definitely couldn’t just dump the entire file in a single request; that would've choked everything.

The crux of the issue is that HTTP, by its nature, isn't designed for large, continuous data streams. We needed to slice these files into smaller, manageable pieces and transfer them sequentially, piecing them back together on the server. This is where the concept of "chunked uploads" or "multipart uploads" becomes essential.

Fundamentally, a chunked upload breaks down a large file into smaller segments, each transmitted individually. This has several advantages:

1.  **Reduced Memory Footprint:** Neither the client nor server needs to load the entire file into memory at once. This is critical, especially for large media files.
2.  **Resiliency:** If a transmission fails, only the current chunk needs to be resent, not the whole file.
3.  **Progress Tracking:** We can easily monitor the upload process, providing the user with feedback on upload progress.

Here’s how we implemented it. On the React/Electron side, we'd leverage the `FileReader` API to read portions of the file and then use `fetch` to send them to our Rails API:

```javascript
async function uploadChunk(file, start, end, uploadId) {
  const blob = file.slice(start, end);
  const formData = new FormData();
  formData.append('chunk', blob, file.name);
  formData.append('uploadId', uploadId);
  formData.append('chunkIndex', Math.floor(start/CHUNK_SIZE))

  try {
    const response = await fetch('/api/upload-chunk', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();

  } catch(error) {
    console.error("Error uploading chunk:", error);
      throw error;
  }
}

async function initiateUpload(file) {
  const response = await fetch('/api/initiate-upload', {method: 'POST'});
  if (!response.ok) throw new Error('Failed to initiate upload');
  return response.json();
}


async function uploadFile(file) {
    const CHUNK_SIZE = 1024 * 1024 * 2; // 2MB chunks
    let start = 0;
    let uploadId;

    try {
        const initData = await initiateUpload(file);
        uploadId = initData.uploadId;
        while(start < file.size) {
            const end = Math.min(start + CHUNK_SIZE, file.size);
            await uploadChunk(file, start, end, uploadId);
            start = end;
        }
        //Call finaliseUpload
        return finalizeUpload(uploadId, file.name);
      }
    catch (e) {
        console.error("error:", e)
        throw e;
    }
}


async function finalizeUpload(uploadId, filename) {
    const response = await fetch('/api/finalize-upload', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
          },
        body: JSON.stringify({uploadId: uploadId, filename: filename})
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}
```

Here, `CHUNK_SIZE` determines the size of each file segment (2MB in this example), and `uploadId` ensures that we can associate the chunks correctly to a single file. The important part is slicing the file using `file.slice` and setting `multipart/form-data` by using the `FormData` object.

On the Rails side, we implemented a few API endpoints: `/api/initiate-upload`, `/api/upload-chunk`, and `/api/finalize-upload`.

The `/api/initiate-upload` endpoint creates a temporary storage location for the chunks using `SecureRandom.uuid` as an ID. It then responds with that identifier which gets passed to the client.

```ruby
# app/controllers/api/upload_controller.rb
class Api::UploadController < ApplicationController
  skip_before_action :verify_authenticity_token

  def initiate_upload
    upload_id = SecureRandom.uuid
    temp_dir = Rails.root.join('tmp', 'uploads', upload_id)
    Dir.mkdir(temp_dir) unless Dir.exist?(temp_dir)
    render json: { uploadId: upload_id }, status: :ok
  end
end
```

The `/api/upload-chunk` endpoint accepts each chunk from the client, verifying the uploadId and chunkIndex before appending it to a file in the temporary storage location.

```ruby
# app/controllers/api/upload_controller.rb
  def upload_chunk
    upload_id = params[:uploadId]
    chunk_index = params[:chunkIndex].to_i
    chunk = params[:chunk]

    temp_dir = Rails.root.join('tmp', 'uploads', upload_id)
    if !Dir.exist?(temp_dir)
      return render json: { error: 'Invalid upload ID' }, status: :bad_request
    end
     filename = params[:chunk].original_filename
     temp_path = Rails.root.join(temp_dir, "chunk_#{chunk_index}")
     File.open(temp_path, 'wb') { |file| file.write(chunk.read) }
     render json: { status: 'Chunk uploaded' }, status: :ok
  end
```

Finally, when all chunks have been received, the `/api/finalize-upload` endpoint is called. This endpoint uses the `uploadId` to reassemble all the received chunks into the original file. It renames the final file to the desired name, clears the temporary upload directory, and stores the file in the appropriate location (in our case, a dedicated storage service).

```ruby
# app/controllers/api/upload_controller.rb
def finalize_upload
    upload_id = params[:uploadId]
    filename = params[:filename]
    temp_dir = Rails.root.join('tmp', 'uploads', upload_id)

    if !Dir.exist?(temp_dir)
      return render json: { error: 'Invalid upload ID' }, status: :bad_request
    end

    final_path = Rails.root.join('storage', filename)
    File.open(final_path, 'wb') do |final_file|
      Dir.glob(Rails.root.join(temp_dir, 'chunk_*'))
         .sort_by { |chunk_path| chunk_path.split('_').last.to_i }
         .each do |chunk_path|
            final_file.write(File.read(chunk_path))
            File.delete(chunk_path)
          end
    end

    Dir.rmdir(temp_dir)
    render json: { status: 'File uploaded successfully', filename: filename }, status: :ok
end
```

This entire process can be extended with techniques like resumable uploads. This would allow resuming interrupted uploads by keeping track of the chunks already sent. If an upload breaks we can check for existing chunks associated with an uploadId and continue from the last uploaded chunk.

I strongly recommend looking into the RFC 7578 specification on multipart/form-data if you want a very detailed understanding of the mechanics behind file uploads. For a comprehensive practical guide on handling file uploads, including chunked ones, I found that “Web Application Architecture: Principles, Protocols and Practices” by Leon Shklar, Richard A. Rosen is an invaluable resource. Understanding the theory in "High Performance Browser Networking" by Ilya Grigorik can also be beneficial to ensure optimal use of HTTP, especially when dealing with large files. Specifically, reading about HTTP/2 features, like multiplexing, can provide additional ideas to optimize your upload performance.

That, in a nutshell, is how we tackled file uploads in our system. The combination of chunking, a solid backend API, and client-side logic allowed us to transfer very large files efficiently and reliably. It's not a single solution, but rather a coordinated strategy that addresses the inherent limitations of browser-based file transfers.
