---
title: "How can I efficiently extract photos from a zip file in C# without using ZipArchive?"
date: "2024-12-23"
id: "how-can-i-efficiently-extract-photos-from-a-zip-file-in-c-without-using-ziparchive"
---

Alright,  It's a situation I've encountered more than once, often when dealing with legacy systems or specific constraints that prevent the use of the `System.IO.Compression.ZipArchive` class. While that class provides a convenient and robust solution, sometimes we have to delve into the raw structure of zip files ourselves. So, yes, extracting images efficiently from a zip file in C# without relying on the convenience of `ZipArchive` is definitely achievable, though it does involve a lower level of interaction with the file format.

The key is understanding the basic structure of a zip archive. Fundamentally, it consists of a series of local file headers, each followed by its corresponding compressed data, and then a central directory at the end that provides a table of contents. This central directory is critical for efficiently locating the files within the archive without having to parse through the whole thing sequentially.

My first brush with this was back at 'Acme Solutions', where we had a dedicated imaging system that worked on embedded systems with severely constrained resources. Using `ZipArchive` simply was not an option. We had to work directly with the byte stream and the zip format. We weren't extracting just any files, we needed to identify image files specifically (jpg, png, etc) and process them as fast as possible, so a sequential parse through all file headers was a performance killer.

What we ended up implementing involved primarily these steps:

1. **Read the End of Central Directory Record (EOCDR):** This record is always located at the very end of a zip file. Knowing the length of this record, you can quickly locate its start position by reading backwards from the end of the file. This record contains vital information such as the offset of the central directory and the number of entries within it. This avoids a potentially huge sequential read.
2. **Parse the Central Directory:** Once you have the offset, you read the central directory, which lists each file within the archive. Each entry contains, among other things, the filename and the offset to the local file header.
3. **Locate Image Files:** Within this central directory, identify the entries corresponding to image files by examining their filenames and potentially file type magic numbers.
4. **Parse Local File Header:** For each identified image file, use the provided offset to reach the local file header. This header contains, among other things, the compressed size and the compression method.
5. **Decompress the Data:** Using the specified compression method (commonly deflation), decompress the data and obtain the original image file contents.
6. **Save or Process the Image.**

Here's an example, illustrating a simplified approach to identifying and extracting JPEG images:

```csharp
using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
public class ZipExtractor
{
    public static void ExtractJpegs(string zipFilePath, string outputDirectory)
    {
        const int eocdRecordSize = 22;
        const int minEocdSize = 16 + eocdRecordSize;

        using (var fileStream = new FileStream(zipFilePath, FileMode.Open, FileAccess.Read))
        {
            if(fileStream.Length < minEocdSize)
            {
                Console.WriteLine("Invalid Zip File");
                return;
            }

            // find EOCD record
            fileStream.Seek(-eocdRecordSize, SeekOrigin.End);
            byte[] eocdBytes = new byte[eocdRecordSize];
            fileStream.Read(eocdBytes, 0, eocdRecordSize);

            if(BitConverter.ToUInt32(eocdBytes, 0) != 0x06054b50)
            {
                Console.WriteLine("Invalid EOCD");
                return;
            }


            // get central directory offset
            long cdOffset = BitConverter.ToUInt32(eocdBytes, 16);
            fileStream.Seek(cdOffset, SeekOrigin.Begin);

            // Get number of entries.
             ushort numEntries = BitConverter.ToUInt16(eocdBytes, 8);

            //read entries
           for(int i = 0; i < numEntries; ++i)
           {
                byte[] centralDirEntry = new byte[46]; // minimal size of central dir entry
                fileStream.Read(centralDirEntry, 0, 46);
                if(BitConverter.ToUInt32(centralDirEntry,0) != 0x02014b50)
                {
                     Console.WriteLine("Invalid CD Entry");
                     return;
                }

                ushort fileNameLength = BitConverter.ToUInt16(centralDirEntry, 28);
                ushort extraFieldLength = BitConverter.ToUInt16(centralDirEntry, 30);
                ushort fileCommentLength = BitConverter.ToUInt16(centralDirEntry, 32);
                
                byte[] fileNameBytes = new byte[fileNameLength];
                fileStream.Read(fileNameBytes, 0, fileNameLength);
                 fileStream.Seek(extraFieldLength + fileCommentLength, SeekOrigin.Current);

                string fileName = System.Text.Encoding.UTF8.GetString(fileNameBytes);

                if (fileName.ToLower().EndsWith(".jpg"))
                {
                     uint localFileHeaderOffset = BitConverter.ToUInt32(centralDirEntry, 42);
                    
                    fileStream.Seek(localFileHeaderOffset, SeekOrigin.Begin);

                    byte[] localFileHeader = new byte[30];
                    fileStream.Read(localFileHeader, 0, 30);

                     if(BitConverter.ToUInt32(localFileHeader,0) != 0x04034b50)
                    {
                            Console.WriteLine("Invalid local header");
                            return;
                    }

                    ushort localFileHeaderFileNameLength = BitConverter.ToUInt16(localFileHeader, 26);
                    ushort localFileHeaderExtraFieldLength = BitConverter.ToUInt16(localFileHeader, 28);
                    
                    fileStream.Seek(localFileHeaderFileNameLength + localFileHeaderExtraFieldLength, SeekOrigin.Current);
                    uint compressedSize = BitConverter.ToUInt32(localFileHeader, 18);
                    ushort compressionMethod = BitConverter.ToUInt16(localFileHeader, 8);


                   byte[] compressedData = new byte[compressedSize];
                   fileStream.Read(compressedData, 0, (int)compressedSize);

                    using (MemoryStream memoryStream = new MemoryStream(compressedData))
                    {
                        if (compressionMethod == 8) // Deflate
                        {
                         using(DeflateStream deflateStream = new DeflateStream(memoryStream, CompressionMode.Decompress))
                         {
                            using (MemoryStream decompressedStream = new MemoryStream())
                            {
                                deflateStream.CopyTo(decompressedStream);

                                 string outputFilePath = Path.Combine(outputDirectory, Path.GetFileName(fileName));
                                 File.WriteAllBytes(outputFilePath, decompressedStream.ToArray());
                                Console.WriteLine($"Extracted: {fileName}");
                             }
                         }
                        } else {
                            //handle other compression methods
                        }
                    }

                }
           }
        }
    }
}
```

This snippet shows the foundational logic. Here’s a couple things to keep in mind:

*   **Error handling** is minimal here, you'd absolutely add more robust checks. For instance, make sure you have the right magic numbers at various points when parsing the zip file, such as the start of local file header, or that you're handling invalid compression methods gracefully.
*   This code currently only looks for JPEG files. You'd need to expand this condition (the `if (fileName.ToLower().EndsWith(".jpg"))`) and add more filename checks and/or file signature (magic bytes) inspection to identify other image formats you want to support.
*   **Efficiency:** While this bypasses `ZipArchive`, there's room for further optimization. For example, we could use buffered reading/writing and optimize memory allocations.

Here's another example showing how you might enhance the extraction process with a more comprehensive file extension check:

```csharp
using System;
using System.IO;
using System.IO.Compression;

public class ZipExtractorEnhanced
{
    public static void ExtractImages(string zipFilePath, string outputDirectory)
    {
         const int eocdRecordSize = 22;
        const int minEocdSize = 16 + eocdRecordSize;

        using (var fileStream = new FileStream(zipFilePath, FileMode.Open, FileAccess.Read))
        {
             if(fileStream.Length < minEocdSize)
            {
                Console.WriteLine("Invalid Zip File");
                return;
            }

            // find EOCD record
            fileStream.Seek(-eocdRecordSize, SeekOrigin.End);
            byte[] eocdBytes = new byte[eocdRecordSize];
            fileStream.Read(eocdBytes, 0, eocdRecordSize);

             if(BitConverter.ToUInt32(eocdBytes, 0) != 0x06054b50)
            {
                Console.WriteLine("Invalid EOCD");
                return;
            }

            // get central directory offset
            long cdOffset = BitConverter.ToUInt32(eocdBytes, 16);
             fileStream.Seek(cdOffset, SeekOrigin.Begin);


              // Get number of entries.
            ushort numEntries = BitConverter.ToUInt16(eocdBytes, 8);

            //read entries
           for(int i = 0; i < numEntries; ++i)
           {
                byte[] centralDirEntry = new byte[46]; // minimal size of central dir entry
                fileStream.Read(centralDirEntry, 0, 46);

                if(BitConverter.ToUInt32(centralDirEntry,0) != 0x02014b50)
                {
                    Console.WriteLine("Invalid CD Entry");
                     return;
                }

                 ushort fileNameLength = BitConverter.ToUInt16(centralDirEntry, 28);
                ushort extraFieldLength = BitConverter.ToUInt16(centralDirEntry, 30);
                ushort fileCommentLength = BitConverter.ToUInt16(centralDirEntry, 32);
                
                 byte[] fileNameBytes = new byte[fileNameLength];
                fileStream.Read(fileNameBytes, 0, fileNameLength);
                 fileStream.Seek(extraFieldLength + fileCommentLength, SeekOrigin.Current);


                string fileName = System.Text.Encoding.UTF8.GetString(fileNameBytes);
                string extension = Path.GetExtension(fileName).ToLower();

                  if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" || extension == ".gif")
                {
                     uint localFileHeaderOffset = BitConverter.ToUInt32(centralDirEntry, 42);

                     fileStream.Seek(localFileHeaderOffset, SeekOrigin.Begin);

                    byte[] localFileHeader = new byte[30];
                    fileStream.Read(localFileHeader, 0, 30);


                     if(BitConverter.ToUInt32(localFileHeader,0) != 0x04034b50)
                    {
                           Console.WriteLine("Invalid local header");
                            return;
                    }

                    ushort localFileHeaderFileNameLength = BitConverter.ToUInt16(localFileHeader, 26);
                    ushort localFileHeaderExtraFieldLength = BitConverter.ToUInt16(localFileHeader, 28);

                    fileStream.Seek(localFileHeaderFileNameLength + localFileHeaderExtraFieldLength, SeekOrigin.Current);

                     uint compressedSize = BitConverter.ToUInt32(localFileHeader, 18);
                     ushort compressionMethod = BitConverter.ToUInt16(localFileHeader, 8);


                   byte[] compressedData = new byte[compressedSize];
                   fileStream.Read(compressedData, 0, (int)compressedSize);

                    using (MemoryStream memoryStream = new MemoryStream(compressedData))
                    {
                        if (compressionMethod == 8) // Deflate
                        {
                           using(DeflateStream deflateStream = new DeflateStream(memoryStream, CompressionMode.Decompress))
                         {
                            using (MemoryStream decompressedStream = new MemoryStream())
                            {
                                deflateStream.CopyTo(decompressedStream);

                                 string outputFilePath = Path.Combine(outputDirectory, Path.GetFileName(fileName));
                                 File.WriteAllBytes(outputFilePath, decompressedStream.ToArray());
                                 Console.WriteLine($"Extracted: {fileName}");
                             }
                         }
                        } else {
                            //handle other compression methods
                        }
                    }
                }
           }

        }
    }
}
```

And finally, here's an example demonstrating how to enhance the code further by adding the handling of more compression methods, such as storing:

```csharp
using System;
using System.IO;
using System.IO.Compression;

public class ZipExtractorAdvanced
{
     public static void ExtractImagesAdvanced(string zipFilePath, string outputDirectory)
    {
         const int eocdRecordSize = 22;
        const int minEocdSize = 16 + eocdRecordSize;

        using (var fileStream = new FileStream(zipFilePath, FileMode.Open, FileAccess.Read))
        {
              if(fileStream.Length < minEocdSize)
            {
                Console.WriteLine("Invalid Zip File");
                return;
            }

            // find EOCD record
            fileStream.Seek(-eocdRecordSize, SeekOrigin.End);
            byte[] eocdBytes = new byte[eocdRecordSize];
            fileStream.Read(eocdBytes, 0, eocdRecordSize);

            if(BitConverter.ToUInt32(eocdBytes, 0) != 0x06054b50)
            {
                Console.WriteLine("Invalid EOCD");
                return;
            }

            // get central directory offset
            long cdOffset = BitConverter.ToUInt32(eocdBytes, 16);
            fileStream.Seek(cdOffset, SeekOrigin.Begin);


            // Get number of entries.
            ushort numEntries = BitConverter.ToUInt16(eocdBytes, 8);

            //read entries
           for(int i = 0; i < numEntries; ++i)
           {
                byte[] centralDirEntry = new byte[46]; // minimal size of central dir entry
                fileStream.Read(centralDirEntry, 0, 46);
                if(BitConverter.ToUInt32(centralDirEntry,0) != 0x02014b50)
                {
                    Console.WriteLine("Invalid CD Entry");
                     return;
                }

                ushort fileNameLength = BitConverter.ToUInt16(centralDirEntry, 28);
                 ushort extraFieldLength = BitConverter.ToUInt16(centralDirEntry, 30);
                 ushort fileCommentLength = BitConverter.ToUInt16(centralDirEntry, 32);
                
                 byte[] fileNameBytes = new byte[fileNameLength];
                fileStream.Read(fileNameBytes, 0, fileNameLength);
                fileStream.Seek(extraFieldLength + fileCommentLength, SeekOrigin.Current);

                string fileName = System.Text.Encoding.UTF8.GetString(fileNameBytes);
                string extension = Path.GetExtension(fileName).ToLower();

                 if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" || extension == ".gif")
                {
                    uint localFileHeaderOffset = BitConverter.ToUInt32(centralDirEntry, 42);

                     fileStream.Seek(localFileHeaderOffset, SeekOrigin.Begin);

                    byte[] localFileHeader = new byte[30];
                    fileStream.Read(localFileHeader, 0, 30);

                      if(BitConverter.ToUInt32(localFileHeader,0) != 0x04034b50)
                    {
                        Console.WriteLine("Invalid local header");
                        return;
                    }

                    ushort localFileHeaderFileNameLength = BitConverter.ToUInt16(localFileHeader, 26);
                     ushort localFileHeaderExtraFieldLength = BitConverter.ToUInt16(localFileHeader, 28);

                     fileStream.Seek(localFileHeaderFileNameLength + localFileHeaderExtraFieldLength, SeekOrigin.Current);


                    uint compressedSize = BitConverter.ToUInt32(localFileHeader, 18);
                    ushort compressionMethod = BitConverter.ToUInt16(localFileHeader, 8);

                    byte[] compressedData = new byte[compressedSize];
                    fileStream.Read(compressedData, 0, (int)compressedSize);

                    using (MemoryStream memoryStream = new MemoryStream(compressedData))
                    {
                         using (MemoryStream decompressedStream = new MemoryStream())
                         {
                           if (compressionMethod == 8) // Deflate
                             {
                                 using(DeflateStream deflateStream = new DeflateStream(memoryStream, CompressionMode.Decompress))
                                  {
                                     deflateStream.CopyTo(decompressedStream);
                                   }
                             } else if (compressionMethod == 0) {
                                //stored
                                memoryStream.CopyTo(decompressedStream);
                            }
                            else {
                                //handle other compression methods
                            }

                            string outputFilePath = Path.Combine(outputDirectory, Path.GetFileName(fileName));
                            File.WriteAllBytes(outputFilePath, decompressedStream.ToArray());
                             Console.WriteLine($"Extracted: {fileName}");
                        }
                    }
                }
           }

        }
    }
}
```

For deeper dives, I recommend examining the "PKWare Appnote" specification for the zip file format. Specifically, sections on the "End of Central Directory Record" and the "Central Directory File Header" will provide precise structural details. This document is considered the authority on the format. Also, "Data Compression: The Complete Reference" by David Salomon can prove invaluable for gaining a robust understanding of various compression algorithms, like the deflate algorithm, which is almost always encountered when extracting files from a zip.

Remember, working with file formats at this level requires precise attention to detail. Errors, if any, often manifest as corrupt or incorrectly decoded files. So, thorough testing is essential. While it might seem daunting at first, understanding the inner workings of a zip file can be both educational and exceptionally useful when you find yourself constrained by the available standard libraries.
