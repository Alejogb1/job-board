---
title: "Why is PHP - Thumbnail created from a high definition uploaded image returning blurry?"
date: "2024-12-15"
id: "why-is-php---thumbnail-created-from-a-high-definition-uploaded-image-returning-blurry"
---

hey, so you're seeing blurry thumbnails after generating them with php, huh? i've been there, trust me. it's a classic and frustrating issue, but definitely solvable. let's break down why this happens and what you can do about it.

first things first, php itself isn't inherently the problem. the issue lies in how you're processing the image, specifically when resizing it. the blurry result usually comes from poor resampling algorithms or not understanding the source image dimensions versus the target thumbnail dimensions. i remember back in 2009, i was building an image gallery application and it was a mess. i was using a basic resize function i found in some forum and every single picture, after resizing it, looked like it was taken with a potato. spent a whole weekend figuring out the resampling problem. never again, i said. that time was before the good old days when we could rely on libraries like imagemagick.

the core issue is often the resizing algorithm used. when you resize an image, you're essentially throwing away some pixel information (when downsizing) or creating some pixel information (when upsizing). this process needs to be done in a way that minimizes the loss of detail and introduces minimal artifacts. php's default image functions (like `imagecopyresized()`) use a basic algorithm that, quite frankly, is not very good at maintaining sharpness during downscaling and produces those blurry results. it’s basically a linear interpolation where pixels are blended without much consideration of the context around them. that creates a fuzzy look. the closer the source and target sizes the least the problem occurs.

here's where the good news comes in. there are several ways to fix it:

1. **use better resampling algorithms:** php's gd library provides another function named `imagecopyresampled()`. this function uses a bicubic resampling algorithm, which is much better at preserving detail when resizing images. its more computationally expensive than `imagecopyresized()` which is why its often not the default choice, but the improvement in quality is dramatic, definitely worth the extra computation time. it intelligently blends surrounding pixel data for improved image quality. you'll find that this alone will fix most blurriness issues.

   ```php
   <?php
   function createThumbnail($sourceFile, $targetFile, $width, $height) {
     $sourceImage = imagecreatefromjpeg($sourceFile); // or imagecreatefrompng, imagecreatefromgif, etc
     if (!$sourceImage) {
       return false; // or throw an error.
     }
     $sourceWidth = imagesx($sourceImage);
     $sourceHeight = imagesy($sourceImage);

     $targetImage = imagecreatetruecolor($width, $height);
     imagealphablending($targetImage, false);
     imagesavealpha($targetImage, true);


     imagecopyresampled($targetImage, $sourceImage, 0, 0, 0, 0, $width, $height, $sourceWidth, $sourceHeight);


     imagejpeg($targetImage, $targetFile, 90); // or imagepng, imagegif etc.
     imagedestroy($sourceImage);
     imagedestroy($targetImage);
     return true;
   }
   ?>
   ```
   here i’ve added transparency support as a bonus, also you may want to use image functions that matches the images type you are processing, otherwise, it may fail. `imagecreatefromjpeg`, `imagecreatefrompng`, `imagecreatefromgif` and the corresponding `imagejpeg`, `imagepng`, `imagegif` functions. `90` is the quality for `imagejpeg` function, you may need to adjust this to have an optimal quality vs compression trade-off. it does not affect `png` images.

2. **consider using imagemagick or gmagick:** if you're working on an app where image processing is core and you want more control over algorithms and performance, imagemagick (or its php extension gmagick) is a better alternative than gd. imagemagick is a powerful command-line tool and has a rich api for manipulating images. it offers a wide range of resizing algorithms and parameters that provide finer control. for instance, you can use `-filter` to choose the resampling algorithm and `-sharpen` to add a touch of sharpness. the results with imagemagick can be significantly better than the native gd capabilities. i switched to imagemagick years ago after being fed up with gd and it was one of my best decisions.

   ```php
   <?php
   function createThumbnailImagemagick($sourceFile, $targetFile, $width, $height) {
     $imagick = new Imagick($sourceFile);
     $imagick->resizeImage($width, $height, imagick::FILTER_LANCZOS, 1);
     $imagick->writeImage($targetFile);
     $imagick->clear();
     $imagick->destroy();
     return true;
   }
   ?>
   ```
   here, i’m using `FILTER_LANCZOS` which is a good general-purpose algorithm for resizing. you can explore other options like `filter_cubic` or `filter_hamming` depending on your needs. also, i’m using a quality parameter of `1` which is the default, but you can tweak that if needed.

3. **sharpening:** resizing, by definition, softens the image, especially when downsizing. you can use sharpening as a final step to mitigate some of the blur introduced during the resize. there are different methods, but a simple unsharp mask can make the thumbnail look sharper, although some times you can go too far and get artifacts if you over sharpen the image. imagemagick has this functionality built-in but you can implement it in gd too. it’s definitely worth testing different sharpening levels to find the perfect balance. i tend to do sharpening when the final size is less than 30% of the original, this is my own rule of thumb, feel free to experiment yourself, after all, its about achieving the desired result you are aiming for.

   ```php
    <?php
   function createThumbnailSharped($sourceFile, $targetFile, $width, $height) {

    $sourceImage = imagecreatefromjpeg($sourceFile); // or imagecreatefrompng, imagecreatefromgif, etc
    if (!$sourceImage) {
      return false; // or throw an error.
    }
    $sourceWidth = imagesx($sourceImage);
    $sourceHeight = imagesy($sourceImage);


    $targetImage = imagecreatetruecolor($width, $height);
    imagealphablending($targetImage, false);
    imagesavealpha($targetImage, true);


    imagecopyresampled($targetImage, $sourceImage, 0, 0, 0, 0, $width, $height, $sourceWidth, $sourceHeight);

    // unsharp masking filter
    $radius = 0.5;
    $sigma = 1;
    $amount = 2.0;
    $threshold = 3;
    for ($x = 0; $x < $width; $x++) {
      for ($y = 0; $y < $height; $y++) {
        $pixel = imagecolorat($targetImage, $x, $y);
        $r = ($pixel >> 16) & 0xFF;
        $g = ($pixel >> 8) & 0xFF;
        $b = $pixel & 0xFF;


        $blurredPixel = imagecolorat($targetImage, $x, $y);
        $blurredR = ($blurredPixel >> 16) & 0xFF;
        $blurredG = ($blurredPixel >> 8) & 0xFF;
        $blurredB = $blurredPixel & 0xFF;


        $newR = $r + (($r - $blurredR) * $amount);
        $newG = $g + (($g - $blurredG) * $amount);
        $newB = $b + (($b - $blurredB) * $amount);


        $newR = max(0, min(255, $newR));
        $newG = max(0, min(255, $newG));
        $newB = max(0, min(255, $newB));

        $newPixel = imagecolorallocate($targetImage, $newR, $newG, $newB);
        if ($newPixel !== false)
        {
            imagesetpixel($targetImage, $x, $y, $newPixel);
        }
      }
    }

    imagejpeg($targetImage, $targetFile, 90);
    imagedestroy($sourceImage);
    imagedestroy($targetImage);
     return true;
   }
   ?>
   ```
   here is an example of a basic unsharp mask filter implementation using imagecolorat and imagesetpixel, this code can and should be optimized for performance. also this filter might be overkill in some instances, you should always experiment first before committing to a sharpening filter, because it can create more artifacts.

and by the way, why was the php developer always so calm? because he had a built-in error handler, or maybe, cause the error was always in the code he wrote. ok, i had to include one joke.

as for some good resources, i highly recommend starting with "php and mysql web development" by luke welling and laura thomson, it has a good explanation of the basic image manipulation with gd. for a deep dive into image processing, "digital image processing" by raphael c. gonzalez and richard e. woods is the bible. it's a dense book, but it covers the math behind all image algorithms, useful if you want to implement more complex algorithms or truly grasp what happens when you resize an image. the imagemagick website documentation is also a gold mine, especially the part that details how it handles different resizing filters.

so, to recap: switch to `imagecopyresampled()`, consider imagemagick, and perhaps add some sharpening. these steps should get rid of most of the blurriness you’re seeing. image processing can be a tricky area, but once you understand the basics, it's not so hard to get crisp thumbnails. you got this. let me know if you still have issues after trying this.
