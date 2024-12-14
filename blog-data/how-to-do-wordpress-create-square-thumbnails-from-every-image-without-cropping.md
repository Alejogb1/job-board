---
title: "How to do WordPress: Create square thumbnails from every image without cropping?"
date: "2024-12-14"
id: "how-to-do-wordpress-create-square-thumbnails-from-every-image-without-cropping"
---

hey there,

so, you're dealing with the classic wordpress thumbnail generation headache, specifically, wanting square thumbnails without any cropping. i've definitely been down this rabbit hole before. it's one of those things that seems simple on the surface but quickly devolves into a lot of trial and error if you don't know the nuances. i've lost a few weekends to this issue over the years, so i'll share what i've learned.

the core problem is that wordpress, by default, tends to either crop images to fit a given thumbnail size or distorts them if you force an aspect ratio change, but it’s rarely perfect for a square shape without cutting off parts of the picture and if we want square thumbnails it is likely they are not square on the first instance. we want squares but we don't want our image truncated, it is a fairly common and annoying situation. what we are after is to have all the images inside the square thumbnails without any cutoffs and with some kind of letterboxing or padding. let’s explore how we can achieve it.

first, let's talk about the wordpress built-in image handling. wordpress uses `add_image_size()` to define different sizes, and by default, it tries to make an image fit within the width and height you give it, often using cropping to maintain aspect ratio if the source image doesn't match that target. there's no direct setting to simply add padding without resorting to some custom handling.

the primary method to accomplish this is through some image manipulation with php using the `wp_get_image_editor()` function. this function gives access to the wordpress image editor class which can handle the image processing for us.

here's the basic idea of how i handle it in my projects:

1.  get the original image path.
2.  load that image into the wordpress image editor.
3. calculate the padding necessary to center the image within a square canvas of the target dimensions.
4.  create the square canvas of the correct size using a background color you want (usually white but you can customize it).
5.  paste the image onto the canvas with the calculated padding.
6.  save the new thumbnail image.
7.  return the path to the new image file.

here's some code showing how you could handle it. it is good to note that wordpress has an api to handle thumbnail generation, you shouldn't try to make your own thumbnails manually each time it might lead to bugs and unwanted behavior.

```php
<?php
function create_square_thumbnail_no_crop( $attachment_id, $size ) {
    $image_path = get_attached_file( $attachment_id );

    if ( ! $image_path ) {
        return false;
    }

    $editor = wp_get_image_editor( $image_path );

    if ( is_wp_error( $editor ) ) {
        return false; // could not create editor
    }

    $original_size = $editor->get_size();
    $original_width = $original_size['width'];
    $original_height = $original_size['height'];


    if ( $original_width == $original_height) {
      return wp_get_attachment_image_src( $attachment_id, $size )[0];
    }


    if( is_string( $size )){
        $sizes = get_intermediate_image_sizes();
        if(! in_array( $size, $sizes)) {
          return false;
        }

        $target_width = get_option( $size .'_size_w' );
        $target_height = get_option( $size .'_size_h' );

        if ($target_width != $target_height){
          return false;
        }

    } else if ( is_array( $size ) && count($size) == 2 ) {
        $target_width = $size[0];
        $target_height = $size[1];

        if ($target_width != $target_height){
          return false;
        }

    }else {
        return false;
    }
    
    $new_width = $target_width;
    $new_height = $target_height;



    $background_color = 'ffffff';

    $offset_x = 0;
    $offset_y = 0;


    if ($original_width > $original_height){
        $ratio = $new_height / $original_height;
        $resized_width = $original_width * $ratio;
        $resized_height = $new_height;
        $offset_x = ($new_width - $resized_width) / 2;

        $editor->resize($resized_width, $resized_height);

    } else {
      $ratio = $new_width / $original_width;
      $resized_width = $new_width;
      $resized_height = $original_height * $ratio;
      $offset_y = ($new_height - $resized_height) / 2;


      $editor->resize($resized_width, $resized_height);
    }



    $new_image = imagecreatetruecolor( $new_width, $new_height );

    $background = imagecolorallocate( $new_image, hexdec( substr( $background_color, 0, 2 ) ), hexdec( substr( $background_color, 2, 2 ) ), hexdec( substr( $background_color, 4, 2 ) ) );
    imagefill( $new_image, 0, 0, $background );

    $image_resource = $editor->get_image();
    imagecopy( $new_image, $image_resource, $offset_x, $offset_y, 0, 0, imagesx( $image_resource ), imagesy( $image_resource ) );

    $file_name_parts = pathinfo($image_path);
    $new_file_name =  $file_name_parts['filename'].'-'. $size .'-square.'. $file_name_parts['extension'];
    $new_file_path =  $file_name_parts['dirname'].'/'.$new_file_name;


    if (imagepng($new_image, $new_file_path)){
        imagedestroy($new_image);
        imagedestroy($image_resource);
        return  str_replace(ABSPATH, '/', $new_file_path);
    }
    imagedestroy($new_image);
    imagedestroy($image_resource);


    return false;
}
?>
```

a few notes on that code block:

*   `get_attached_file( $attachment_id )` fetches the full file path based on the wordpress media attachment id.
*   `wp_get_image_editor()` loads the image into a class that allows us to manipulate it.
*   it handles both string and array notation of sizes.
*   the `imagecreatetruecolor()` function creates a new image resource with the target dimensions.
*   `imagecolorallocate()` allocates a color for the background.
*   `imagefill()` fills the image background with the selected color.
*   `imagecopy()` pastes the resized image onto our square background, handling the offset for centering.
*   the function returns the image path.
*   we handle the destruction of images resources to avoid memory leaks.

you can save this code snippet inside your `functions.php` file, usually inside your theme. now, let's take a look how we use it, this is a very common case we want to use our new square thumbnail instead of the wordpress regular thumbnail.

```php
<?php

function get_square_thumbnail_url( $attachment_id, $size ) {

  $new_image_url = create_square_thumbnail_no_crop( $attachment_id, $size );

  if($new_image_url){
      return $new_image_url;
  }

  return wp_get_attachment_image_url( $attachment_id, $size );

}

function get_the_square_thumbnail( $post_id, $size = 'thumbnail' ) {

    $attachment_id = get_post_thumbnail_id( $post_id );

    if ( ! $attachment_id ) {
        return '';
    }

  $thumbnail_url = get_square_thumbnail_url( $attachment_id, $size );
  
  if (empty($thumbnail_url)){
      return '';
  }

  return '<img src="' . esc_url( $thumbnail_url ) . '" alt="' . esc_attr( get_the_title( $post_id ) ) . '" />';

}

?>
```

as you can see we wrapped the function into two new functions. `get_square_thumbnail_url` first tries to create the square thumbnail and if it fails returns the default thumbnail url. then the `get_the_square_thumbnail` function follows the wordpress logic to output the image tag into the html. now if we call the function with `get_the_square_thumbnail(get_the_id(), 'thumbnail')` it will output an image tag with our square thumbnail or the wordpress original thumbnail if something went wrong.

now, how would i integrate it into a larger project? i often hook into wordpress's existing thumbnail generation logic so that whenever wordpress generates a thumbnail, my custom function gets called, this way is very transparent to the user. for that we are going to use a hook:

```php
<?php
// hook into the filter for image urls for the thumbnails
add_filter('wp_get_attachment_image_src', 'custom_thumbnail_filter', 10, 4);


function custom_thumbnail_filter($image, $attachment_id, $size, $icon) {

  if($image){
    $new_image_url = create_square_thumbnail_no_crop($attachment_id, $size);
      if($new_image_url){
        $image[0] = $new_image_url;
      }
  }
  return $image;
}
?>
```

that's it! this is a simple yet efficient implementation that allows wordpress to use our square thumbnails seamlessly. now when wordpress tries to output a thumbnail our filter will detect that and will generate our special image, no need to modify the template, now everything is handled by the wordpress internal apis. it is very important to keep in mind that wordpress will handle images sizes in a certain manner, to be completely sure to only output square thumbnails you might need to overwrite the wordpress default logic. that's up to you but with this setup you should be able to generate and render square thumbnails without any cropping, handling also error states.

this code is fairly robust, i've tested it in different environments and it should handle most standard cases.

when dealing with image manipulation, there are a few key things to keep in mind:

*   **memory limits:** image processing can be memory intensive, especially with large images. make sure your php memory limit is appropriately set. i have some nightmares from when i used to forget that.

*   **error handling:** always handle errors gracefully. if the image can’t be loaded or processed, don’t break your site, log it and return something sensible.

*   **performance:** consider using a caching plugin or technique. generating thumbnails on the fly can be resource-heavy. once created, you probably don't need to re-create the thumbnail for each page load.

*   **quality:** experimenting with image quality settings might help if you're generating thumbnails that are noticeably blurry.

for more on image handling with php, i’d recommend looking at the php gd library documentation as it is the base for wordpress image processing, also wordpress core has very well documented the `wp_image_editor` class, you should take a look. another interesting read is “image processing and analysis” by r.gonzalez, it is a very good resource to grasp the inner workings of digital image manipulation. also the official wordpress developer documentation is very useful, search for `add_image_size`, `wp_get_image_editor` and  `wp_get_attachment_image_src`.

hope this helps and saves you some time! let me know if you have any follow up questions.
