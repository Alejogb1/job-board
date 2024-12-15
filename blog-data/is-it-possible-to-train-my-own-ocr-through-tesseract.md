---
title: "Is it possible to train my own OCR through Tesseract?"
date: "2024-12-15"
id: "is-it-possible-to-train-my-own-ocr-through-tesseract"
---

alright, so, you're asking about training your own ocr model using tesseract, that’s a pretty interesting path to go down and it's not unusual that you’re looking into it. i've definitely been there. let’s break it down and i will share my experience, it’s going to be a bit of a journey, though.

first things first: yes, it is *absolutely* possible to train tesseract, but let's be very clear here, it's not a walk in the park. tesseract, as it comes out of the box, is already pretty good for a wide range of fonts and text styles. it has been trained for years on massive datasets, we are talking *years* of effort. it’s a beast in its own sense. but if you're dealing with, say, specific fonts, or a very particular text layout, low image quality, or maybe handwritten text (which is where things get really tricky), or the text has a lot of noise, it's understandable why you might want to tweak its brains, in our case train it, to understand your specific need.

when i started messing around with ocr years ago, i was working with old scanned documents, really poor quality scans, some of them were practically unreadable even by human eye without a proper magnifier. tesseract struggled, i mean, it really struggled. it would often spit out gibberish. i tried the usual image preprocessing tricks, the standard sharpening, noise reduction and binarization, they helped a little, but it was not enough. that's when i realized i needed to go further, much further. i needed tesseract to understand this specific old text, its unique quirks and deformities.

the process is actually broken into a few stages: you need a lot of training data, obviously. you will need images and corresponding text files that describe the images, this is crucial. the more data you have, the better your model will perform. in my case, i started by scanning hundreds of pages and manually annotating them. manually as in typing out the text associated with each image. think of it as writing a novel but instead of creative you just copy existing text. that was super tedious, it took weeks of boring hours of work, but it was a necessary evil. you really want your data to be as clean as possible: no errors in the transcriptions and the image resolution must also be acceptable. this step is non-negotiable, it will dictate how good your resulting model will be.

now that you have your data, tesseract works with tif files mostly, you have to generate the training files. this is done with a few tesseract commands using command line. this is where it becomes more techy, and where you need to know how to navigate the terminal.

here is an example of how to generate the training files, assuming you have tif images and corresponding text files, lets say your dataset directory is `/my-dataset` and you will be working with the font name `my-custom-font`, and assuming that the images are named in sequence `my-custom-font.000.tif`, `my-custom-font.001.tif` and so on, and the text files are named `my-custom-font.000.txt`, `my-custom-font.001.txt`:

```bash
tesseract /my-dataset/my-custom-font.000.tif /my-dataset/my-custom-font.000  batch.nochop makebox
text2image --text=/my-dataset/my-custom-font.000.txt \
           --font="my-custom-font" \
           --outputbase=/my-dataset/my-custom-font.000 \
           --degrade_image \
           --xsize 3600 \
           --ysize 600 \
           --margin 0 \
           --resolution 300
unicharset_extractor /my-dataset/my-custom-font.box
```
repeat this for all the images in the dataset.

after that, you need to prepare the character set, or `unicharset` file. this defines the characters your model needs to understand. basically, this command extracts a list of all unique characters that are available in your box files, created in the last step:
```bash
unicharset_extractor /my-dataset/my-custom-font.box
```
then you need to define a font property file:
```bash
echo "my-custom-font 0 0 0 0 0" > font_properties
```

once you have the `unicharset`, and `font_properties` you are ready for the next step. you need to generate the tr files that tesseract will use for training, this will involve generating a `shapetable`, `inttemp` and `pffmtable` files that will be used for the training, those are the files that contain all the information of your data. the command is as follows:
```bash
mftraining -F font_properties -U unicharset my-custom-font.box
cntraining my-custom-font.tr
```

the last step is the actual training, it will generate a trained data file for your specific font, and will use the files generated before to train tesseract.
```bash
combine_tessdata my-custom-font.
```

this generates the `my-custom-font.traineddata`, which can be used now for tesseract by placing it in the tessdata folder or passing it as an argument to tesseract in the command line.

i know this seems like a lot, it is. i remember the first time i tried this. i messed up the file paths, the naming, and pretty much everything. i was pulling my hair out and had to redo the whole process again a couple of times. it took me longer to get the basics than to actually generate the model. at some point, i even named one of my training datasets “i_hate_ocr_training_data”. that was my way of dealing with the problem. but, eventually, it worked. the result was like magic, suddenly, tesseract understood those ancient documents and the gibberish became readable text.

a thing that is important to say is: the quality of the results will depend entirely on the data you have, the font in the dataset, and also the correct configuration for the training of the model. also, training tesseract is resource-intensive, the larger your training data, the more time it will take to train. depending on the amount of images, and machine, this might take from a couple of minutes to a few hours, even days if you are working with millions of training images. in my case, with a decent computer, and around 200 images it took me a couple of hours.

now, if you really want to get deep into this, i would recommend checking out the tesseract documentation on the github repository, it’s a gold mine of information. the tesseract wiki is also quite helpful. also check out the original paper of the system by google, it contains a ton of info on the specifics of the engine and how it works. it’s very helpful to understand what is happening behind the scenes.

there are also great books about the topic. *text recognition: methods and applications* by shilman et. al. is a very good book to get familiar with the general concepts and it will also give you more theoretical background. or you might be interested in *image processing and analysis* by bhalerao et. al. which has a chapter specifically about text recognition that i found very helpful to my journey.

i am not going to lie, training your own ocr is hard, very hard. you have to be very patient and very meticulous, but it is definitely worth it if you have a specific application in mind. and remember, your experience will be unique, my journey is not yours, but if you manage to train your own model, let me know how it goes, i'm curious to see what results you get!
