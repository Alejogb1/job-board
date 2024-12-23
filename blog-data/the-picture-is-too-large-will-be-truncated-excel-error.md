---
title: "the picture is too large will be truncated excel error?"
date: "2024-12-13"
id: "the-picture-is-too-large-will-be-truncated-excel-error"
---

 so you're hitting that classic "excel picture too big it's getting chopped off" problem Been there done that countless times It's a pain I know

I remember this one time back when I was still figuring things out on my old Pentium machine I was trying to build this massive dashboard with all sorts of fancy charts and of course pictures And I'd just copy paste images in because well seemed easier right Then I tried to send it over to my boss who back then still used a tiny 13 inch screen And man did that excel file just look like hot garbage all the images were cut off or warped it was a total mess Lesson learned the hard way about excel image limitations

The core issue we are dealing with here is excel's somewhat finicky way of handling image sizes Excel doesn't resize images gracefully by default and when an image's resolution or the space it's suppose to occupy gets bigger than the cells it's occupying well you get that truncation effect It just chops off whatever doesn't fit The reason for this is excel handles images as objects attached to the worksheet or in cells and it does not handle as a simple resize algorithm and it cannot resize the pictures automatically based on the column or row cell size I'm sure other spreadsheets do that better but here we are

To get this straight and prevent future headache there are several things you can do

First the most straightforward thing is just resizing the image before you even paste it into excel This means using an image editor like paint irfanview or if you're fancy photoshop or gimp I use mostly IrfanView and sometimes the online version of gimp so here you resize to fit the size of the cells so it does not overlap The cell size is important to take into account

If for example the cell size is 100 by 100 pixels then you make sure the picture has same pixel proportions or less Another basic thing is compressing the picture itself some picture formats like PNG can be much larger then JPEG and even if the resolution is the same the file size can be much bigger so using JPEG could do the trick

But we are not living in caveman times right so let's talk about some basic VBA code for excel

Here is some code that resizes the selected image to fit within its cell

```vba
Sub ResizeImageToFitCell()
    Dim shp As Shape
    If TypeName(Selection) = "Picture" Then
        Set shp = Selection
        With shp
            .LockAspectRatio = msoFalse
            .Top = shp.Parent.Top
            .Left = shp.Parent.Left
            .Width = shp.Parent.Width
            .Height = shp.Parent.Height
        End With
    Else
        MsgBox "Please select an image first"
    End If
End Sub
```

 lets break down what this code does we declare a shape variable shp  then we check if the selection is a picture or not using TypeName If it is not a picture it sends a message Box and exits If it is an image then we set it to shp variable then we use with to handle it's properties first we turn off the aspect ratio lock so we can resize it freely and then the rest of the code matches image top and left to the cell and the height and width so it matches with the cell if you are doing this in bulk you will notice that the performance can be terrible because it recalculates the sheet every single time so there is a solution for that

Here is another piece of code to avoid that which goes faster when you need to resize many pictures at once

```vba
Sub ResizeMultipleImages()
    Dim shp As Shape
    Dim ws As Worksheet
    Dim i As Long
    Set ws = ActiveSheet
    Application.ScreenUpdating = False
    For Each shp In ws.Shapes
        If shp.Type = msoPicture Then
            With shp
                .LockAspectRatio = msoFalse
                .Top = shp.Parent.Top
                .Left = shp.Parent.Left
                .Width = shp.Parent.Width
                .Height = shp.Parent.Height
            End With
        End If
    Next shp
    Application.ScreenUpdating = True
End Sub
```

This code is very similar to the previous one but it takes all the shapes in the sheet and checks each one if it's a picture then resizes it to fit the cell and the key difference is Application ScreenUpdating which turns off screen updates which makes it run much faster also note that both codes also works with different image formats like JPEG PNG BMP and etc

And now you might be thinking "hey this is great but I need the picture to stay the same proportions"  I get you another piece of code for that it's very similar to the last one with minor changes so the picture does not look stretched

```vba
Sub ResizeMultipleImagesWithAspectRatio()
   Dim shp As Shape
    Dim ws As Worksheet
    Dim i As Long
    Set ws = ActiveSheet
     Application.ScreenUpdating = False
    For Each shp In ws.Shapes
        If shp.Type = msoPicture Then
           With shp
             .LockAspectRatio = msoTrue
               If .Width > .Parent.Width Or .Height > .Parent.Height Then
                Dim cellWidth As Double
                Dim cellHeight As Double
                 cellWidth = .Parent.Width
                cellHeight = .Parent.Height
                If .Width / .Height > cellWidth / cellHeight Then
                    .Width = cellWidth
                    .Top = .Parent.Top + (cellHeight - .Height) / 2
                 Else
                    .Height = cellHeight
                    .Left = .Parent.Left + (cellWidth - .Width) / 2
                End If
            End If
               .Top = shp.Parent.Top
                .Left = shp.Parent.Left
          End With
         End If
    Next shp
    Application.ScreenUpdating = True
End Sub

```

This one is a bit more complicated what we do is first lock the aspect ratio then compare image size to cell size to see if we need to resize it then based on the proportions and sizes we scale width and height to fit in cell without stretching and we maintain the aspect ratio Then it re aligns the picture within the cell

Now the million-dollar question why even bother with VBA code instead of just resizing before pasting Well sometimes you cannot for example when the person using the spreadsheet copies and pastes images or when the images are generated from a different software I had to dealt with these scenarios countless times believe me Also some people cannot really be trained to do the correct way so you automate the process for them

 so that's pretty much all you need to know I guess a good resource about VBA is the book "VBA for dummies" it is a bit old but the basics are always the same there are also many resources online and excel own documentation for vba If you want to dig deeper into image processing there is a book called "Digital Image Processing" by Rafael C Gonzalez and Richard E Woods is quite heavy but has a ton of info about image manipulation

Oh by the way did you hear about the programmer who got stuck in the shower he was trying to read the shampoo instructions but they kept looping back to "lather rinse repeat"? Yeah I know bad joke but still anyway if you get stuck feel free to ask more questions
