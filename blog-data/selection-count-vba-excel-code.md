---
title: "selection count vba excel code?"
date: "2024-12-13"
id: "selection-count-vba-excel-code"
---

Okay so you're asking about selection counts in VBA Excel right been there done that a few times trust me it sounds simple on paper "just count selected cells right" but it can get weird fast Let me tell you about my past adventures with this kind of thing I once worked on a project that involved a heavily user interacted spreadsheet think stock trading platform with tons of custom macros and VBA code The users were constantly making selections all over the place copying pasting deleting and we needed to track precisely how many cells they were working with and which ones They were complaining about some calculations being wrong and my team and I suspected selection-related bugs were in play So i had to dive deep into the VBA side of things and get this working reliably Lets get to it

First off the most basic thing is to get the count of cells in the current selection its a one-liner really

```vba
Sub GetSelectionCount()
    Dim cellCount As Long
    cellCount = Selection.Count
    MsgBox "Number of selected cells: " & cellCount
End Sub
```

See straightforward enough the `Selection` object in VBA is the current selected area in excel and its `.Count` property gives you the total number of cells It works if you select a single cell a range of cells or even multiple non-contiguous ranges

But here is where the fun begins that number includes _all_ the cells even those in non-rectangular ranges What if you need to check for specific things in the selected area or if there are specific things you need to do with the selected area this one count will not give it to you Also what about things that are not cells are selected in Excel like shapes or charts or the user selects entire columns rows and you need only the cells in a selected area within the spreadsheet this is another issue the previous snippet does not address You can check for object types but still lets say that a user selects a range of cells with different types or formats VBA needs to iterate though them and this can become tricky

Here is one case where we need to get the number of selected rows or columns it can get complex and you need to consider all the edge cases

```vba
Sub GetSelectedRowsAndColumns()
    Dim selectedRange As Range
    Dim rowCount As Long
    Dim columnCount As Long

    On Error Resume Next
    Set selectedRange = Selection
    On Error GoTo 0

    If selectedRange Is Nothing Then
        MsgBox "No selection detected"
        Exit Sub
    End If

    'Check if an entire row or column is selected
    If selectedRange.Rows.Count = Rows.Count Then
        rowCount = Rows.Count
        MsgBox "Entire rows selected"
    Else
      rowCount = selectedRange.Rows.Count
      MsgBox "Selected rows " & rowCount
    End If

    If selectedRange.Columns.Count = Columns.Count Then
       columnCount = Columns.Count
        MsgBox "Entire columns selected"
    Else
        columnCount = selectedRange.Columns.Count
      MsgBox "Selected columns " & columnCount
    End If
End Sub
```

In this snippet we are doing a few things first error handling and checking if a selection exists then we are making sure to check if an entire row or column was selected The check `selectedRange.Rows.Count = Rows.Count` and `selectedRange.Columns.Count = Columns.Count` do just that we compare the selected range rows and column count to the maximum available rows and columns in the current worksheet if they are the same that means that the user selected the entire column or row

But what if we wanted to do something else like iterating through the selected cells to check a specific property or count blank cells in the selected range this is where the real fun begins the selected range can be non-contiguous or the user can select a mix of cells and ranges Lets take the blank cells check as an example

```vba
Sub CountBlankCells()
    Dim selectedRange As Range
    Dim cell As Range
    Dim blankCellCount As Long

    On Error Resume Next
    Set selectedRange = Selection
    On Error GoTo 0

    If selectedRange Is Nothing Then
        MsgBox "No selection detected"
        Exit Sub
    End If

    blankCellCount = 0
    For Each cell In selectedRange
        If IsEmpty(cell.Value) Then
            blankCellCount = blankCellCount + 1
        End If
    Next cell

    MsgBox "Number of blank cells in selection: " & blankCellCount
End Sub
```

Here we iterate using a `For Each` loop which is designed to work with ranges effectively iterating over each cell in the selection individually and each cell value is checked using the `IsEmpty()` function

Now here is what you need to keep in mind you have to avoid using the `.Cells` function to iterate through a selection that is not contiguous using `For Each` is much safer and efficient especially in older excel versions where the `selection.cells` function can be buggy also avoid doing any writing or editing operations directly while looping though the selected range this will make excel unresponsive and can cause issues and errors with the `Selection` object

You know I remember this one time i was so confused by a bug I spent three hours debugging because a user kept saying the counts were off but I had a similar code and I had a hunch about the nature of this specific bug and I did the next level debugging move I wrote every variable value to a log file and boom the log file gave me the hint I was missing at the very end of a complex nested loop i had another condition that was throwing off all the counts and it was just in the wrong place in the code the debugging process can be tricky sometimes you know it is the price for being a code whisperer right hehe

For further reading on how excel and the VBA engine work you should get your hands on some good resources I always recommend _Microsoft Excel VBA Programming for Dummies_ yes the "for dummies" part is for everybody not just beginners the book provides a good overview of VBA and its interaction with Excel and all the different objects and range selection handling For advanced topics on excel objects and their properties _Excel 2019 Power Programming with VBA_ by Michael Alexander and Dick Kusleika is an incredible resource and can become your VBA Bible on excel automation It goes deep into object model of excel and more complex uses of VBA to automate excel tasks

Anyways that should cover most common cases with selection counts in VBA Excel remember to handle the edge cases check if something is selected before doing anything iterate carefully and dont assume contiguous selections and if something goes wrong use a log file to dump all your variable values to find tricky bugs hope this helped you avoid spending 3 hours debugging a single issue like I did you have to be more wise than me if you want to be a good developer hehe
