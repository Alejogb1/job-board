---
title: "21h interrupt dos?"
date: "2024-12-13"
id: "21h-interrupt-dos"
---

 so 21h interrupt DOS right seen this one a few times back in the day lets dive in I’ve been wrestling with this stuff since probably before some of you were even born yeah seriously I'm talking about real DOS era stuff not just some retro emulator game it was a beast back then no fancy GUI no stack overflow we had to figure things out ourselves with debuggers and cryptic error messages those were the good days well good in a really painful kind of way

So 21h interrupt it's the heart of DOS the whole system pretty much relies on it it’s like the central switchboard for all kinds of stuff input output file operations memory management you name it when you call the 21h interrupt you’re basically asking DOS to do something for you its the API before APIs even had names you load up the AH register with a function number and that tells DOS what you want it to do sometimes you’ll need to set other registers with data or addresses depending on the specific function

Now this is where it gets interesting and frustrating back in the day we didn’t have the luxury of online documentation we had this massive bible of a book called the "Undocumented DOS" its like the bible but for old school nerds and sometimes you'd need to get out a debugger to trace what exactly was happening when you hit an issue you'd spend hours looking at hex code trying to figure out why your little program would hang or crash

I remember one specific project I was trying to make a simple file manager back when windows was just a gleam in someone’s eye and I was trying to rename a file it just wasn't working and I thought I was losing it i kept looking at my code for hours thinking it was me it turned out that there was this tiny detail in how the function worked with directory entries i had missed it it was about the flags used for directory entries not being in the right format it wasn't in the official docs it was hidden deep in some random forum post I had to dig up in the pre-google era. That's the kind of pain and joy I'm talking about with DOS stuff

The 21h interrupt it has a lot of functions like seriously a ton AH equals to 01 for input from console AH equals 02 output to console AH equals 09 displaying a string AH equals 3ch creating a file AH equals 3eh closing a file and so on this is just a small sample of them the AH register dictates which function is being invoked but there is no way to know exactly all the functions other than looking at the documentation or using a debugger

Lets get some examples going here is some basic assembler code that i used back in the day

```assembly
; Example 1: Printing a string to the console
mov ah, 09h  ; Function 09h: Display string
mov dx, offset message  ; DX points to string
int 21h ; Call DOS interrupt
mov ah 4ch ; Exit Program
int 21h ; Call DOS Interrupt
message: db 'Hello DOS!', '$'
```

This is as simple as it gets load 09h into AH point DX to your null-terminated string with '$' and call the int 21h that's all it does it displays your text on the screen remember that '$' termination thing it's key if you forget it you might end up printing a whole lot of garbage

Now let's try to do something a bit more complicated:

```assembly
; Example 2: Reading a character from the console
mov ah, 01h ; Function 01h: Read character from stdin
int 21h ; Call DOS interrupt
mov bl, al ; Move character from AL to BL register for later use
mov ah, 02h ; function 02h display character to stdout
mov dl, bl ; Move BL into DL register to be displayed
int 21h ; call DOS interrupt
mov ah 4ch ; Exit Program
int 21h ; Call DOS Interrupt
```
This code reads a character from the keyboard using function 01h and then displays it back to the screen using function 02h easy enough isn’t it?

Now let's say you want to do some file stuff
```assembly
; Example 3: Creating a file and writing some text to it
mov ah, 3ch ; Function 3ch: Create file
mov cx, 0 ; normal file
mov dx, offset filename ; DX points to filename string
int 21h ; Call DOS interrupt
jc error  ; Jump if carry flag is set (error)
mov bx, ax ; AX contains the file handle now move it to bx register
mov ah 40h ; Function 40h write to file
mov cx, message_len ; Number of bytes to write
mov dx, offset message ; DX points to string
int 21h ; call DOS interrupt
mov ah 3eh ; function 3eh close the file
int 21h ; call DOS interrupt
mov ah 4ch ; Exit Program
int 21h ; Call DOS Interrupt
error: ; error handler
mov ah 09h ; function 09h display string
mov dx offset errormsg ; point DX to errormsg
int 21h ; call DOS interrupt
mov ah 4ch ; exit program
int 21h ; call DOS interrupt
filename: db 'myfile.txt', 0 ; null terminated string
message: db 'Hello from the DOS file', 0
message_len equ $-message
errormsg: db "Error creating file", '$'
```

 this one is more involved here we create a file named myfile.txt write the text "Hello from the DOS file" to it and then close it there are a few more registers and parameters to manage here so keep that in mind the error handler is just here to give you an idea of how to deal with errors with the carry flag.

For reading more on these topics I strongly recommend “Programmer’s Guide to the IBM PC” by Peter Norton and Richard Wilton or “Advanced MS-DOS Programming” by Ray Duncan those are the ones i had back in the day and they were lifesavers “Undocumented DOS” as i said before is also a really good source but might be a bit hard to find these days

The important thing with the 21h interrupt is to always look up the correct function number in AH the specific parameter for the other registers and the error codes because if you miss just a small piece of data it can ruin your day and make you question your existence i have seen some weird stuff happen with this 21h interrupt it can make you cry sometimes and other times it just works you know how it is with computers sometimes they feel like they’re working just to make you feel dumb. And it can make you feel a little too smart at the same time this is the beauty of programming it's a love and hate relationship

So to recap 21h interrupt the heart of DOS use it to make things happen remember to use the correct AH functions and keep the book or old forum close to you it can feel ancient but the underlying concepts are still useful even today and there you have it old DOS is dead long live DOS.
