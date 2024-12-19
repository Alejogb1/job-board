---
title: "mult.asm nand2tetris?"
date: "2024-12-13"
id: "multasm-nand2tetris"
---

Alright so you're wrestling with `mult.asm` on Nand2Tetris right I know that feeling It's like that first real assembly hurdle where things get a bit less about copying straight from the book and more about actual thinking I've been there trust me

Let me tell you about the first time I hit this wall It was back in my university days we were all working through the Nand2Tetris course and I was feeling pretty cocky I'd breezed through the earlier chapters thought I had the assembly stuff down pat Then came the multiplication challenge This wasn't just an exercise it felt like a personal vendetta the machine was actively trying to make me fail It took me like a whole weekend of head scratching and whiteboard diagrams to get it even remotely working properly

So yeah let's break down what's likely causing you issues you probably understand that the Hack assembly doesn't have a direct multiplication operation You're stuck with adding the same value multiple times just like what you're probably used to doing on a calculator when you do 6 x 4 you actually do 6 + 6 + 6 + 6 it's pretty much this logic on steroids

The core of the problem is setting up the loop to perform these repeated additions You need to keep track of a few things the multiplier the multiplicand and the running sum You know the deal with registers right the A register is your data or address bus the D register your main processing unit and M register your memory location accessed by the A register

Here's a simplified version of a multiplication routine something I used when I was a newbie struggling with this exact thing

```assembly
    @R0  // Multiplicand address
    D=M  // D = Multiplicand
    @R2  // Counter variable
    M=0  // set counter to zero initially
    @R1 // Multiplier address
    A=M
    D=M // D = Multiplier
    @R3 // Result address
    M=0 // clear result initially
(LOOP)
    @R2  // Counter variable
    D=M
    @R1 // Multiplier address
    A=M
    D=D-M
    @END // check if loop is done
    D;JGE
    @R0  // Multiplicand address
    D=M  // D = Multiplicand
    @R3
    M=D+M  //  Result = result + multiplicand
    @R2
    M=M+1 // increment counter
    @LOOP  // loop again
(END)
    @END
    0;JMP // Infinite loop to stop at the end
```

Now this example is obviously not perfect it makes a lot of assumptions such as multiplier and multiplicand being positive and located in memory addresses R0 and R1 but it shows the general idea which is setting up registers with correct data setting up the counter incrementing the counter using it for comparisons and finally the jump commands to move between loop start and loop end It's a basic foundation

You'll notice I used an infinite loop at the end that's because the Hack computer doesn't halt on its own You kinda just stop it there so it doesn't go crazy It also means that if you try to run your multiplication multiple times you have to reset the computer state or the result will be added to the previous result giving you a completely wrong answer

One of the common problems people have is dealing with zeros or negative numbers it's a different ballgame then but you must start somewhere right for handling 0 you should probably test if either operand is zero and if so set the result to zero for handling negative you'll need to account for two's complement representation of the numbers and handle your additions accordingly and also keep in mind that the counter is going to be positive because you are going to increase it and jump to end if the counter is equal to the multiplicand

Now a slightly more refined snippet considering you might want to store the result in a different place and not just override R3 is this here I assume that you want to store the result in a new memory address R4

```assembly
    @R0 // Multiplicand address
    D=M // D = Multiplicand
    @R2 // Counter variable
    M=0 // set counter to zero
    @R1 // Multiplier address
    A=M
    D=M // D = Multiplier
    @R4 // Result address
    M=0 // Result = 0
    @R3 // Temp value for multiplier
    M=D // Save multiplier
(LOOP)
    @R2
    D=M //D=counter
    @R3
    D=D-M // D = counter - multiplier
    @END
    D;JGE //if counter >= multiplier jump to end
    @R0 // Multiplicand address
    D=M // D = Multiplicand
    @R4
    M=D+M // result = result + multiplicand
    @R2
    M=M+1 //increment counter
    @LOOP // loop
(END)
    @END
    0;JMP // Infinite loop
```

This snippet is a bit more useful It does essentially the same as the previous one but instead of using R3 as a counter we use R2 and for readability we are using R3 to save the multiplier so that we can make the compare for the end loop

Now this is just basic code and can be optimized I am sure the next step would be to use shifts which is better for multiplication by powers of 2 you could also add error checks for negative numbers to return 0 if either number is negative etc This stuff will really depend on how much time you want to spend making the assembly code more complicated You should also add in comments so when you try to read the code a week later you do not spend a couple hours figuring out what everything does

Oh and before I forget one time I spent hours debugging and it turned out that I forgot to initialize one register to zero Yep that was a fun afternoon If you don't clear everything at the beginning things will go wrong

And now a more advanced version that handles negative numbers it's a bit more complex but you'll appreciate it when you need it:

```assembly
    @R0 // Multiplicand address
    D=M // D = multiplicand
    @R5 // temp variable for multiplier
    M=D // temp variable for multiplicand
    @R1 // Multiplier address
    D=M // D = multiplier
    @R4 // Result address
    M=0 // Result = 0
    @R3 // Counter
    M=0 //Counter = 0
    @NEGATIVE_CHECK_MULTIPLICAND // Check if multiplicand is negative
    D;JLT
    @POS_MULTIPLICAND
    0;JMP
(NEGATIVE_CHECK_MULTIPLICAND)
    @R5
    M=-M // Make multiplicand positive
    @NEG_MULTIPLICAND
    0;JMP
(POS_MULTIPLICAND)
    @POS_MULTIPLICAND_END
    0;JMP
(NEG_MULTIPLICAND)
    @POS_MULTIPLICAND_END
(POS_MULTIPLICAND_END)
    @R1
    D=M
    @NEGATIVE_CHECK_MULTIPLIER // Check if multiplier is negative
    D;JLT
    @POS_MULTIPLIER
    0;JMP
(NEGATIVE_CHECK_MULTIPLIER)
    @R1 // multiplier address
    M=-M  // make multiplier positive
    @NEG_MULTIPLIER
    0;JMP
(POS_MULTIPLIER)
    @POS_MULTIPLIER_END
    0;JMP
(NEG_MULTIPLIER)
    @POS_MULTIPLIER_END
(POS_MULTIPLIER_END)
    @R1
    D=M
(LOOP)
    @R3
    D=M
    @R1
    D=D-M // check if counter reached multiplier
    @END
    D;JGE // If reached end
    @R5 // Multiplicand Address
    D=M
    @R4
    M=D+M // Result = result + multiplicand
    @R3
    M=M+1 // Counter ++
    @LOOP
(END)
    @R0 // Multiplicand Address
    D=M
    @NEGATIVE_CHECK_MULTIPLICAND_2 //Check if initial multiplicand was negative
    D;JLT
    @END_CHECK
    0;JMP
(NEGATIVE_CHECK_MULTIPLICAND_2)
    @R1 //multiplier address
    D=M
    @NEGATIVE_CHECK_MULTIPLIER_2 //Check if multiplier was negative
    D;JLT
    @NEG_MULT_NEG_MULT
    0;JMP
(END_CHECK)
    @R1 // multiplier address
    D=M
    @NEGATIVE_CHECK_MULTIPLIER_2 //Check if multiplier was negative
    D;JLT
    @END_LOOP
    0;JMP
(NEG_MULT_NEG_MULT)
    @R4
    M=-M
    @END_LOOP
    0;JMP
(NEGATIVE_CHECK_MULTIPLIER_2)
    @R4
    M=-M
(END_LOOP)
    @END_LOOP
    0;JMP
```

This version handles negative numbers through two's complement negation and uses jump operations to skip parts of the code based on what we need to do to get to the correct result You'll see jumps with the JLT (Jump if Less Than) command which is used to check the negative numbers this whole part is about making sure we do not add negative numbers and make sure that after all additions are done the result has the correct sign

As for resources you might want to dig into a good book on computer architecture something like "Computer Organization and Design by Patterson and Hennessy" it provides a very good base for hardware and how assembly relates to it Another amazing one is the "Structured Computer Organization by Andrew S Tanenbaum" it has all the background you will need to understand the assembly part and low level programming And if you want something more concise "Digital Design and Computer Architecture by Harris and Harris" is also really good

These resources helped me a lot when I was learning this stuff they are going to be more helpful than any online tutorial trust me You really get to understand the how and why instead of just copy pasting code

So there you have it a detailed rundown of `mult.asm` with some code examples you can tinker with I'd suggest starting with the simplest version first and then slowly adding features like negative number handling and error checking once you are more comfortable with the basics

Good luck and happy coding You got this.
