---
title: "plotting in fortran library guide?"
date: "2024-12-13"
id: "plotting-in-fortran-library-guide"
---

Okay so plotting in Fortran huh I've been down this road more times than I'd like to admit Back in the day when I was still wrestling with punch cards I had this physics simulation spitting out mountains of numerical data and visualizing it well that was a whole other ballgame Let me tell you it wasn't as straightforward as importing a magic library in Python

Alright so first off Fortran isn’t exactly known for its out-of-the-box graphics prowess It's a number crunching beast not a pixel pushing prodigy But that doesn't mean we're stuck staring at raw data tables forever There are definitely viable options you just need to roll up your sleeves a bit and get your hands dirty

The "classic" approach and I use that term loosely is to use a plotting library that's designed to work well with Fortran often through what we call "binding" These libraries are usually C based or C++ but Fortran is not a lone wolf you can connect with them and leverage them for this purpose It's not the most elegant solution but it gets the job done especially if you’re working with older codebases or specific requirements

I’ve personally used PGPLOT which is a relatively old but very robust library for graphics This library is an old reliable friend I have used in several occasions and I bet it has not been refactored for years but it works so that is what counts and it is extremely stable. Getting it installed and linked up can sometimes be a pain especially when dealing with different compilers and operating systems But once you’ve got it configured you’re pretty much golden for basic 2D plots You'll need to write some glue code in Fortran to call the PGPLOT functions but the documentation is pretty good and it's not rocket science

Here’s a simple Fortran snippet showing a basic 2D line plot using PGPLOT:

```fortran
program plot_example
  implicit none
  real, dimension(100) :: x, y
  integer :: i

  ! Initialize some data
  do i = 1, 100
    x(i) = real(i)
    y(i) = sin(real(i)/10.0)
  enddo

  ! Initialize PGPLOT
  call pgbeg(0, '/xw', 1, 1)  ! Open X window
  call pgscr(0, 1.0, 1.0, 1.0) ! White background
  call pgscr(1, 0.0, 0.0, 0.0) ! Black foreground
  call pgsci(1)

  ! Set up viewport
  call pgvport(0.1, 0.9, 0.1, 0.9)

  ! Set up axes
  call pgbox('BCNST',0.0,0,'BCNST',0.0,0)

  ! Plot the data
  call pglin(x,y,100)

  ! Finish
  call pgeop()
  call pgend()

end program plot_example
```

This example is very basic but it shows how you call the external function after having imported the necessary shared libraries to compile you need to indicate to the compiler the libraries to use and their locations for example `-lpgplot`

Now if you want something more modern and flexible you're probably looking at using something like gnuplot or Python for plotting These libraries aren’t inherently Fortran specific You'd write your Fortran code to output the data to a file and then have another script (gnuplot or python) read that file and generate the plot That’s a bit more involved but it's often the best way to go for complicated visualizations or if you need interactivity

I've been through all kinds of contortions trying to get different systems and libraries to play nice together trust me and you wouldn't believe the amount of time I have spent debugging path settings This might sound cumbersome but it gives you a lot of flexibility and it lets you use the best plotting tools available regardless of their original language

Here’s a Fortran code example that writes data to a file that’s suitable for plotting with gnuplot:

```fortran
program write_data
  implicit none
  real, dimension(100) :: x, y
  integer :: i
  integer :: unit

  ! Initialize some data
  do i = 1, 100
    x(i) = real(i)
    y(i) = cos(real(i)/10.0)
  enddo

  ! Open a data file
  unit = 10
  open(unit=unit, file='data.txt', status='replace')

  ! Write data to file
  do i = 1, 100
    write(unit,'(2f10.5)') x(i), y(i)
  enddo

  ! Close the data file
  close(unit)

end program write_data
```

After you execute this code snippet it will generate a file called `data.txt` that contains two columns of numbers You can then plot this file using gnuplot very simple just do the following:

```gnuplot
plot "data.txt" with lines
pause -1
```

This gnuplot script will open a gnuplot interactive window and display the generated data. Notice that this snippet is not Fortran code this is gnuplot's specific code.

There are also other plotting libraries I’ve played with over the years These are more specialized and it depends a lot on the data you want to plot for instance if you are into scientific visualization you might want to check things like VTK This requires a lot more setup and it’s generally overkill if you just want some simple line graphs but for advanced 3D visualization or complicated data analysis you will probably be using it at some point in your career There's no one-size-fits-all solution here

Okay so I know that we are having some fun here talking about plotting data and I know that I'm digressing here a little but you will see that what I'm about to say is relevant I remember when I was working on that nuclear physics simulation it would have made my life a lot easier if I had a decent plotting workflow. I swear debugging that spaghetti Fortran code and making the visualizations was like trying to debug a kernel driver written in assembly language by a squirrel high on caffeine. That code was so bad I’m still having nightmares about it to this day

Let’s assume that for some reason you don't want to use an external library or output the data into a file to plot them with gnuplot. There are some ways of doing this too like using the standard output for very basic visualizations This method is limited to text-based plots but might be enough for simple use cases when you do not need anything complex or fancy

Here’s an example:

```fortran
program text_plot
  implicit none
  real, dimension(100) :: x, y
  integer :: i, j
  integer, parameter :: height = 20

  ! Initialize some data
  do i = 1, 100
    x(i) = real(i)
    y(i) = 5.0 * sin(real(i)/10.0) + 10.0
  enddo

  ! Scale and shift the data
  integer, dimension(100) :: scaled_y
  real :: min_y, max_y
  min_y = minval(y)
  max_y = maxval(y)
  do i=1, 100
    scaled_y(i) = int(real(height)*(y(i)-min_y)/(max_y - min_y))
  enddo

  ! Plot the data
  do j = height, 1, -1
    do i = 1, 100
      if (scaled_y(i) == j) then
         write(*,'(A)', advance='no') "*"
      else
        write(*,'(A)', advance='no') " "
      endif
    enddo
    write(*,*)
  enddo

end program text_plot
```

This Fortran code example creates an ASCII character representation of a sine wave. The asterisk characters represents the shape of the curve and you are plotting directly from the standard output.

In terms of resources If you want to get a really good grounding in Fortran specifically modern Fortran I highly recommend “Modern Fortran Explained” by Metcalf Reid and Cohen. It’s not directly about plotting but it will teach you the fundamentals so you will understand what you are doing I also suggest to read the documentation about PGPLOT in particular the manual is very exhaustive. If you decide to use gnuplot you should check their documentation too there are tutorials out there that are very good as well as their very good manuals. I also have heard that "Numerical Recipes" is a good source of Fortran code that although old still has some good information about numerical methods and this sort of algorithms that could help you if you decide to implement your plotting algorithms from scratch.

Ultimately the best approach depends a lot on what you need to plot and the context of your project If you have a small dataset and just want a quick visualization gnuplot might be the best bet If you need to interact with the plot or have advanced visualization needs python with matplotlib or VTK could be a good solution For simple plots with a "close to the metal" approach PGPLOT might be what you are looking for It all depends

So yeah plotting in Fortran it’s a journey but one that’s very satisfying when you finally get that beautiful graph that you were looking for. I hope this helps you get started feel free to ask if you have more doubts I’m here to help
