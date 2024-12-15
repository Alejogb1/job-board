---
title: "How can I add math symbols to altair plots?"
date: "2024-12-15"
id: "how-can-i-add-math-symbols-to-altair-plots"
---

alright, so you're looking to get some math symbols into your altair plots, eh? i've been down that road myself. it's not always as straightforward as just typing symbols in, that's for sure. early in my career, i remember trying to label some charts for a physics simulation project and the usual text just didn't cut it. i ended up with these weird looking plots where the axis labels were just placeholder text and some unreadable character combinations that made no sense to anyone. looked pretty bad. learned the hard way that you need a bit of a workaround.

basically, altair itself doesn't directly render complex mathematical symbols using things like latex commands you would in a document. it leans heavily on vega-lite, which, at its core, treats text as, well, text. it's not equipped to interpret that stuff without some help. so the strategy is to use a mark type that interprets some kind of markup as math. in this case, we're talking about using the 'text' mark together with latex. not pure latex but a flavor of it.

here’s the gist of it: we're gonna embed the latex-like math into the text mark using unicode for the text string. then we make sure that mark type supports that flavor of latex.

first, let's start with a simple example using a basic greek symbol. often, you would like to label your axis or points with greek symbols and the way to go around is using unicode notation in a string inside altair.

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3, 4, 5],
        'y': [2, 5, 8, 2, 7]}
df = pd.DataFrame(data)

chart = alt.Chart(df).mark_point().encode(
    x=alt.X('x', title='x-axis (\u03B1)'),
    y=alt.Y('y', title='y-axis (\u03B2)')
).properties(
    title='simple plot with greek symbols in axis'
)
chart.show()

```

in this first snippet, `\u03B1` and `\u03B2` are unicode characters for the greek letters alpha and beta respectively. you can find lists of unicode characters all over the internet, or even in many ide code completion tips. the `alt.x` and `alt.y` encode functions accept that unicode string and will display it. easy right?.

now, this works for simple characters, but what if we want a more complex expression, like a fraction or a square root? here's where the fun starts. we're going to use latex-like markup. but altair doesn't directly process latex. it does, however, support a vega-lite's expression engine which can interpret certain latex-like strings. so, we can use the mark `text` to render these expressions. we embed the latex-like string inside a `text` mark, with an offset to place it correctly near a point, or even at the title of the plot.

let's say i needed to show the famous quadratic formula in one of my plots many years ago. i struggled with getting that into the plot. i tried using a 'label' chart type at first. and man...it was quite the rabbit hole back then. so, after banging my head against the wall for a while, here's what i eventually did.

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3],
        'y': [2, 5, 8]}
df = pd.DataFrame(data)

formula_text = r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"

chart = alt.Chart(df).mark_point(size=100).encode(
    x='x',
    y='y'
).properties(
    title='plot with complex latex-like expression'
)
text_chart = alt.Chart(pd.DataFrame({'x': [2], 'y': [7]})).mark_text(
    text=formula_text,
    fontSize=16,
    align='center',
    baseline='middle',
).encode(x='x', y='y')
(chart+text_chart).show()
```

in that code, we create a scatter plot, just as in the first example. but, instead of using titles, we add a text mark containing the quadratic formula as a latex-like string `x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}`. this string is not a fully fledged latex document, but the engine interprets `\frac` for fractions and `\sqrt` for square roots. and because we use a raw string with the `r` prefix the backslashes are interpreted correctly, you have to make sure to use it when you have that kind of strings.

important thing to mention is that you might have to play a bit with the `x` and `y` coordinates in the `text_chart` to correctly position the formula relative to your plot elements. Also, you can specify font size, alignment and baseline for a better presentation.

one thing to keep in mind is that this implementation, like vega-lite itself, uses a specific subset of latex syntax. you're not getting the full power of a latex document here. you're limited to some basic elements like fractions, square roots, greek letters and some other specific notation. there are no document commands, you can't include other packages or anything similar to that.

now, let's make a more complex expression in the title of a plot. this is something i really wish i knew when i was making plots of gaussian distributions at university. it would have saved me some time.

```python
import altair as alt
import pandas as pd

data = {'x': [1, 2, 3, 4, 5],
        'y': [2, 5, 8, 2, 7]}
df = pd.DataFrame(data)
formula_title = r'f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}'

chart = alt.Chart(df).mark_line().encode(
    x=alt.X('x', title='x axis'),
    y=alt.Y('y', title='y axis')
).properties(
    title=alt.Title(formula_title, anchor='middle', offset=10, fontSize=16)
)
chart.show()
```

here we have a gaussian function in the title. note that it uses the `alt.title` object with a `string` in it, just as in our formula plot example. and you can specify properties as `anchor`, `offset`, or `fontSize` for presentation.

now, about resources, i'd recommend having a good look at the vega-lite documentation on text marks. that's your go-to for getting the specifics of how vega-lite interprets text as latex-like markup. "the grammar of graphics" by wilkinson is a very good book if you want to dig deeper into the concepts behind the theory of plots and visualization and then there are also plenty of good books on scientific plotting like "matplotlib plotting cookbook" but that will not be super useful for altasir directly. also, don't forget to check the altair documentation itself. it's really very good. oh, and, speaking of altair, i have to say, it's not just about the math. i used it to plot sales data once and the finance department thought i was some kind of data magician when i presented my results. i swear, they nearly hired a fortune teller. (they didn't, thankfully).

so, to recap, you're mostly going to use unicode or latex-like text strings within text marks and titles to embed math expressions. altair itself is just going to treat that text as text and then the vega-lite rendering engine is going to process these strings into graphical symbols. it is a very useful way to add mathematical notation, but you will have to remember its limitations. it's something you'll get the hang of with a little practice and experimentation. you'll have beautiful plots with mathematical expressions in no time.
