---
title: "vhdl port mapping map different ports of a component into different entities?"
date: "2024-12-13"
id: "vhdl-port-mapping-map-different-ports-of-a-component-into-different-entities"
---

 so you wanna map VHDL ports from a component to different entities thats a classic yeah I get it. Been there done that got the scars to prove it. Let me tell you kid not the easiest thing but definitely doable. Its a bit like trying to untangle a mess of wires behind your computer but hey someone has to do it right

First off lets get the terminology straight we're talking about VHDL right Very High Speed Integrated Circuit Hardware Description Language you knew that I know but gotta make sure we’re all on the same page. We have a component a declared module and we want to use it in different places with different connections that's port mapping in a nutshell. You’ve got your component with its formal ports and then you've got your entities where you instantiate the component and those entities might have their own ports and you want to connect em up right different ports for different entities not a problem I tell ya.

Now the basic thing is the `port map` statement you’ve probably seen that before. The `port map` connects a component's formal ports to actual signals or ports in the entity you're in. This allows you to use the component and its functionality within the bigger design context. The power of port mapping is how it lets you reuse the component multiple times but connecting different things to it. You’re not stuck with a one size fits all approach.

Let's start with an example simple component lets call it `my_adder` its basically just an adder with two inputs and one output.

```vhdl
entity my_adder is
    Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);
           b : in  STD_LOGIC_VECTOR (3 downto 0);
           sum : out STD_LOGIC_VECTOR (3 downto 0));
end my_adder;

architecture Behavioral of my_adder is
begin
  sum <= std_logic_vector(unsigned(a) + unsigned(b));
end Behavioral;
```

So far so good right standard stuff. Now lets say I have two entities that need to use this adder but with different signal names. Here’s where the port mapping magic happens. Lets say one is called `top_module_one` and the other one `top_module_two`.

```vhdl
entity top_module_one is
    Port ( input_x : in  STD_LOGIC_VECTOR (3 downto 0);
           input_y : in  STD_LOGIC_VECTOR (3 downto 0);
           output_z : out STD_LOGIC_VECTOR (3 downto 0));
end top_module_one;

architecture Behavioral of top_module_one is
   signal internal_sum : STD_LOGIC_VECTOR (3 downto 0);
begin
  adder_inst : entity work.my_adder
    port map ( a => input_x,
               b => input_y,
               sum => internal_sum );
   output_z <= internal_sum;
end Behavioral;
```

See here in `top_module_one` I've connected `my_adder`s port `a` to my `input_x` signal and `b` to my `input_y` signal and `sum` to `internal_sum`. This is what its all about. I could call the signals anything I like. The ordering of the `port map` is not really important its all about the names. Now `top_module_two` gets its turn

```vhdl
entity top_module_two is
    Port ( data_in_1 : in  STD_LOGIC_VECTOR (3 downto 0);
           data_in_2 : in  STD_LOGIC_VECTOR (3 downto 0);
           result : out STD_LOGIC_VECTOR (3 downto 0));
end top_module_two;

architecture Behavioral of top_module_two is
    signal inter_add_res : STD_LOGIC_VECTOR (3 downto 0);
begin
   adder_instance : entity work.my_adder
      port map ( a => data_in_1,
               b => data_in_2,
               sum => inter_add_res );
   result <= inter_add_res;
end Behavioral;
```

In `top_module_two` I have a totally different set of signal names I’m mapping `my_adder`s `a` to `data_in_1` `b` to `data_in_2` and the result to `inter_add_res`. It’s the same adder component reused but with totally different input and output signals. This is one of the coolest features that has saved me a lot of time by using components in a generic way. It keeps your code modular and more readable. Imagine if you had to rewrite the same adder logic for every part of the design. It would be a hot mess.

The thing is you are not only dealing with simple signals like `STD_LOGIC_VECTOR`s you might have other types of ports like generics too. A common problem that used to really make me pull my hair out is dealing with generics for different configurations. Sometimes you need different sizes or different behavior. VHDL helps with generics. Let me show you a very simple case.

```vhdl
entity parameterized_adder is
    Generic ( WIDTH : integer := 4);
    Port ( a : in  STD_LOGIC_VECTOR (WIDTH -1 downto 0);
           b : in  STD_LOGIC_VECTOR (WIDTH -1 downto 0);
           sum : out STD_LOGIC_VECTOR (WIDTH - 1 downto 0));
end parameterized_adder;

architecture Behavioral of parameterized_adder is
begin
  sum <= std_logic_vector(unsigned(a) + unsigned(b));
end Behavioral;
```

Here the `parameterized_adder` has a generic called `WIDTH` that dictates the size of the vectors. Now if you instantiate this with different generic values you get different adders. That is how you make things configurable.

```vhdl
entity top_module_three is
    Port ( input_a : in STD_LOGIC_VECTOR(7 downto 0);
           input_b : in STD_LOGIC_VECTOR(7 downto 0);
           output_sum : out STD_LOGIC_VECTOR(7 downto 0);
           input_c : in STD_LOGIC_VECTOR(3 downto 0);
           input_d : in STD_LOGIC_VECTOR(3 downto 0);
           output_res : out STD_LOGIC_VECTOR(3 downto 0));
end top_module_three;

architecture Behavioral of top_module_three is
    signal inter_sum_8bit : STD_LOGIC_VECTOR (7 downto 0);
    signal inter_sum_4bit : STD_LOGIC_VECTOR(3 downto 0);
begin
    adder_instance_8bit : entity work.parameterized_adder
    generic map (WIDTH => 8)
    port map ( a => input_a,
               b => input_b,
               sum => inter_sum_8bit );
    output_sum <= inter_sum_8bit;
    adder_instance_4bit : entity work.parameterized_adder
    generic map (WIDTH => 4)
    port map ( a => input_c,
               b => input_d,
               sum => inter_sum_4bit );
    output_res <= inter_sum_4bit;
end Behavioral;
```

So here in `top_module_three` I instantiate `parameterized_adder` twice once with `WIDTH` of 8 and the second time with `WIDTH` of 4 and of course they have different inputs and outputs. It's a flexible way to make your designs adaptable to different situations. It's so flexible that sometimes I forget where I am at in the code!

When you're doing this especially on bigger systems you need to be really careful with signal naming. It's very easy to get lost in a web of signals and components. The biggest mistake I see is using bad signal names like `signal1 signal2 signal3`. It's important to use descriptive names. If you have a sum signal call it something like `adder_sum` and give it an origin so you can keep track of things easier later when debugging.

One last important point is consistency. If you use one naming convention stick to it. If you mix styles then good luck to you. One thing I’d avoid is mixing the ordering and the name mapping this creates more confusions than benefits. I know sometimes you think you might be clever but believe me stick to one method. I try to always map by name that’s a good habit to develop. It's like when you write code and someone else has to read it six months later so you need to write it for that person.

For learning more about VHDL I'd recommend looking into textbooks like "VHDL Primer" by Bhasker or "Digital Design Principles and Practices" by John Wakerly. Those books are really good for getting the basics and some of the advanced techniques too. And always always experiment and play with the code that’s how you learn best. Remember this is not magic it is just code and it is deterministic once you get a better feeling about how it all works.

So there you go mapping different ports to different entities using VHDL. It's a core concept but when you know how to do it right it becomes quite powerful. It is like a swiss army knife once you master it! It is a lot to take in the first time but over time it becomes second nature you will be writing these things in your sleep.
