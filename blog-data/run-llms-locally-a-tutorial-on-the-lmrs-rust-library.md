---
title: "Run LLMs Locally: A Tutorial on the LMRS Rust Library"
date: "2024-11-16"
id: "run-llms-locally-a-tutorial-on-the-lmrs-rust-library"
---

dude so this talk was awesome right  phil packs the aussie dude from sweden  totally nailed it talking about lmrs  it's like this super cool rust library for running massive language models locally on your own machine  no more cloud charges  no more waiting for responses from some server halfway across the planet it's all right there on your laptop  or maybe even your toaster if you're feeling adventurous

the whole point of the video  was basically to show off lmrs  but it also took this amazing deep dive into why local LLMs are such a game changer and how this library makes it all possible  he spends like half the time explaining the why and then the other half on the how and the cool stuff you can do with it

first thing that hit me was this chart he showed  like this massive graph comparing the sizes of different LLMs  gpt-3 gpt-4  all these behemoths on one side and then the more modest open-source models on the other  it was hilarious how much bigger the closed-source ones are they're practically dinosaurs compared to the others  he even joked that nobody actually knows how big gpt-4 really is it's a total mystery  the size is kind of a proxy for how "smart" they are  bigger models generally do better  but the size also determines how much memory and processing power you need to run them which is why local models tend to be smaller  but often still surprisingly powerful for their size

then he goes into these key differences between cloud vs local models speed  latency cost and privacy  cloud models are ridiculously fast because they run on specialized hardware but they cost money  for every token you send and receive  local models are slower  but the cost is just your electricity bill  plus the hardware cost if you don't own it already  and the latency is much lower  you get answers basically instantly  because you're not waiting for a round trip to a remote server this is super important for conversational AI because the model can react in real time  and finally privacy  well  you don't want your personal embarrassing questions to end up on some data center somewhere  local models keep your data local

he explained quantization  which is how they make these massive models actually fit and run on regular computers  imagine the model as a giant image a ridiculously high-res image  you can shrink it down by lowering the resolution right you still have the image  but it's much smaller and easier to handle  that's kinda what quantization does it reduces the precision of the model's parameters  making it much smaller without losing too much of its intelligence  it's like magic

then the real fun starts  he dives into lmrs itself   he talks about how it all began  this whole saga of building it in rust  because it's better than other languages at this kinda thing he even mentions that he got totally beat to the punch by someone else but they collaborated and now he's a maintainer  it's a total underdog story which adds to the appeal  the library has these core design principles  it has to be a real library  not just a single application  it has to be highly customizable  support tons of model architectures  and run on pretty much any hardware  windows linux mac  cpu gpu  maybe even a raspberry pi  pretty crazy right


here's a bit of the code he showed its awesomely minimal


```rust
use lmrs::*;

async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model.  path to your model
    let model = Model::new("path/to/your/model.bin").await?;

    // Create a session
    let mut session = model.create_session()?;

    // Generate text.  prompt you're gonna send it
    let prompt = "Once upon a time,";
    let tokens = session.infer(prompt, 100).await?;


    // Do something with the generated tokens
    println!("Generated text: {:?}", tokens);

    Ok(())
}
```

this little snippet shows how easy it is to use lmrs  you load a model  create a session which is like a persistent connection to the model  send it a prompt and get the generated text back  pretty straight forward

another example he showed involved customizing the generation process  giving you way more control than cloud APIs generally offer


```rust
use lmrs::*;

// ... (model loading and session creation as before) ...

let params = InferenceParameters::default()  //you start with the defaults
  .with_top_p(0.95) //and you change what you want to change
  .with_top_k(40);

let tokens = session.infer_with_params(prompt, params, 100).await?;


```

see  you can tweak things like `top_p` and `top_k`  which control how the model samples tokens  these are crucial parameters  affecting the randomness creativity and coherence of the generated text  having this level of control is a huge advantage over cloud APIs  where you're often stuck with whatever parameters they provide

and here's one more example showing how you can integrate lmrs into a more complex system  this is like a tiny example but it could form the core of much larger applications


```rust
use lmrs::*;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... (model loading and session creation) ...

    let (tx, mut rx) = mpsc::channel(100); //channel to communicate
    let session_clone = session.clone(); //clone session for sending to another task

    tokio::spawn(async move { //spawn new task for handling model
        while let Some(prompt) = rx.recv().await {
            let tokens = session_clone.infer(&prompt, 100).await.unwrap();
            // ... (process the generated tokens and send back the response somehow) ...
        }
    });


    // ... (your main application logic, sending prompts to the channel) ...
    tx.send("What is the meaning of life?".to_string()).await?;
     // ... (wait for responses from the model) ...
     //more stuff

    Ok(())
}
```

this example uses a channel  to handle requests asynchronously  this is essential for building more robust apps  the main thread sends prompts to the channel  and a separate thread processes them using the model  you can have as many such threads as you want  or have them pick up prompts from different sources

he then went on to show off some pretty cool projects built using lmrs  a local ai app that's insanely simple  lm chain a rust version of langchain that lets you chain different llm prompts together for complex tasks  and flum a flowchart-based app for building custom workflows using the library  pretty neat


he also talked about his own applications  an lm powered discord bot  a github copilot clone  and a super practical  real-world application for extracting structured data from wikipedia pages  he used gpt-3 initially but it was too slow and expensive  so he fine-tuned his own small model using a tool called axel  and now he has a super efficient pipeline


the benefits of local models and lmrs  were huge he highlighted deployment ease  rusts excellent cross-platform compatibility makes it super easy to ship binaries that "just work"  the robust rust ecosystem  it's super easy to integrate llms into all kinds of projects  and the fine-grained control over how models generate text  you don't have to rely on the cloud provider's limited APIs

but he didn't just paint a rosy picture  he addressed the downsides too  hardware limitations  the trade-offs between model size speed and quality  the constantly evolving llm ecosystem  which can break existing applications  and the licensing complexities of open-source models  it's a double edged sword   but overall it was  an optimistic and very informative talk


in the end  phil basically said local models are awesome  lmrs is pretty amazing too  and the whole field is changing super fast  but there are still challenges  but also  a massive amount of opportunity to innovate and create cool stuff  he closed by inviting people to contribute to lmrs  which is a great community project
