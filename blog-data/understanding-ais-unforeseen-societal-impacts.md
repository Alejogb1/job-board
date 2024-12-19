---
title: "Understanding AI's Unforeseen Societal Impacts"
date: "2024-11-16"
id: "understanding-ais-unforeseen-societal-impacts"
---

dude so this video right it’s all about how ai is gonna totally reshape our world but not in the ways we initially expect kinda like a butterfly effect on steroids  the dude's talking about second-order effects the ripple effects of ai's more direct impacts like  it's not just about whether ai beats humans at chess or go it's about how that affects the whole ecosystem of chess and go players and how we learn those games

first off the visual he shows that graph of professional go player performance after alphago totally blew everyone away you'd think go would be dead but nope the quality of play went up!  it’s like the ai sparked this massive upswing in interest and skill  kinda crazy right then he shows this conway's game of life thing a simple grid-based simulation that generates complex patterns out of simple rules it's like a mini universe born from a few lines of code. and get this he even shows conway’s game of life *implemented* in conway’s game of life which is like inception level meta


a key concept here is *emergent behavior*  basically simple rules interacting at scale create unpredictable complex results. think of an ant colony  each ant follows simple rules but the colony as a whole exhibits incredibly complex behavior like building elaborate nests or finding food efficiently.  ai is kinda the same way.  the video points out that predicting emergent behavior is often NP-complete which means it's computationally hard to solve exactly.  you need to zoom out and look at the bigger picture the macrodynamics rather than getting bogged down in low-level details

another key takeaway is his idea about *who’s doing the learning*  is it the machine or is it the human who's using the machine? he uses the example of drawing software if the ai does all the work you're not learning anything but if it acts as a smart assistant offering suggestions or autocompletion, you still do the work and you learn  he talks about a “slider” where at one end the ai does everything and the other end you do it all. the sweet spot is somewhere in the middle you're using the ai to boost your skills not to replace them think of it like a cheat code for life but the cheat code still makes you better at the game not just skip it entirely


then there's this whole thing about *widening information bandwidth* this one's wild  we currently communicate using language a pretty low-bandwidth system that's also prone to misinterpretations  he suggests that ai can personalize communication tailoring messages to each person's unique understanding and preferences imagine getting an explanation that’s written in your *personal language* a language that incorporates your background experiences and even artistic style  the video even throws in an analogy to the movie *arrival* where the alien language helps humans unlock new cognitive abilities. sounds pretty sci-fi but that's the point it's extrapolating current trends to their logical extremes


he uses some pretty neat examples like this:

```javascript
// simple example of personalized feedback in a drawing app
function provideFeedback(userStroke, aiModel) {
  let feedback = aiModel.analyzeStroke(userStroke); // analyze user's drawing stroke
  if (feedback.suggestion) {
    console.log("AI suggestion:", feedback.suggestion);
    // show the suggestion to the user like "try adding more curves here"
  }
  // adapt the AI to user's actions and styles
  aiModel.learnFromStroke(userStroke); 
}
```
this code snippet shows a basic idea of how an ai could analyze a user's drawing and offer personalized feedback, improving the learning process.


another example is changing how we interact with UIs he talks about how a simple tap or stroke on a screen could mean many different things and this information depends on context not just the tool selected.  he argues for using machine learning to interpret user actions based on a rich context including recent actions, current state, and even their overall goals

```python
# a super simplified example of context-aware UI interactions
class ContextAwareUI:
    def __init__(self):
        self.context = {}  # store user context like selected tools, recent actions etc

    def handleGesture(self, gesture, location):
        self.context['lastGesture'] = gesture # update context
        if gesture == 'tap' and self.context.get('selectedTool') == 'lasso':
          # select object at location
          self.selectObject(location)
        elif gesture == 'stroke' and self.context.get('selectedTool') == 'pen':
          # draw a line, but behavior adjusts based on location relative to existing objects
          self.drawStroke(location, self.context.get('nearbyObjects'))
        else:
            # handle other cases default behavior or error handling
            pass
```
this snippet demonstrates the idea of a context aware UI: the same gesture ('tap' or 'stroke') can have different meanings depending on the current state of the app and the previous actions made by the user.


and finally he gets into this idea of  extrapolating a quantity or quality to an extreme  he uses the example of smalltalk a programming language from the 70s that used message passing between objects as if each object were a computer.  it had a huge issue: each “computer” was resource intensive which limited the total number of objects.  the solution modern machine learning uses is having many simple dumb nodes instead of few super smart ones.  it's like a massive distributed network that gains power through sheer scale rather than relying on the intelligence of individual components

```c++
// a simplified representation of a massively parallel system
#include <vector>

struct Node {
  //Each node has a state and some simple logic
  int state;
  void process(std::vector<Node>& neighbors) {
    // update state based on neighbor states; keep simple to maintain efficiency
    // this would be the actual algorithm for interaction between nodes
    // very important: use fast, simple operations here
  }
};

int main() {
  std::vector<Node> nodes(1000000); // a million dumb nodes!
  for (int i = 0; i < 10000; ++i) { // simple processing loop
    for (auto& node : nodes) {
      node.process(nodes); //each node updates state based on neighbors. parallel processing is crucial here
    }
  }
  return 0;
}
```
This C++ code snippet illustrates the concept of a massively parallel system with a million simple nodes, each updating its state based on interactions with its neighbors. The key is the simplicity of each node, allowing for massive scalability.  the point is not what *each node* can do but what the *entire network* does as a whole.


overall the video’s message is this:  ai isn’t just about building smarter machines  it’s about understanding and anticipating the cascading effects of those machines on society and on how *we* learn and interact with the world.  it's less about the destination and more about the journey the unexpected turns the crazy emergent behavior that comes along the way.  it’s about embracing the chaos and figuring out how to use it to our advantage before it just happens to us. pretty wild right
