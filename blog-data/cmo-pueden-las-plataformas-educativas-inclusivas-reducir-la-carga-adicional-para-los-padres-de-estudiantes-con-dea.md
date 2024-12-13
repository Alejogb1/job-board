---
title: "¿Cómo pueden las plataformas educativas inclusivas reducir la carga adicional para los padres de estudiantes con DEA?"
date: "2024-12-12"
id: "cmo-pueden-las-plataformas-educativas-inclusivas-reducir-la-carga-adicional-para-los-padres-de-estudiantes-con-dea"
---

okay so how do inclusive edtech platforms actually lighten the load for parents dealing with kids with learning differences like ADHD and dyslexia its a real thing and parents are often doing the work of like three full-time jobs just trying to keep things going the way it is right now we can break this down into a few key areas where tech can help a lot

first off a huge part of the parental burden is simply keeping track of everything deadlines assignments which materials to use if you’ve got a kid with a learning difference this is often magnified because theres a lot more moving pieces individualized learning plans specific accommodations things that might look very different from a typical classroom setting right so a good platform should have a centralized dashboard a place where everything is visible assignments deadlines messages from teachers all of it should be presented in a clear and concise format that can easily be checked by both parents and students this eliminates the need for parents to constantly nag their kids or dig through emails and folders it puts the information in one spot and its constantly updated

think about it like a project management tool but designed for schoolwork things like this eliminate the feeling of constantly chasing paper and updates and it gives parents a way to be involved without becoming overwhelmed this also includes making sure the platform has good accessibility options built in features like text-to-speech adjustable font sizes and color schemes this isn't just about meeting accessibility guidelines it's about ensuring the platform is usable for the widest range of users including parents themselves who might also have their own tech hurdles to deal with

now let's talk about communication gaps a major stressor for parents is often the lack of clear communication between themselves their kids and the teachers its not about pointing fingers but the flow of information can be clunky its like a game of telephone things get lost in translation or delayed platforms should really streamline communication not add another layer of confusion ideally this means built-in messaging systems that allow parents to quickly communicate with teachers about specific issues ask questions and receive feedback on their childs progress without the need for separate email chains and phone calls this is super important to ensure that everyone is on the same page and that small concerns don’t escalate into bigger problems

another big piece is that individualized learning platforms can help bridge the gap when a teacher is managing a classroom of say 25 kids and a child needs specific types of help and a parent is not an expert on pedagogy these platforms can create personalized learning paths and deliver individualized assignments and materials based on each students specific learning needs a kid with dyslexia might benefit from text-to-speech or visual aids while a student with ADHD might need shorter more frequent assignments with regular feedback loops this type of personalization takes pressure off the parents who can then focus on supporting their kid instead of feeling like they have to create custom curriculum at home

also good edtech platforms integrate with other learning resources they’re not isolated systems if there is a good online resource or software that is specific to the learning profile of a kid the platform should be able to sync or integrate these tools this also reduces the complexity of managing and using multiple applications its about creating a unified learning ecosystem not a jumbled mess of tech tools that parents have to somehow stitch together

consider how much time parents spend searching for help at home and trying to teach a specific topic or skill its a constant cycle of frustration that a good learning platform can solve lets say a child struggles with math instead of asking a parent to spend hours on tutoring or researching teaching strategies the platform can direct the child to targeted practice exercises personalized instructional videos and perhaps even interactive problem-solving simulations this self-directed learning is crucial for independence and also frees up parent time by being the first point of contact for the learning experience

also analytics and data driven insights good edtech platforms should be providing parents and educators with data that is actionable not just endless data dumps this includes progress monitoring reports highlighting areas of strength and weakness and providing suggestions for further support so this will help parents know where and how to guide a child in a specific study. this also helps facilitate discussions between parent teacher and child making sure that everyone is aware of both the progress and the challenges faced

and of course all this is contingent on usability if the technology is clunky unreliable or difficult to navigate it will add more work not less so good platform design has to be accessible user friendly and intuitive ideally it should have a clean interface clear icons and easy navigation menus the user experience for parents must be taken into account if this is not the case parents will be turned off by the technology which would be a net negative

here's a snippet of python-like pseudocode to illustrate how assignment management can be centralized

```python
class Assignment:
  def __init__(self, title, due_date, materials, student_id):
    self.title = title
    self.due_date = due_date
    self.materials = materials #list of file paths or web links
    self.student_id = student_id

class Dashboard:
  def __init__(self):
    self.assignments = {} #key=student_id ,value =list of assignments

  def add_assignment(self, assignment):
    if assignment.student_id not in self.assignments:
      self.assignments[assignment.student_id] = []
    self.assignments[assignment.student_id].append(assignment)

  def get_assignments(self, student_id):
      return self.assignments.get(student_id,[])
```

in this simplification the dashboard class manages assignments allowing to retrieve them based on student id and adding assignments a very basic simplification that makes the core function of centralization of assignments clear.

here's an example of the message system

```javascript
const messageList = document.querySelector('#messageList');
const messageInput = document.querySelector('#messageInput');
const sendButton = document.querySelector('#sendButton');
const userId = 'parent-123'; //hard coded for now for illustration

sendButton.addEventListener('click', () => {
  const messageText = messageInput.value;
    if (messageText.trim() !== ''){
       const newMessage = document.createElement('li');
       newMessage.textContent = `${userId}: ${messageText}`
       messageList.appendChild(newMessage);
       messageInput.value = ""; //reset message
  }
})

```

a very simple example using client side javascript for a chat interface where a message is sent and displayed instantly for a specific user. a backend system would be required to have different users and persistent messages but the core functionality is shown.

and here is a rough outline of an individualized learning path

```java
class LearningPath {
    private List<Module> modules;
    private Map<String,Double> progress; //moduleid and completed percentage
    private double progressThreshold;
    private LearningProfile profile;

    public LearningPath(LearningProfile learningProfile){
        this.modules=new ArrayList<>();
        this.progress = new HashMap<>();
        this.progressThreshold = 0.80;
        this.profile = learningProfile;
    }
    public void addModule(Module module){
      this.modules.add(module);
    }

    public Module getNextModule(){
       for (Module module : this.modules) {
           if(this.progress.getOrDefault(module.id,0.0) < this.progressThreshold) return module;
       }
       return null; //if all are completed
    }
}
```

a very basic learning path definition in java showing how it would iterate through modules based on a progress and a threshold and linked to a learning profile (not defined in this example).

now instead of links i would suggest a few things for further investigation in the form of papers and books if someone is curious about diving deep the first is the book "universal design for learning theory and practice" by anne meyer david rose and david gordon its a good starting point for anyone interested in understanding the principles of UDL its a great framework for design decisions and has a lot of examples of how to address various learning needs another book that is helpful is "the dyslexic advantage unlocking the hidden potential of the dyslexic brain" by brock l and fernette eide its a good read because it offers a different perspective focusing on the cognitive advantages of learning differences

a research paper that is always cited is the "effectiveness of computer based interventions for students with learning disabilities a meta analysis" by stacey schneider and her team this paper analyzes a bunch of research to show the impact of tech in special education a more recent paper on personalized learning “personalizing learning at scale the digital learning journey” by heather staker and her team can give a more holistic look at all the different facets of personalized learning

so its not just about flashy features its about designing for genuine usability accessibility and integration with other learning resources that can make a very meaningful difference in reducing the load for parents of kids with learning differences
