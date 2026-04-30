# Diátaxis Documentation Framework - Knowledge Base

> **Purpose**: This document contains distilled knowledge from the Diátaxis framework to guide consistent, high-quality documentation creation. Use this as your reference when building documentation.

---

## Core Concept

Diátaxis identifies **four fundamental types of documentation** that serve distinct user needs:

1. **Tutorials** - Learning-oriented (acquisition + action)
2. **How-to Guides** - Goal-oriented (application + action)
3. **Reference** - Information-oriented (application + cognition)
4. **Explanation** - Understanding-oriented (acquisition + cognition)

## The Two-Dimensional Map

Documentation exists across two dimensions of craft:

### Dimension 1: Action vs Cognition
- **Action**: Practical steps, doing (Tutorials, How-to Guides)
- **Cognition**: Theoretical knowledge, thinking (Reference, Explanation)

### Dimension 2: Acquisition vs Application
- **Acquisition**: Study, learning skills (Tutorials, Explanation)
- **Application**: Work, applying skills (How-to Guides, Reference)

```
                    ACQUISITION (Study)
                           |
        TUTORIALS          |          EXPLANATION
    (learning-oriented)    |      (understanding-oriented)
         "Can you          |           "Why...?"
       teach me to...?"    |
                           |
ACTION -------------------|------------------- COGNITION
    (practical steps)      |         (theoretical knowledge)
         "How do I...?"    |           "What is...?"
                           |
      HOW-TO GUIDES        |          REFERENCE
    (goal-oriented)        |      (information-oriented)
                           |
                    APPLICATION (Work)
```

## The Diátaxis Compass (Decision Tool)

Use this to classify any content:

| If the content... | ...and serves the user's... | ...then it belongs to... |
|-------------------|----------------------------|--------------------------|
| informs action | acquisition of skill | **Tutorial** |
| informs action | application of skill | **How-to Guide** |
| informs cognition | application of skill | **Reference** |
| informs cognition | acquisition of skill | **Explanation** |

**Quick questions to ask:**
- Action or cognition? (doing vs knowing)
- Acquisition or application? (study vs work)

---

## 1. TUTORIALS

### Purpose
A **lesson** - a learning experience that builds confidence and basic competence through guided practice.

### Key Characteristics
- **Learning-oriented**: Serves acquisition of skills
- **Practical activity**: User learns by doing
- **Teacher-student relationship**: Teacher has all responsibility
- **Safe environment**: Must be repeatable, reliable
- **Concrete and specific**: Focus on particular examples

### Core Principles

#### DO:
✅ **Show where they'll be going** - Set expectations at the start  
✅ **Deliver visible results early and often** - Every step produces comprehensible result  
✅ **Maintain narrative of expected outcomes** - "You will notice that..."  
✅ **Point out what learner should notice** - Close learning loops  
✅ **Target the "feeling of doing"** - Create flow and rhythm  
✅ **Encourage repetition** - Make steps repeatable  
✅ **Focus on concrete, specific examples** - This problem, this action, this result  
✅ **Ignore options and alternatives** - Single path to conclusion  
✅ **Aspire to perfect reliability** - Must work every time  

#### DON'T:
❌ **Don't try to teach by explaining** - Let learning happen through doing  
❌ **Don't include extensive explanation** - Link to it instead  
❌ **Don't offer choices** - Keep single, clear path  
❌ **Don't assume prior knowledge** - Be explicit about basics  
❌ **Don't use abstract concepts** - Stay concrete and particular  

### Language Patterns
- "We..." (first-person plural - we're in this together)
- "In this tutorial, we will..."
- "First, do x. Now, do y. Now that you have done y, do z."
- "The output should look something like..."
- "Notice that... Remember that... Let's check..."
- "You have built a..."

### Analogy
**Teaching a child to cook** - What matters is what the child learns and their pleasure in the experience, not the culinary outcome.

---

## 2. HOW-TO GUIDES

### Purpose
**Directions** that guide the already-competent user through a real-world problem to achieve a specific goal.

### Key Characteristics
- **Goal-oriented**: Addresses specific tasks/problems
- **Work-focused**: Serves users applying their skills
- **Assumes competence**: User knows what they want to achieve
- **Real-world**: Deals with actual complexity
- **Result-focused**: Guides to successful outcome

### Core Principles

#### DO:
✅ **Address real-world problems** - Written from user perspective, not machinery  
✅ **Maintain focus on the goal** - No digression, explanation, or teaching  
✅ **Address real-world complexity** - Adaptable to various use-cases  
✅ **Omit the unnecessary** - Practical usability > completeness  
✅ **Provide executable solutions** - Clear sequence of actions  
✅ **Seek flow** - Ground in user's activities and thinking  
✅ **Pay attention to naming** - "How to integrate X" not just "X"  

#### DON'T:
❌ **Don't teach** - User already has skills  
❌ **Don't explain why** - Link to explanation instead  
❌ **Don't provide complete reference** - Link to it instead  
❌ **Don't define by machinery operations** - Define by user needs  

### Language Patterns
- "This guide shows you how to..."
- "If you want x, do y. To achieve w, do z." (conditional imperatives)
- "Refer to the x reference guide for..."

### Distinction from Tutorials
| Tutorial | How-to Guide |
|----------|--------------|
| Learning experience | Directs work |
| Carefully-managed path | Real-world, unpredictable |
| Contrived, safe setting | Real world |
| Eliminates unexpected | Prepares for unexpected |
| No choices/alternatives | Forks and branches |
| Must be safe | Cannot promise safety |
| Teacher has responsibility | User has responsibility |
| Concrete and particular | General approach |
| Teaches general skills | Completes particular task |

### Analogy
**A recipe** - Addresses specific question, requires basic competence, focuses only on how to make the dish.

---

## 3. REFERENCE

### Purpose
**Technical description** of the machinery - accurate, complete, reliable information for users at work.

### Key Characteristics
- **Information-oriented**: Contains propositional/theoretical knowledge
- **Austere**: One consults it, doesn't read it
- **Authoritative**: No doubt or ambiguity
- **Structured by machinery**: Mirrors product architecture
- **Neutral**: Not concerned with what user is doing

### Core Principles

#### DO:
✅ **Describe and only describe** - Neutral description is key  
✅ **Adopt standard patterns** - Consistency is crucial  
✅ **Respect structure of machinery** - Mirror product architecture  
✅ **Provide examples** - Illustrate without explaining  
✅ **Be accurate, precise, complete** - Truth and certainty  

#### DON'T:
❌ **Don't instruct** - Link to how-to guides  
❌ **Don't explain** - Link to explanation  
❌ **Don't discuss or opine** - Stay factual  
❌ **Don't vary format** - Maintain consistency  

### Language Patterns
- "X is available as `code.path` and defined in `file.py`"
- "Sub-commands are: a, b, c, d, e, f." (lists)
- "You must use a. You must not apply b unless c. Never d." (warnings)

### Analogy
**Information on food packet** - Facts presented in standard ways, governed by law-like seriousness.

---

## 4. EXPLANATION

### Purpose
**Discursive treatment** that deepens understanding, provides context, and answers "why?"

### Key Characteristics
- **Understanding-oriented**: Serves acquisition through cognition
- **Reflective**: Occurs after and depends on something else
- **Higher/wider perspective**: Joins things together
- **Can be read away from product**: "Bath-time reading"
- **Permits opinion and perspective**: Discussion, not instruction

### Core Principles

#### DO:
✅ **Make connections** - Weave web of understanding  
✅ **Provide context** - Why things are so, design decisions, history  
✅ **Talk *about* the subject** - "About X" should fit title  
✅ **Admit opinion and perspective** - Consider alternatives  
✅ **Keep closely bounded** - Don't absorb instruction/reference  

#### DON'T:
❌ **Don't instruct** - Not for active work  
❌ **Don't provide reference** - Link to it  
❌ **Don't try to be complete** - Draw reasonable boundaries  

### Language Patterns
- "The reason for x is because historically, y..."
- "W is better than z, because..."
- "An x in system y is analogous to a w in system z. However..."
- "Some users prefer w (because z). This can be a good approach, but..."
- "An x interacts with a y as follows:..."

### Alternative Names
- Discussion
- Background
- Conceptual guides
- Topics

### Analogy
**"On Food and Cooking" by Harold McGee** - Places subject in context of history, society, science. Read for reflection, not while cooking.

---

## Working with Diátaxis

### Workflow (Iterative Approach)

1. **Choose something** - Any piece of documentation (or start from nothing)
2. **Assess it** - What user need? How well does it serve? What can improve?
3. **Decide what to do** - What single next action will improve it?
4. **Do it** - Complete that action and publish/commit
5. **Repeat** - Go back to step 1

### Key Principles

**Use as a guide, not a plan**
- Don't create empty structures
- Structure emerges from inside-out improvements
- Documentation changes from within

**Work one step at a time**
- Every step in right direction worth publishing immediately
- Don't try to work on big picture
- Small steps lead to destination

**Allow organic growth**
- Like a living organism - always complete, never finished
- Structure develops from healthy cells (well-formed content)
- At every stage: useful, appropriate, ready for next stage

**Don't worry about structure initially**
- Structure will emerge as you improve content
- Follow Diátaxis principles, structure follows naturally

---

## Quality in Documentation

### Functional Quality (Objective, Measurable)
- Accuracy
- Completeness
- Consistency
- Usefulness
- Precision

**Characteristics:**
- Independent of each other
- Objective (measured against world)
- Aspects of constraint
- Condition for deep quality

### Deep Quality (Subjective, Experiential)
- Feeling good to use
- Having flow
- Fitting to human needs
- Being beautiful
- Anticipating the user

**Characteristics:**
- Interdependent
- Subjective (assessed against human needs)
- Aspects of liberation
- Conditional upon functional quality

**Diátaxis Role:**
- Cannot address functional quality directly
- Exposes lapses in functional quality
- Helps create deep quality by fitting user needs and preserving flow

---

## Common Pitfalls to Avoid

### Blurring Boundaries
The most common problems arise from mixing types:

| Blur | Problem |
|------|---------|
| Tutorial + How-to | Confuses learning with work - deadly for newcomers |
| Tutorial + Explanation | Breaks learning flow with distracting theory |
| How-to + Reference | Pollutes practical guide with completeness |
| Reference + Explanation | Interrupts description with digression |

### The Tutorial/How-to Confusion
**Most common and harmful conflation**

Remember:
- Tutorials = **Study** (acquisition of skills)
- How-to = **Work** (application of skills)
- NOT Basic vs Advanced
- Both can be simple or complex

---

## Quick Reference Card

### When writing, ask:

**For any content:**
1. Action or cognition?
2. Acquisition or application?
3. What user need does this serve?

**For tutorials:**
- Does it provide a learning experience?
- Is it safe and repeatable?
- Does it minimize explanation?

**For how-to guides:**
- Does it address a real-world problem?
- Does it assume competence?
- Does it maintain focus on the goal?

**For reference:**
- Is it neutral description only?
- Does it mirror product structure?
- Is it consistent and authoritative?

**For explanation:**
- Does it provide context and understanding?
- Can it be read away from work?
- Does it make connections?

---

## Implementation Notes

### Starting Fresh
1. Don't create empty four-section structure
2. Start with what you have (or one piece)
3. Improve iteratively using compass
4. Let structure emerge naturally

### Improving Existing Docs
1. Pick any piece of content
2. Use compass to identify what it should be
3. Make one improvement
4. Publish immediately
5. Repeat

### Maintaining Quality
- Tutorials require most maintenance (end-to-end journey)
- Reference can often be auto-generated
- How-to guides most-read section
- Explanation least urgent but equally important

---

## Remember

> "The best way to get started with Diátaxis is by applying it - to something, however small."

> "Documentation is never finished, but can always be complete."

> "Don't try to teach - allow learning to take place."

> "Good documentation feels good - it fits and moves with you."

---

**This knowledge base captures the essence of Diátaxis. Refer to it when:**
- Planning new documentation
- Improving existing documentation
- Unsure which type of documentation to write
- Reviewing documentation quality
- Training others on documentation

**The framework works because it serves user needs systematically across the two fundamental dimensions of craft: action/cognition and acquisition/application.**