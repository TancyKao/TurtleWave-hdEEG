# TurtleWave Documentation Style Guide

## Philosophy: Documentation as User Experience

Great documentation isn't just informative—it's a pleasure to read and makes users want to come back. Every page should feel like a conversation with a knowledgeable friend who genuinely wants to help.

## Emoji Usage Policy

**Use emojis sparingly and only when they add genuine value.**

### When to Use Emojis

✅ **Acceptable uses:**
- Section navigation in index/README files (e.g., "📚 Tutorials")
- Status indicators (✅ ❌ ⚠️) in lists or tables
- Sparingly in admonition titles when it enhances clarity

❌ **Avoid:**
- Decorative emojis that don't add meaning
- Multiple emojis in body text
- Emojis in technical documentation or reference pages
- Emojis in code examples or command outputs

### Guideline

Ask: "Does removing this emoji make the content less clear?" If no, remove it.

**Good:**
```markdown
## Quick Start Paths

### I'm New to TurtleWave
**Perfect if:** You want to learn by doing
```

**Avoid:**
```markdown
## Quick Start Paths 🚀

### 📚 I'm New to TurtleWave 🎓
**Perfect if:** You want to learn by doing 💪
```

## Core Principles

### 0. Absolute Factual Accuracy

**Documentation must be 100% factual. Never sacrifice accuracy for marketing appeal.**

#### The Principle

While it's natural to want documentation that makes the package attractive and exciting, **credibility comes from accuracy, not exaggeration**. Users trust documentation that tells the truth, even when that truth is modest.

#### What This Means

❌ **Never do this:**
- Make up usage statistics ("Used by 1000+ researchers")
- Claim performance without benchmarks ("10x faster than alternatives")
- Reference non-existent publications ("Published in leading journals")
- Exaggerate capabilities ("Handles unlimited channels")
- Make baseless comparisons ("Best tool for sleep analysis")
- Use vague superlatives without evidence ("Industry-leading", "Revolutionary")

✅ **Do this instead:**
- Describe actual capabilities: "Handles high-density arrays with 256+ channels"
- Use conditional language: "Designed for large-scale studies"
- Provide verifiable examples: "See examples/ directory for real use cases"
- Let working code demonstrate value
- Be specific about what's tested: "Tested with datasets up to 512 channels"
- Acknowledge limitations honestly

#### Why Accuracy Matters More Than Hype

**Trust is earned through honesty:**
- Researchers need reliable tools, not marketing promises
- One false claim destroys credibility for everything else
- Honest limitations help users make informed decisions
- Accurate documentation prevents wasted time and frustration

**The package sells itself:**
- Working examples are more convincing than claims
- Clear, accurate documentation shows professionalism
- Honest limitations demonstrate understanding of the domain
- Users appreciate transparency

#### Making It Attractive Without Exaggeration

You can make documentation engaging and appealing while staying factual:

**Instead of:** "Revolutionary algorithm processes data 100x faster"
**Write:** "Optimized for high-density arrays. Process a 256-channel overnight recording in under 10 minutes on standard hardware."

**Instead of:** "Trusted by thousands of sleep researchers worldwide"
**Write:** "Actively developed and maintained. See our GitHub for community discussions and contributions."

**Instead of:** "The most accurate sleep event detector available"
**Write:** "Uses validated algorithms adapted from Massimini et al. (2004). See explanation/ for algorithm details."

#### The Rule

**If you don't have evidence, don't claim it.**

When in doubt:
- Describe what the code actually does
- Show working examples
- Link to source code or papers
- Use "can" instead of "will"
- Be specific rather than superlative

### 1. Anticipate User Needs

**Before they ask:**
- Address common questions proactively
- Provide context for why something matters
- Explain the "why" before the "how"

**Example:**
```markdown
!!! tip "Why this matters"
    Detecting slow waves during N3 sleep is crucial because they're markers of 
    sleep quality and synaptic homeostasis. Getting this right can make or break 
    your research findings.
```

### 2. Create Flow and Rhythm

**Use varied content types:**
- Short paragraphs (2-3 sentences max)
- Code examples with clear outputs
- Visual breaks with admonitions
- Lists for scannable content
- Tables for comparisons

**Example structure:**
```markdown
Brief introduction paragraph.

!!! example "Quick Example"
    ```python
    # Show, don't just tell
    result = processor.detect()
    ```

Key points in a list:
- Point one
- Point two

!!! tip
    Helpful insight that adds value
```

### 3. Strategic Use of Admonitions

**Purpose-driven placement:**

**`!!! note`** - Provide context or background
```markdown
!!! note "Understanding the algorithm"
    TurtleWave uses a modified Massimini algorithm that adapts to 
    individual subject characteristics.
```

**`!!! tip`** - Share insider knowledge, best practices
```markdown
!!! tip "Pro tip"
    Processing overnight? Use `batch_mode=True` to save memory and 
    get email notifications when complete.
```

**`!!! warning`** - Prevent common mistakes
```markdown
!!! warning "Common pitfall"
    Don't run detection before generating annotations—you'll get 
    zero results and waste hours debugging.
```

**`!!! example`** - Show real-world usage
```markdown
!!! example "Real-world scenario"
    When analyzing 256-channel data from 20 subjects, this approach 
    reduced processing time from 3 days to 8 hours.
```

**`!!! success`** - Celebrate achievements, confirm success
```markdown
!!! success "You did it!"
    You've successfully detected your first slow waves. The results 
    are saved and ready for analysis.
```

**`!!! question`** - Address likely questions
```markdown
!!! question "What if I have fewer than 64 channels?"
    No problem! TurtleWave adapts automatically. Just ensure your 
    channel layout is properly defined.
```

### 4. Write for Humans, Not Machines

**Bad:**
```markdown
Execute the command to initialize the processor instance.
```

**Good:**
```markdown
Let's create a processor to handle your data:

```python
processor = SWProcessor(eeg_file='your_data.edf')
```

This sets up everything you need to start detecting slow waves.
```

### 5. Progressive Disclosure

**Start simple, add complexity gradually:**

```markdown
## Basic Usage

The simplest way to detect slow waves:

```python
processor.detect_slow_waves()
```

!!! tip "Want more control?"
    You can customize detection parameters—we'll cover that next.

## Advanced Configuration

Now that you've got the basics, let's fine-tune the detection...
```

### 6. Provide Context and Consequences

**Always explain:**
- Why this step matters
- What happens if you skip it
- What success looks like

```markdown
!!! warning "Why annotations matter"
    Running detection without annotations is like searching for a needle 
    without knowing which haystack. You'll waste computational resources 
    and get unreliable results.
```

### 7. Make Examples Realistic

**Bad:**
```python
# Generic example
data = load_data()
result = process(data)
```

**Good:**
```python
# Real scenario: Processing overnight sleep study
processor = SWProcessor(
    eeg_file='subject_001_night1.edf',
    output_dir='results/subject_001/'
)

# Detect slow waves during N3 sleep only
processor.set_sleep_stages(['N3'])
results = processor.detect_slow_waves()

print(f"Found {len(results)} slow waves during deep sleep")
# Output: Found 847 slow waves during deep sleep
```

### 8. Create Emotional Connection

**Use language that:**
- Acknowledges user feelings ("This can be frustrating...")
- Celebrates progress ("Great! You're halfway there...")
- Shows empathy ("We've all been there...")
- Builds confidence ("You've got this...")

```markdown
!!! tip "Feeling overwhelmed?"
    Start with the tutorial. It walks you through everything step-by-step, 
    and you'll have working results in 15 minutes. You've got this!
```

## Admonition Strategy by Document Type

### Tutorials (Learning-oriented)

**Use frequently:**
- `!!! note` - Explain what's happening
- `!!! tip` - Point out what to notice
- `!!! success` - Celebrate milestones

**Use sparingly:**
- `!!! warning` - Only for critical safety issues
- `!!! example` - The whole tutorial IS an example

**Example:**
```markdown
!!! note "What just happened?"
    The processor analyzed each channel independently, then combined 
    results to identify slow waves that appear across multiple regions.

!!! success "Checkpoint reached!"
    You've completed the detection phase. Take a moment to review the 
    statistics before moving on.
```

### How-to Guides (Goal-oriented)

**Use frequently:**
- `!!! tip` - Share best practices
- `!!! warning` - Prevent common mistakes
- `!!! example` - Show real scenarios

**Use sparingly:**
- `!!! note` - Only for essential context
- `!!! success` - Only at completion

**Example:**
```markdown
!!! warning "Before you start"
    Make sure you have at least 50GB free disk space. Processing 
    high-density EEG generates large temporary files.

!!! tip "Speed optimization"
    If you're processing multiple files, use `parallel=True` to 
    leverage all CPU cores. This can reduce processing time by 70%.
```

### Reference (Information-oriented)

**Use sparingly:**
- `!!! note` - For important technical details
- `!!! warning` - For critical constraints
- `!!! example` - For clarifying complex parameters

**Avoid:**
- `!!! tip` - Reference should be neutral
- `!!! success` - Not appropriate for reference

**Example:**
```markdown
!!! warning "Thread safety"
    This class is not thread-safe. Create separate instances for 
    concurrent processing.

!!! note "Performance characteristics"
    Time complexity: O(n*m) where n=samples, m=channels
    Memory usage: ~2GB per 100 channels for 8-hour recording
```

### Explanation (Understanding-oriented)

**Use frequently:**
- `!!! note` - Provide historical context
- `!!! example` - Illustrate concepts
- `!!! question` - Address common wonderings

**Use sparingly:**
- `!!! tip` - Only for conceptual insights
- `!!! warning` - Only for conceptual pitfalls

**Example:**
```markdown
!!! note "Historical context"
    The slow wave detection algorithm evolved from Massimini's 2004 
    work on sleep homeostasis, adapted for high-density recordings.

!!! question "Why not use simple amplitude thresholding?"
    Great question! Simple thresholding fails because slow wave 
    amplitude varies by brain region, sleep depth, and individual 
    characteristics. Our adaptive approach handles this variability.
```

## Writing Checklist

Before publishing any documentation page, verify:

- [ ] **Opening hook** - First paragraph answers "why should I read this?"
- [ ] **Clear structure** - Headings create logical flow
- [ ] **Visual rhythm** - Mix of text, code, admonitions, lists
- [ ] **Strategic admonitions** - Each one adds value, not just decoration
- [ ] **Real examples** - Code shows actual use cases, not toy examples
- [ ] **User empathy** - Language acknowledges user experience
- [ ] **Progressive disclosure** - Simple first, complexity later
- [ ] **Clear outcomes** - User knows what they'll achieve
- [ ] **Next steps** - Clear path forward at the end

## Anti-Patterns to Avoid

### ❌ Admonition Overload
```markdown
!!! note
    This is a note.

!!! tip
    This is a tip.

!!! warning
    This is a warning.

!!! example
    This is an example.
```
**Problem:** Too many admonitions create visual noise and dilute impact.

### ❌ Vague Warnings
```markdown
!!! warning
    Be careful with this parameter.
```
**Problem:** Doesn't explain what could go wrong or how to avoid it.

### ❌ Redundant Admonitions
```markdown
## Installation

!!! note "Installation"
    To install, use pip install...
```
**Problem:** Admonition repeats heading—adds no value.

### ❌ Missing Context
```markdown
```python
processor.set_threshold(2.5)
```
```
**Problem:** No explanation of what 2.5 means or why you'd choose it.

## Examples of Excellence

### Great Tutorial Opening
```markdown
# Getting Started with TurtleWave

In this tutorial, you'll analyze your first sleep EEG recording and detect 
slow waves. By the end, you'll have working results and understand the 
complete workflow.

!!! tip "New to sleep EEG analysis?"
    Perfect! This tutorial assumes no prior experience. We'll explain 
    everything as we go.

## What You'll Learn

In the next 15 minutes, you will:
1. Load a sleep recording
2. Generate sleep stage annotations
3. Detect slow waves
4. Understand your results

Let's dive in!
```

### Great How-to Guide
```markdown
# How to Optimize Detection for High-Density Arrays

This guide shows you how to configure TurtleWave for optimal performance 
with 128+ channel arrays.

!!! warning "Prerequisites"
    Before starting, ensure you have:
    - At least 16GB RAM
    - Sleep stage annotations already generated
    - Channel locations properly defined

## The Challenge

High-density arrays create unique challenges:
- Massive data volumes (>50GB per subject)
- Spatial redundancy between nearby channels
- Longer processing times

Here's how to handle each one...
```

## Remember

> "Documentation is a love letter that you write to your future self."
> — Damian Conway

Every page should feel like you're sitting next to the user, helping them 
succeed. Use admonitions strategically to guide, warn, encourage, and 
celebrate—making the documentation not just useful, but genuinely enjoyable.