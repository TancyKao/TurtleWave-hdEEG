# TurtleWave Documentation

This directory contains the complete documentation for TurtleWave hdEEG, organized using the [Diátaxis framework](DIATAXIS_FRAMEWORK.md).

## Documentation Structure

```
docs/
├── index.md                          # Documentation home page
├── DIATAXIS_FRAMEWORK.md            # Documentation framework guide
├── README.md                        # This file
│
├── tutorials/                       # Learning-oriented lessons
│   ├── getting-started.md          # First steps with TurtleWave
│   └── eeg-review-gui-tutorial.md  # Complete GUI walkthrough
│
├── how-to/                          # Problem-solving guides
│   ├── installation.md             # Setup instructions
│   ├── detect-slow-waves.md        # Slow wave detection
│   └── review-eeg-events.md        # Event review workflows
│
├── reference/                       # Technical specifications
│   ├── eeg-review-gui.md           # GUI reference
│   └── api/                        # API documentation
│       ├── index.md
│       ├── eventprocessor.md
│       ├── swprocessor.md
│       ├── pacprocessor.md
│       └── gui.md
│
├── explanation/                     # Understanding-oriented
│   ├── overview.md                 # System architecture
│   └── eeg-review-gui-architecture.md  # GUI design
│
├── images/                          # Screenshots and diagrams
│   ├── gui-setup-tab.png
│   ├── gui-data-loaded.png
│   ├── gui-slow-wave-detection.png
│   ├── gui-spindle-detection.png
│   └── gui-pac-analysis.png
│
└── stylesheets/                     # Custom CSS
    └── extra.css
```

## Diátaxis Framework

Our documentation follows the Diátaxis framework, which organizes content into four distinct types:

### 📚 Tutorials (Learning-Oriented)

**Purpose:** Teach through hands-on lessons

**Characteristics:**
- Step-by-step instructions
- Learning by doing
- Guaranteed to work
- Builds confidence

**When to use:** You're new to TurtleWave and want to learn the basics

**Example:** [EEG Review GUI Tutorial](tutorials/eeg-review-gui-tutorial.md)

### 🔧 How-to Guides (Goal-Oriented)

**Purpose:** Solve specific problems

**Characteristics:**
- Practical steps
- Assumes basic knowledge
- Focused on results
- Multiple approaches possible

**When to use:** You have a specific task to accomplish

**Example:** [Review EEG Events](how-to/review-eeg-events.md)

### 📖 Reference (Information-Oriented)

**Purpose:** Provide technical specifications

**Characteristics:**
- Comprehensive details
- Accurate and up-to-date
- Structured for lookup
- Dry and factual

**When to use:** You need to look up specific details

**Example:** [EEG Review GUI Reference](reference/eeg-review-gui.md)

### 💡 Explanation (Understanding-Oriented)

**Purpose:** Clarify concepts and design decisions

**Characteristics:**
- Discusses alternatives
- Explains "why"
- Provides context
- Deepens understanding

**When to use:** You want to understand how/why things work

**Example:** [EEG Review GUI Architecture](explanation/eeg-review-gui-architecture.md)

## EEG Review GUI Documentation

Complete documentation for the EEG Review GUI is now available:

### Tutorial
[**Your First EEG Event Review Session**](tutorials/eeg-review-gui-tutorial.md)

A hands-on walkthrough that teaches you to:
- Launch the GUI and load data
- Navigate through events
- Accept/reject events
- Filter and customize the display
- Export results

**Time:** ~30 minutes  
**Prerequisites:** TurtleWave installed, sample data

### How-to Guide
[**Review EEG Events**](how-to/review-eeg-events.md)

Practical solutions for specific tasks:
- Filter events by multiple criteria
- Review only high-confidence events
- Resume a previous session
- Adjust display for different event types
- Export only accepted events
- Troubleshoot common issues

**Format:** Problem → Solution  
**Prerequisites:** Basic GUI familiarity

### Reference
[**EEG Review GUI Reference**](reference/eeg-review-gui.md)

Complete technical specifications:
- Component API documentation
- Keyboard shortcuts
- Menu and toolbar reference
- Filter options
- Database schema
- Performance characteristics

**Format:** Structured lookup  
**Prerequisites:** None

### Explanation
[**EEG Review GUI Architecture**](explanation/eeg-review-gui-architecture.md)

Design principles and technical decisions:
- Why a dedicated review GUI?
- Architecture overview
- Component design rationale
- Performance optimizations
- Design trade-offs
- Future directions

**Format:** Conceptual discussion  
**Prerequisites:** Programming knowledge helpful

## Building the Documentation

### Local Preview

```bash
# Install MkDocs and dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Build Static Site

```bash
mkdocs build
```

Output will be in `site/` directory.

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

## Documentation Guidelines

### Writing Style

- **Be clear and concise** - Avoid jargon when possible
- **Use active voice** - "Click the button" not "The button should be clicked"
- **Show, don't just tell** - Include examples and screenshots
- **Test your instructions** - Ensure steps actually work

### Markdown Formatting

Follow MkDocs Material requirements:

**Blank lines required:**
- Before and after lists
- Before and after code blocks
- Before and after admonitions
- After headings

**Code blocks:**
```python
# Always specify language
def example():
    pass
```

**Admonitions:**
```markdown
!!! note
    Use admonitions for important information.

!!! tip
    Helpful suggestions go here.

!!! warning
    Important warnings here.
```

### Cross-References

Link between documentation types:

```markdown
See the [Tutorial](../tutorials/eeg-review-gui-tutorial.md) to learn the basics.
Check the [Reference](../reference/eeg-review-gui.md) for technical details.
Read the [Explanation](../explanation/eeg-review-gui-architecture.md) to understand why.
```

### Images

Store images in `docs/images/`:

```markdown
![GUI Screenshot](../images/gui-data-loaded.png)
```

Include alt text for accessibility.

## Contributing to Documentation

### Adding New Content

1. **Determine the type** - Tutorial, How-to, Reference, or Explanation?
2. **Follow the template** - Use existing docs as examples
3. **Test your content** - Ensure code examples work
4. **Update navigation** - Add to `mkdocs.yml`
5. **Cross-reference** - Link to related docs

### Updating Existing Content

1. **Maintain the type** - Don't mix tutorial content into reference
2. **Preserve structure** - Keep consistent formatting
3. **Update cross-references** - Check for broken links
4. **Test changes** - Preview with `mkdocs serve`

### Review Checklist

- [ ] Content matches its Diátaxis type
- [ ] Blank lines before/after lists, code blocks, admonitions
- [ ] Code blocks specify language
- [ ] Links work correctly
- [ ] Images display properly
- [ ] Spelling and grammar checked
- [ ] Examples tested and working

## Questions?

- **Documentation issues:** [GitHub Issues](https://github.com/your-repo/turtlewave-hdEEG/issues)
- **Framework questions:** See [DIATAXIS_FRAMEWORK.md](DIATAXIS_FRAMEWORK.md)
- **Style guide:** See [.roo/context/documentation-style-guide.md](../.roo/context/documentation-style-guide.md)

## Resources

- [Diátaxis Framework](https://diataxis.fr/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocstrings](https://mkdocstrings.github.io/)
