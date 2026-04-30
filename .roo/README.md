# Roo Configuration for TurtleWave-hdEEG

This directory contains Roo-specific configuration files that enhance the AI assistant's behavior when working on this project.

## Structure

### `/rules/`
Contains rule files that define when and how to apply specific behaviors:
- `documentation-framework.md` - Auto-loads Diátaxis framework and MkDocs Material requirements for documentation work

### `/context/`
Contains context files that provide project-specific information:
- `documentation.md` - Documentation structure, framework reference, and quality checklist
- `mkdocs-material.md` - Complete MkDocs Material markdown syntax guide

## How It Works

When you work on documentation in this project, Roo will automatically:

1. **Load Diátaxis Framework** (`docs/DIATAXIS_FRAMEWORK.md`)
   - Classify documentation type (Tutorial, How-to, Reference, Explanation)
   - Apply appropriate writing principles
   - Use correct language patterns

2. **Apply MkDocs Material Requirements**
   - Enforce blank line spacing rules
   - Ensure code blocks have language tags
   - Use proper admonition syntax
   - Follow Material theme best practices

3. **Maintain Quality Standards**
   - Check Diátaxis compliance
   - Verify MkDocs syntax
   - Ensure consistent formatting

## Documentation Framework

This project uses the **Diátaxis framework** for all documentation:

| Type | Orientation | User State | Focus |
|------|-------------|------------|-------|
| **Tutorials** | Learning | Study | Action (doing) |
| **How-to Guides** | Goals | Work | Action (doing) |
| **Reference** | Information | Work | Cognition (knowing) |
| **Explanation** | Understanding | Study | Cognition (knowing) |

Complete framework guide: `docs/DIATAXIS_FRAMEWORK.md`

## MkDocs Material Requirements

### Critical Spacing Rules

**ALWAYS include blank lines before:**
- Lists (ordered and unordered)
- Code blocks
- Admonitions
- Headings (after previous content)

**ALWAYS include blank lines after:**
- Headings (before content)
- Code blocks
- Admonitions

**ALWAYS specify language for code blocks:**
```markdown
```python
# Correct - language specified
```
```

### Common Admonitions
```markdown
!!! note "Title"
    Content here.

!!! tip
    Helpful suggestion.

!!! warning
    Important warning.

!!! example
    Code example.
```

Full syntax guide: `.roo/context/mkdocs-material.md`

## Usage

Simply work on documentation as normal. When you:
- Edit files in `docs/` directory
- Create or modify `.md` files
- Mention "documentation", "tutorial", "how-to", etc.

Roo will automatically:
- Reference the Diátaxis framework
- Apply MkDocs Material syntax rules
- Provide appropriate guidance

## Quality Checklist

Before committing documentation, verify:

**Diátaxis:**
- [ ] Correct type identified (Tutorial/How-to/Reference/Explanation)
- [ ] Appropriate principles applied
- [ ] No boundary blurring

**MkDocs Material:**
- [ ] Blank lines before lists
- [ ] Blank lines after headings
- [ ] Blank lines around code blocks
- [ ] Language tags on code blocks
- [ ] Proper admonition syntax

**General:**
- [ ] Clear, concise writing
- [ ] Links tested
- [ ] Images have alt text
- [ ] Consistent terminology

## File Organization

```
docs/
├── index.md              # Home page
├── tutorials/            # Learning-oriented lessons
├── how-to/              # Goal-oriented guides
├── reference/           # API documentation
└── explanation/         # Conceptual guides
```

## Customization

You can add more rules or context files to this directory to further customize Roo's behavior for your project needs.

## References

- **Diátaxis Framework**: https://diataxis.fr/
- **MkDocs**: https://www.mkdocs.org/
- **Material for MkDocs**: https://squidfunk.github.io/mkdocs-material/