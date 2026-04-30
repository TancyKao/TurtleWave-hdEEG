# Documentation Context

## Primary Framework Reference
- **File**: `docs/DIATAXIS_FRAMEWORK.md`
- **Purpose**: Comprehensive guide for all documentation work
- **Usage**: Reference this file for any documentation-related tasks

## MkDocs Material Reference
- **File**: `.roo/context/mkdocs-material.md`
- **Purpose**: MkDocs Material markdown syntax and requirements
- **Usage**: Follow these rules for all markdown documentation

## Documentation Structure
This project uses the Diátaxis framework with four documentation types:

1. **Tutorials** - Learning-oriented lessons
2. **How-to Guides** - Goal-oriented directions
3. **Reference** - Technical descriptions (API docs)
4. **Explanation** - Understanding-oriented discussion

## Documentation Format
- **Primary format**: Markdown (`.md`)
- **Build system**: MkDocs with Material theme
- **Location**: `docs/` directory

## When Working on Documentation

### Step 1: Classify Content Type
Use the Diátaxis compass:
1. **Action or Cognition?** (doing vs knowing)
2. **Acquisition or Application?** (study vs work)

### Step 2: Apply Framework Principles
Consult `docs/DIATAXIS_FRAMEWORK.md` for:
- Appropriate writing principles
- Correct language patterns
- Quality standards

### Step 3: Follow MkDocs Material Syntax
Consult `.roo/context/mkdocs-material.md` for:
- Required blank lines (before lists, headings, code blocks)
- Code block language specification
- Admonition syntax
- Link formatting

## Critical MkDocs Requirements

**ALWAYS include blank lines before:**
- Lists
- Code blocks
- Admonitions
- Headings (after content)

**ALWAYS include blank lines after:**
- Headings (before content)
- Code blocks
- Admonitions

**ALWAYS specify language for code blocks:**
```python
# Good
```

```
# Bad - no language specified
```

## Quick Decision Tool

| Content Type | User Need | User State | Content Focus |
|--------------|-----------|------------|---------------|
| Tutorial | Learning | Study | Action (doing) |
| How-to | Goals | Work | Action (doing) |
| Reference | Information | Work | Cognition (knowing) |
| Explanation | Understanding | Study | Cognition (knowing) |

## File Organization

```
docs/
├── index.md              # Home page
├── tutorials/            # Tutorial documents
├── how-to/              # How-to guides
├── reference/           # API reference
└── explanation/         # Conceptual guides
```

## Quality Checklist

Before committing documentation:

**Diátaxis Compliance:**
- [ ] Correct documentation type identified
- [ ] Appropriate principles applied
- [ ] Correct language patterns used
- [ ] No boundary blurring

**MkDocs Material Compliance:**
- [ ] Blank line before all lists
- [ ] Blank line after all headings
- [ ] Blank line before/after code blocks
- [ ] Language specified for all code blocks
- [ ] Blank line before/after admonitions
- [ ] Proper heading hierarchy (H1 → H2 → H3)

**General Quality:**
- [ ] Clear, concise writing
- [ ] All links tested
- [ ] Images have alt text
- [ ] No spelling/grammar errors
- [ ] Consistent terminology