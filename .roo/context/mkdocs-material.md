# MkDocs Material Markdown Guidelines

## Critical Spacing Rules

### ALWAYS Include Blank Lines Before:
- Lists (ordered and unordered)
- Code blocks
- Headings (after previous content)
- Admonitions
- Tables
- Block quotes

### ALWAYS Include Blank Lines After:
- Headings (before content)
- Code blocks
- Admonitions
- Tables
- Block quotes

## Markdown Syntax Reference

### Headings
```markdown
# H1 Heading

Content here.

## H2 Heading

Content here.

### H3 Heading

Content here.
```

### Lists

**Unordered Lists:**
```markdown
Previous paragraph.

- Item 1
- Item 2
- Item 3

Next paragraph.
```

**Ordered Lists:**
```markdown
Previous paragraph.

1. First item
2. Second item
3. Third item

Next paragraph.
```

**Nested Lists:**
```markdown
Previous paragraph.

- Parent item
    - Child item (4 spaces)
    - Another child
- Another parent

Next paragraph.
```

### Code Blocks

**Inline code:** Use single backticks: `code here`

**Fenced code blocks:**
```markdown
Previous paragraph.

```python
def example():
    return "Always specify language"
```

Next paragraph.
```

**Common languages:**
- `python` - Python code
- `bash` or `shell` - Shell commands
- `yaml` - YAML configuration
- `json` - JSON data
- `markdown` - Markdown examples
- `text` - Plain text
- `console` - Console output

### Admonitions (Material Theme)

```markdown
Previous paragraph.

!!! note "Optional Title"
    Content of the admonition.
    
    Can have multiple paragraphs.

Next paragraph.
```

**Available types:**
- `note` - Blue, general information
- `abstract` or `summary` - Light blue, summaries
- `info` or `todo` - Cyan, information
- `tip` or `hint` - Green, helpful tips
- `success` or `check` - Green, success messages
- `question` or `help` - Light green, questions
- `warning` or `caution` - Orange, warnings
- `failure` or `fail` - Red, failures
- `danger` or `error` - Red, dangers
- `bug` - Red, bugs
- `example` - Purple, examples
- `quote` or `cite` - Gray, quotations

**Collapsible admonitions:**
```markdown
??? note "Click to expand"
    Hidden content here.
```

### Links

**Inline links:**
```markdown
[Link text](https://example.com)
[Internal link](../path/to/file.md)
[Anchor link](#section-heading)
```

**Reference-style links:**
```markdown
This is [a link][ref] and [another link][ref2].

[ref]: https://example.com
[ref2]: https://example.org
```

### Images

```markdown
![Alt text](path/to/image.png)
![Alt text](path/to/image.png "Optional title")
```

### Tables

```markdown
Previous paragraph.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

Next paragraph.
```

**Alignment:**
```markdown
| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |
```

### Block Quotes

```markdown
Previous paragraph.

> This is a quote.
> It can span multiple lines.

Next paragraph.
```

### Horizontal Rules

```markdown
Previous paragraph.

---

Next paragraph.
```

## Common Mistakes to Avoid

❌ **Missing blank line before list:**
```markdown
This is text.
- List item  <!-- WRONG -->
```

✅ **Correct:**
```markdown
This is text.

- List item  <!-- CORRECT -->
```

❌ **Missing blank line after heading:**
```markdown
## Heading
Content starts here.  <!-- WRONG -->
```

✅ **Correct:**
```markdown
## Heading

Content starts here.  <!-- CORRECT -->
```

❌ **Missing blank line before code block:**
```markdown
Some text.
```python  <!-- WRONG -->
code
```
```

✅ **Correct:**
```markdown
Some text.

```python  <!-- CORRECT -->
code
```
```

❌ **No language specified:**
```markdown
```  <!-- WRONG -->
code
```
```

✅ **Correct:**
```markdown
```python  <!-- CORRECT -->
code
```
```

## Material Theme Features

### Content Tabs

```markdown
=== "Tab 1"
    Content for tab 1.

=== "Tab 2"
    Content for tab 2.
```

### Task Lists

```markdown
- [x] Completed task
- [ ] Incomplete task
- [ ] Another task
```

### Keyboard Keys

```markdown
Press ++ctrl+alt+delete++ to restart.
```

### Icons and Emojis

```markdown
:material-account-circle: Material icon
:smile: Emoji
```

## Best Practices

1. **Always preview** - Check rendering in MkDocs before committing
2. **Consistent spacing** - Follow spacing rules religiously
3. **Language tags** - Always specify language for code blocks
4. **Descriptive alt text** - Provide meaningful alt text for images
5. **Semantic headings** - Use heading hierarchy properly (H1 → H2 → H3)
6. **Link validation** - Ensure all links work
7. **Admonition types** - Use appropriate admonition types for context

## Quick Checklist

Before committing markdown documentation:

- [ ] Blank line before all lists
- [ ] Blank line after all headings
- [ ] Blank line before/after code blocks
- [ ] Language specified for all code blocks
- [ ] Blank line before/after admonitions
- [ ] All links tested
- [ ] Images have alt text
- [ ] Proper heading hierarchy
- [ ] No trailing whitespace
- [ ] File ends with newline