# Documentation Framework Rules

## Auto-load Diátaxis Framework

When working on documentation tasks (creating, editing, or reviewing documentation files), automatically load the Diátaxis framework knowledge base as context.

### Trigger Conditions
Load `docs/DIATAXIS_FRAMEWORK.md` when:
- Working with files in `docs/` directory
- Creating or editing `.rst`, `.md` files related to documentation
- User explicitly mentions "documentation", "tutorial", "how-to", "reference", or "explanation"
- Working on README files
- User asks to "build documentation" or "write docs"

### Context Loading
Always reference these documents when working on documentation:

1. **`docs/DIATAXIS_FRAMEWORK.md`** - Framework principles:
   - Determine which type of documentation to create (Tutorial, How-to, Reference, Explanation)
   - Apply appropriate writing principles for each type
   - Use correct language patterns
   - Avoid common pitfalls (especially Tutorial/How-to confusion)

2. **`.roo/context/documentation-style-guide.md`** - Writing excellence:
   - Create engaging, user-friendly content
   - Use admonitions strategically
   - Write with empathy and flow
   - Make documentation a pleasure to read
   - Ensure users want to come back for more

### MkDocs Material Requirements
When creating markdown documentation, ALWAYS follow these MkDocs Material-specific rules:

#### Spacing Requirements
1. **Blank line before lists** - REQUIRED
   ```markdown
   This is a paragraph.
   
   - List item 1
   - List item 2
   ```

2. **Blank line after headings** - REQUIRED
   ```markdown
   ## Heading
   
   Content starts here.
   ```

3. **Blank line before/after code blocks** - REQUIRED
   ```markdown
   Some text.
   
   ```python
   code here
   ```
   
   More text.
   ```

4. **Blank line before/after admonitions** - REQUIRED
   ```markdown
   Regular text.
   
   !!! note
       Admonition content.
   
   More text.
   ```

#### List Formatting
- Use consistent indentation (4 spaces for nested items)
- Blank line between list items with multiple paragraphs
- No blank lines between simple list items

#### Code Blocks
- Always specify language for syntax highlighting
- Use fenced code blocks (```) not indented blocks
- Common languages: `python`, `bash`, `yaml`, `json`, `markdown`

#### Admonitions (Material Theme)
Available types:
- `!!! note` - General information
- `!!! tip` - Helpful suggestions
- `!!! warning` - Important warnings
- `!!! danger` - Critical warnings
- `!!! example` - Code examples
- `!!! quote` - Quotations

Collapsible version: `???` instead of `!!!`

#### Links
- Use reference-style for repeated links
- Internal links: `[text](../path/to/file.md)`
- Anchors: `[text](#section-heading)`

### Application
When creating or improving documentation:
1. Use the Diátaxis compass to classify the content type
2. Apply the specific principles for that type
3. Follow MkDocs Material markdown requirements
4. Follow the language patterns provided
5. Maintain clear boundaries between documentation types
6. Work iteratively - one improvement at a time

This ensures consistent, high-quality documentation that serves user needs effectively and renders correctly in MkDocs Material.