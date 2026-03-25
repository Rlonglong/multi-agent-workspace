# Copilot Instructions for Multi-Agent Workspace

Guidelines for all interactions in this workspace.

## Output Formatting - CRITICAL

Remove ALL unnecessary blank lines:
- No blank line immediately after headers
- No blank lines between consecutive list items
- No consecutive blank lines anywhere
- One blank line maximum between distinct major sections
- Never use `---` as visual separator (only in frontmatter)

## Thinking Process - All Modes

Always show thinking in collapsible sections:
```html
<details>
<summary>💭 Thinking</summary>

Your reasoning...

</details>
```

Agent Mode: Thinking auto-collapses during streaming. User can expand.

## Markdown Structure

- Headers: content starts immediately on next line
- Lists: items use `-` or `1.` with no blanks between
- Code: inline or blocks without preceding blank lines
- Links: `[text](file.ts#L10)` format, never with backticks
- Separators: `---` only in YAML frontmatter

## Writing Style

Compact, direct, structured—use markdown hierarchy not whitespace.
