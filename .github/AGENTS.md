# Agent Modes for Multi-Agent Workspace

## Workspace Agent

**Mode**: Agent Mode / Workspace Mode

### Expected Behavior

1. **Thinking Animation Phase**
   - User sends message
   - UI displays "thinking..." with loading animation
   - No content streaming yet

2. **Content Streaming Phase**
   - Thinking section auto-collapses (if expanded)
   - Output begins streaming
   - User can expand thinking anytime to see details

3. **Thinking Section Format**
   ```html
   <details>
   <summary>💭 Thinking</summary>

   Detailed reasoning...

   </details>
   ```

### Output Requirements

- **NO blank lines** between logical elements
- Headers immediately followed by content
- Lists with no spacing between items
- No `---` separators outside YAML
- Clean, compact structure

### When to Use

This is the default workspace agent for:
- Coding tasks
- Debugging
- Analysis
- Multi-step workflows
