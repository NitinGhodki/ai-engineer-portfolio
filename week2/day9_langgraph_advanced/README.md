============================================================
CODE REVIEW — Human will: APPROVE
============================================================

[Phase 1] Running analysis and suggestions...
  [Analyse] Scanning code for issues...
  [Analyse] Found 3 issues ✓
  [Suggest] Generating improvement suggestions...
  [Suggest] Suggestions ready ✓

[Paused] Suggestions ready:
  Issues: ['Sure, here are the issues found in the provided code:', '1. **SQL Injection Vulnerability**:', '- The code directly concatenates the `id` variable into the SQL query string, which can lead to SQL injection attacks. An attacker could input malicious SQL code through the `id` parameter.']

[Human] Entering decision: 'approve'

  [Human Approval] Pausing for review...
  [Human Approval] Decision: APPROVE
  [Router] routing to: apply_changes
  [Apply] Applying approved changes...

[Final Review]
CODE REVIEW COMPLETE — APPROVED
Original issues: 3
Changes applied: To address the SQL injection vulnerability, you should use parameterized queries or prepared stateme...
Status: Changes merged successfully


============================================================
PART 2: Crash simulation and resume
============================================================

[PHASE 1] Running until simulated crash point (before draft node)...
  [Step 1] Gathering sources for: 'Checkpointing'
  [Step 1] Found 3 sources ✓
  [Step 2] Summarising 3 sources...
  [Step 2] Summaries created ✓

[CRASH] Graph stopped at: ('draft',)
[CRASH] Work preserved — step: 2
[CRASH] Sources saved: 3 ✓
[CRASH] Summaries saved: 3 ✓

[CRASH] In production: server restarts here.
[CRASH] Without checkpointing: start from step 0 again.
[CRASH] With checkpointing: resume from step 2.

[RESUME] Server restarted. Resuming from checkpoint...
  [Step 3] Writing draft...
  [Step 3] Draft written ✓
  [Step 4] Finalising report...
  [Step 4] Report finalised ✓

[RESUME] Complete! Final step: 4
[RESUME] Final report preview: FINAL REPORT
========================================
DRAFT on Checkpointing:
Summary of: Source A: ...



difference between interrupt_before and interrupt_after?

Both are static break points used for pause graph execution.

interrupt_before: it pause before a specified noderuns, allowing insoection or modification of input data.
interrupt_after: Pauses after a specified node completes, allowing inspection or modification of output data