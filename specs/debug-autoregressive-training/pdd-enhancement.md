# PDD SOP Enhancement: Smoke Test Requirements

## Issue

The debugging spec for autoregressive training didn't catch a runtime error because acceptance criteria didn't explicitly require running the code as a verification step.

## Root Cause

- Acceptance criteria said "training should start" but didn't mandate actual execution
- Implementation plan had manual verification but not as a hard requirement
- Runtime errors (like type mismatches in `tf.cond`) only appear when code actually runs

## Proposed Enhancement to PDD SOP

### Add to Step 6 (Create Detailed Design) - Acceptance Criteria section:

**For debugging/bug fix specs:**

```
## Acceptance Criteria Requirements for Bug Fixes

All bug fix specs MUST include smoke test execution in acceptance criteria:

✓ DO:
```gherkin
Given the fixed code
When I run "<exact command>"
Then the command must complete without errors
And at least one successful operation must occur
And the specific error "<error message>" must not appear
```

✗ DON'T:
```gherkin
Then the training should start without errors  # Too vague
And the code should work  # Not verifiable
```

**Mandatory elements:**
1. Exact command to run (with minimal parameters for speed)
2. Expected successful completion (exit code 0)
3. Specific error that must NOT occur
4. At least one concrete operation that proves the fix works

**For training/long-running processes:**
- Use minimal epochs (1-2) and small batch sizes
- Require at least one batch/step to complete
- Verify metrics are logged correctly
```

### Add to Step 7 (Develop Implementation Plan):

**For debugging specs, add mandatory smoke test step:**

```
## Step N: Run Smoke Test (MANDATORY for bug fixes)

**Objective:** Verify the fix resolves the bug by running the actual command.

**Implementation:**
- Run: `<exact command from acceptance criteria>`
- Capture output and exit code
- Verify specific error does not occur
- Confirm at least one successful operation

**Test requirements:**
- Command must exit with status 0
- Original error message must not appear in output
- At least one batch/step/operation must complete

**Integration notes:**
- This is a BLOCKING step - if it fails, the fix is incomplete
- Use minimal parameters for speed (small epochs, batch sizes)
- Log full output for debugging if test fails

**Demo:**
```bash
# Must succeed
<command>
echo $?  # Should be 0
```
```

## Summary

**Change:** Make smoke test execution a mandatory, explicit requirement in acceptance criteria and implementation plan for all debugging specs.

**Rationale:** Runtime errors can't be caught by code review alone - actual execution is required.

**Impact:** Prevents incomplete fixes that pass code review but fail at runtime.
