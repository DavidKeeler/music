# Rough Idea

We're going to create a reusable debugging context to use with the specs. They keep creating code that doesn't run or has memory problems.

## Problem Statement

Specs are generating code with:
- Runtime errors (code that doesn't execute)
- Memory problems (leaks, excessive allocation, inefficient usage)

## Goal

Create a reusable debugging context that can be included in specs to prevent these issues and guide better code generation.
