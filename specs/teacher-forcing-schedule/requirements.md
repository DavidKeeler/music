# Requirements

This document captures the requirements clarification process through Q&A.

---

## Q1: Schedule Type

What type of teacher forcing decay schedule do you want to implement?

Options:
- **Linear decay**: Ratio decreases at a constant rate (e.g., 1.0 → 0.0 over N steps)
- **Exponential decay**: Ratio decreases exponentially (e.g., ratio = initial * exp(-k * step))
- **Step decay**: Ratio drops at specific milestones (e.g., 1.0 → 0.5 → 0.2 → 0.05)
- **Inverse sigmoid**: Smooth S-curve transition
- **Configurable/multiple options**: Support multiple schedules with config selection

**A1:** One option, but let's do research to figure out which schedule is most effective.

