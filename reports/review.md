# Final software review

One independent read-only reviewer audited the implementation against the complete shared specification and surface-wiping plan. The reviewer independently ran 46 original tests, Ruff, dependency checks, fixture checksums, and report recomputation.

## Findings and disposition

- Important: simultaneous excessive force could overwrite a MuJoCo warning or out-of-bounds fault. Two regression cases reproduced this before correction. Simulator faults now take priority and fail-fast diagnostics survive.
- Important: ordinary reports could merge validation and test rows. A regression reproduced this before correction. Reports now reject mixed split labels and display the split explicitly.
- Additional author finding: weak force was incorrectly called contact loss. A regression reproduced this before correction. A full unloaded interval is now required. Historical video labels and claims are withdrawn in `results.md`.
- Deferred minor: strengthen the off-axis static-load test to distinguish world-z force within its tolerance. The production code correctly measures contact-frame normal force. No physics change is required.

The reviewer did not judge learning superiority, physical-robot safety, unexecuted Linux support, or visual aesthetics. Those remain unverified and are not implied by software tests. Full experimental performance remains separate from smoke execution.

The fixes use failing regression tests followed by the complete suite. No second reviewer or long training experiment was launched. `DECISIONS.md` contains decisions, risks, and task evidence.
