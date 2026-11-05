# Smoke evidence

`verified/` is the October 19 historical smoke report. Its quantitative coverage, force, and success metrics remain valid, but contact-loss/recovery flags used an incorrect useful-force threshold. Those flags and the corresponding video label must not be interpreted as actual loss of physical contact.

The corrected completed November 6 pipeline is `artifacts/smoke/final-2026-11-06-complete/`; `final/` contains its committed report. Its evaluator requires an entire interval below 1e-4 N maximum contact force before recording contact loss. No requested example is fabricated when absent. The similarly named directory without `-complete` is a preserved partial run interrupted by the tool timeout.

`media-verification.json` verifies decoding and hashes of the historical videos, not correctness of their now-withdrawn labels. `reproducibility.json` records exact equality of two independently generated smoke datasets and BC initialization tensors. `lock-verification.json` records the exact locked package versions.
