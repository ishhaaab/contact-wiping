# Measured evaluation results

All attempted episodes are included. Smoke results only verify execution.

Evaluation split: `validation_smoke_subset`

| Profile | Method | Suite | Train seeds | Episodes | Coverage | Success | Unsafe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke | bc | curvature | 1 | 3 | 0.0326 | 0.0000 | 0.9059 |
| smoke | bc_ppo | curvature | 1 | 3 | 0.0964 | 0.0000 | 0.1561 |
| smoke | ppo | curvature | 1 | 3 | 0.0000 | 0.0000 | 0.0000 |
| smoke | scripted | curvature | 0 | 3 | 0.6823 | 0.0000 | 0.0000 |
| smoke | bc | friction | 1 | 3 | 0.0430 | 0.0000 | 0.9108 |
| smoke | bc_ppo | friction | 1 | 3 | 0.0690 | 0.0000 | 0.3133 |
| smoke | ppo | friction | 1 | 3 | 0.0000 | 0.0000 | 0.0000 |
| smoke | scripted | friction | 0 | 3 | 0.6706 | 0.0000 | 0.0000 |
| smoke | bc | id | 1 | 3 | 0.0534 | 0.0000 | 0.6463 |
| smoke | bc_ppo | id | 1 | 3 | 0.1029 | 0.0000 | 0.6597 |
| smoke | ppo | id | 1 | 3 | 0.0000 | 0.0000 | 0.0000 |
| smoke | scripted | id | 0 | 3 | 0.6693 | 0.0000 | 0.0000 |

## Per-training-seed results

| Method | Suite | Training seed | Episodes | Coverage | Success |
| --- | --- | --- | ---: | ---: | ---: |
| bc | curvature | 11 | 3 | 0.0326 | 0.0000 |
| bc_ppo | curvature | 11 | 3 | 0.0964 | 0.0000 |
| ppo | curvature | 11 | 3 | 0.0000 | 0.0000 |
| scripted | curvature | baseline | 3 | 0.6823 | 0.0000 |
| bc | friction | 11 | 3 | 0.0430 | 0.0000 |
| bc_ppo | friction | 11 | 3 | 0.0690 | 0.0000 |
| ppo | friction | 11 | 3 | 0.0000 | 0.0000 |
| scripted | friction | baseline | 3 | 0.6706 | 0.0000 |
| bc | id | 11 | 3 | 0.0534 | 0.0000 |
| bc_ppo | id | 11 | 3 | 0.1029 | 0.0000 |
| ppo | id | 11 | 3 | 0.0000 | 0.0000 |
| scripted | id | baseline | 3 | 0.6693 | 0.0000 |

All episodes, including failures, contribute to coverage and success.
Force means include non-contact substeps.
Completion time is conditional on success; null means no successes.
Across-seed SD is sample SD; null for fewer than two models.

`summary.json` includes force, elapsed time, successful completion time, and sample standard deviation across training seeds. Null SD means it cannot be estimated from the available models.
