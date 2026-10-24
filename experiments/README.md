# Accuracy an inference service must reach

Models should be tuned toward a few **strong classes**.
To be used in the experiment, they should reach the numbers below.

| Requirement | Value | Now |
| --- | --- | --- |
| Accuracy on those classes | > 0.8 | 0.849 to 0.952 |
| Accuracy gain on those classes, over the same model with no tuning | > 0.05 | 0.092 to 0.279 |
| Accuracy lost in the blurred condition | > 0.04 | 0.060 to 0.176 |
| General accuracy | > 0.35 | 0.398 to 0.632 |

Every variation should have 1-2 strong classes.

The pool as a whole should reach the numbers below.

| Requirement | Value | Now |
| --- | --- | --- |
| Variations tuned toward each strong class | > 1 | 2 to 3 |
| Variations that hold the best accuracy on some class | > 8 | 11 of 16 |
| Classes whose best variation changes in the blurred condition | > 5 | 14 of 19 |
| Slowest variation over the fastest, in response time | > 5x | 97x, 10.7 ms to 1035 ms |

Those requirements are meant to create a diverse performance for the experiment and to prove the ability of the orchestration system to select the best model for each class and condition.
