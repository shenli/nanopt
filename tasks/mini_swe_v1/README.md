# MiniSWE v1 original task suite

These tiny Python repositories were written for NanoPT and are licensed under Apache-2.0. Each
task keeps its model-visible `snapshot/`, trusted `hidden_tests/`, and reviewed `oracle.patch`
separate. The loader proves the snapshot hash before every reset and copies only `snapshot/` into
the episode workspace.

Hidden tests are public in the educational source repository so humans can audit them. They are
never copied into model observations or model-visible workspaces during an evaluation episode.
