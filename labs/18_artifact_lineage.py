"""Trace the committed Base-to-GRPO evidence without loading a model."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Read the compact M7 evidence and verify the headline lineage contract."""

    path = Path("docs/reference/evidence/m7-reference-pipeline-92564f3.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evaluation = evidence["evaluation"]
    pipeline = evidence["pipeline"]

    print(f"Status:             {evidence['status']}")
    print(f"Stages:             {pipeline['stages']}")
    print(f"Failed attempts:    {pipeline['failed_attempts']}")
    print(f"Base accuracy:      {evaluation['base']['accuracy']:.3f}")
    print(f"Final GRPO accuracy:{evaluation['grpo']['accuracy']:>6.3f}")
    assert evidence["status"] == "m7_reference_pipeline_passed"
    assert pipeline["stages"] == 15
    assert pipeline["failed_attempts"] == 0
    assert evaluation["grpo"]["accuracy"] > evaluation["base"]["accuracy"]
    print("Artifact-lineage lab passed.")


if __name__ == "__main__":
    main()
