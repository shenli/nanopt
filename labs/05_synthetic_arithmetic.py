"""Generate, split, fingerprint, and verify a tiny arithmetic dataset on CPU."""

from __future__ import annotations

from pathlib import Path

import yaml

from nanopt.data.arithmetic import (
    ArithmeticGeneratorConfig,
    generate_tasks,
    render_trusted_completion,
)
from nanopt.data.fingerprints import dataset_fingerprint
from nanopt.data.schemas import SplitName
from nanopt.data.splits import build_splits
from nanopt.eval.verifier import verify_task_response


def main() -> None:
    """Run the complete deterministic M2 data path without writing generated artifacts."""

    raw_config = yaml.safe_load(Path("tasks/arithmetic/generator_config.yaml").read_text())
    config = ArithmeticGeneratorConfig.model_validate(raw_config).model_copy(update={"count": 20})
    first = generate_tasks(config)
    second = generate_tasks(config)
    first_fingerprint = dataset_fingerprint(first, generator_config=config)
    second_fingerprint = dataset_fingerprint(second, generator_config=config)
    assert first_fingerprint == second_fingerprint

    counts: dict[SplitName, int] = {
        "train": 8,
        "validation": 4,
        "test_iid": 2,
        "test_compositional": 2,
        "test_range": 2,
        "test_format_attack": 1,
        "smoke": 1,
    }
    splits, manifest = build_splits(
        first,
        counts=counts,
        seed=7,
        generator_config=config,
    )
    example = splits["smoke"][0]
    response = render_trusted_completion(example)
    verification = verify_task_response(example, response)

    print(f"Generated tasks:     {len(first)}")
    print(f"Dataset fingerprint: {manifest.dataset_fingerprint}")
    print(f"Split counts:        {manifest.counts}")
    print(f"Example prompt:      {example.prompt}")
    print(f"Example response:    {response}")
    print(f"Verifier result:     {verification.status}")
    print("Synthetic arithmetic lab passed.")


if __name__ == "__main__":
    main()
