"""Generate and inspect controlled arithmetic preference pairs on CPU."""

from __future__ import annotations

from nanopt.data.arithmetic import generate_task
from nanopt.data.preferences import generate_preference_pairs


def main() -> None:
    """Show all three declared rejection modes without model weights or network access."""

    tasks = []
    for seed in range(6):
        task = generate_task(family="addition_subtraction", difficulty=1, seed=seed)
        split = "train" if seed < 3 else "validation"
        tasks.append(task.model_copy(update={"split": split}))

    pairs, audit = generate_preference_pairs(
        tasks,
        source_dataset_fingerprint="cpu-lab-source",
        seed=42,
    )
    print(f"Pairs: {audit.pair_count}")
    print(f"Fingerprint: {audit.dataset_fingerprint}")
    print(f"Rejection types: {audit.rejection_type_counts}")
    print(f"Mean rejected/chosen character ratio: {audit.rejected_to_chosen_character_ratio:.3f}")
    for pair in sorted(pairs, key=lambda item: item.rejection_type):
        print(f"{pair.rejection_type:18s} {pair.pair_id} rejected={pair.rejected!r}")
    print("Controlled preference lab passed.")


if __name__ == "__main__":
    main()
