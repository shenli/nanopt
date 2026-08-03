"""Run the fixed arithmetic verifier-attack suite on CPU."""

from __future__ import annotations

from nanopt.config.models import RewardComponentsConfig
from nanopt.data.arithmetic import generate_task
from nanopt.grpo.reward import reward_hacking_suite


def main() -> None:
    """Prove familiar formatting attacks receive no correctness reward."""

    task = generate_task(family="mixed_precedence", difficulty=2, seed=14)
    weights = RewardComponentsConfig(correctness=1.0, format=0.1, length_penalty=0.0)
    results = reward_hacking_suite(task, weights)
    for result in results:
        print(f"{result['case']:<20} correctness={result['correctness_reward']}")
    assert len(results) == 5
    assert all(bool(result["passed"]) for result in results)
    print("Reward-hacking lab passed.")


if __name__ == "__main__":
    main()
