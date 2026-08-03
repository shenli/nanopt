"""Inspect exact stored rollout coordinates without a model or network."""

from __future__ import annotations

from nanopt.grpo.records import (
    GrpoCompletionRecord,
    GrpoPromptRecord,
    GrpoTrajectoryRecord,
)
from nanopt.grpo.trainer import collate_grpo_completions


def main() -> None:
    """Show how completion log probabilities move into causal prediction coordinates."""

    prompt = GrpoPromptRecord(
        messages=[{"role": "user", "content": "Compute 1 + 1."}],
        token_ids=[10, 11, 12],
        attention_mask=[1, 1, 1],
    )
    completion = GrpoCompletionRecord(
        completion_index=0,
        token_ids=[20, 21],
        action_mask=[1, 1],
        old_logprobs=[-0.4, -0.2],
        decoded_text="<answer>2</answer>",
        finish_reason="protocol_stop",
        reward=1.1,
        reward_components={"format_reward": 1.0, "correctness_reward": 1.0},
        advantage=1.0,
        parser_status="valid",
        parsed_answer="2",
        verifier_status="correct",
        generation_seconds=0,
    )
    trajectory = GrpoTrajectoryRecord(
        trajectory_id="lab",
        run_id="lab",
        iteration=0,
        task_id="one-plus-one",
        prompt=prompt,
        group_reward_mean=0.55,
        group_reward_std=0.55,
        advantage_mode="group_zscore",
        completions=[completion, completion.model_copy(update={"completion_index": 1})],
    )
    restored = GrpoTrajectoryRecord.model_validate_json(trajectory.model_dump_json(), strict=True)
    batch = collate_grpo_completions(
        [(restored, restored.completions[0])],
        pad_token_id=0,
    )

    print("Full token IDs:       ", batch.input_ids.tolist()[0])
    print("Full action mask:     ", batch.action_mask.tolist()[0])
    print("Causal old logprobs:  ", batch.old_logprobs.tolist()[0])
    assert batch.input_ids.tolist()[0] == [10, 11, 12, 20, 21]
    assert batch.action_mask.tolist()[0] == [False, False, False, True, True]
    assert batch.old_logprobs.shape == (1, 4)
    print("Exact RLVR trajectory lab passed.")


if __name__ == "__main__":
    main()
