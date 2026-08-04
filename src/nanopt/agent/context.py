"""One visible contract for turning agent state into chat messages.

Keeping this logic outside the policy and trainer is important: online generation and offline
Agent SFT must see byte-identical prompts for a given trajectory prefix.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from nanopt.agent.records import ACTION_ADAPTER, AgentObservation, AgentTrajectory

AgentContextPolicy = Literal["observation_snapshot", "full_transcript"]


def agent_system_instruction() -> str:
    """Return the stable instruction shared by data collection and inference."""

    schema = json.dumps(ACTION_ADAPTER.json_schema(), sort_keys=True)
    return (
        "You are editing a tiny repository through allow-listed tools. Return exactly one JSON "
        "object and no prose. Arbitrary shell commands are unavailable. Never modify tests. "
        f"Your action must validate against this schema: {schema}"
    )


def observation_text(observation: AgentObservation, *, include_transcript: bool) -> str:
    """Serialize an observation canonically, optionally removing duplicated history."""

    value = observation.model_dump(mode="json", exclude_none=False)
    if not include_transcript:
        value["transcript"] = []
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def trajectory_messages(
    trajectory: AgentTrajectory,
    step_index: int,
    *,
    context_policy: AgentContextPolicy,
) -> list[dict[str, str]]:
    """Rebuild the exact prompt messages for one retained trajectory action."""

    if step_index < 0 or step_index >= len(trajectory.steps):
        raise ValueError("step_index is outside the trajectory")
    current = AgentObservation.model_validate(trajectory.steps[step_index].observation, strict=True)
    system = {"role": "system", "content": agent_system_instruction()}
    if context_policy == "observation_snapshot":
        return [
            system,
            {"role": "user", "content": observation_text(current, include_transcript=True)},
        ]

    messages = [system]
    for index in range(step_index + 1):
        observation = AgentObservation.model_validate(
            trajectory.steps[index].observation, strict=True
        )
        messages.append(
            {
                "role": "user",
                "content": observation_text(observation, include_transcript=False),
            }
        )
        if index < step_index:
            messages.append(
                {"role": "assistant", "content": trajectory.steps[index].model_response}
            )
    return messages


def canonical_message_value(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Narrow helper used in hashes without depending on mutable caller dictionaries."""

    return [{"role": item["role"], "content": item["content"]} for item in messages]
