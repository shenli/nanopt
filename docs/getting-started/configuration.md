# Configuration

NanoPT validates every hardware, model, experiment, and recipe profile with strict typed models.
Unknown fields fail before a model is downloaded. Profile namespaces remain separate in the
resolved document so similarly named settings cannot overwrite one another.

```bash
nanopt config resolve \
  --hardware rtx_4070_ti_super_16gb \
  --model qwen3_0_6b_base \
  --experiment math_grpo \
  --set rollout.group_size=2 \
  --output resolved_config.yaml
```

This writes `resolved_config.yaml` and `resolved_config.provenance.yaml`. An unprefixed dotted
override targets the experiment profile. Use `hardware.` or `model.` prefixes for those namespaces.
Only scalar leaves may be overridden; unknown paths, list replacement through the CLI, and type
mismatches are rejected.
