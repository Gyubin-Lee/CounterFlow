# Results

Generated outputs and evaluation results live here and are ignored by Git.

Canonical layout is grouped by CounterFlow research axis:

```text
results/research_axes/latent_update_method/
results/research_axes/target_prompt_generation/
results/research_axes/evaluation_metrics/
```

Recommended experiment convention:

```text
results/research_axes/<primary_axis>/prompt_generation/<dataset>/YYYYMMDD_<run_description>/
results/research_axes/<primary_axis>/evaluation/<dataset>/qualitative/YYYYMMDD_<exp_description>/
results/research_axes/<primary_axis>/evaluation/<dataset>/quantitative/YYYYMMDD_<exp_description>/
results/research_axes/<primary_axis>/logs/YYYYMMDD_<run_description>/
```

New result folder names must use `YYYYMMDD_<short_experiment_description>`.
Use an 8-digit KST date with no hyphens, followed by lower snake case. Keep the
same experiment description across qualitative media, quantitative metrics, and
logs when they belong to the same run.

Use the largest/primary research axis as the single owner for all artifacts from
one experiment. Do not split prompt files, generated media, metrics, and logs
across multiple axis folders.

Examples:

```text
Prompt-generation method experiment:
  results/research_axes/target_prompt_generation/prompt_generation/<dataset>/YYYYMMDD_qwen_temporal_preserve36/
  results/research_axes/target_prompt_generation/evaluation/<dataset>/qualitative/YYYYMMDD_qwen_temporal_preserve36_mmaudio/
  results/research_axes/target_prompt_generation/evaluation/<dataset>/quantitative/YYYYMMDD_qwen_temporal_preserve36_mmaudio/

Latent-update method experiment:
  results/research_axes/latent_update_method/evaluation/<dataset>/qualitative/YYYYMMDD_mmaudio_vggsparse_transition13_p2neg/
  results/research_axes/latent_update_method/evaluation/<dataset>/quantitative/YYYYMMDD_mmaudio_vggsparse_transition13_p2neg/
```

Evaluation-metric experiments are the exception: do not store generated audio or
video under `results/research_axes/evaluation_metrics/`. That axis should contain
metric tables, plots, analysis files, and logs only. Point metric runs at the
media directory owned by the prompt-generation or latent-update axis that
created the media.

Latent trajectory diagnostics remain under the latent-update axis:

```text
results/research_axes/latent_update_method/latent_trajectory_pca/YYYYMMDD_<run_description>/
```

Use the axis-grouped paths above as the only canonical result locations. Avoid
creating legacy result aliases or duplicate result folders; update old
notes and commands to point at `results/research_axes/<axis>/...` when they are
used again.

Do not commit generated samples, metric tables, plots, or intermediate outputs.
