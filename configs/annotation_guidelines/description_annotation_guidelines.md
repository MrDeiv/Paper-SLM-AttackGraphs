# Description Annotation Guidelines

These guidelines define how to annotate generated attack-step descriptions.

Each row in `descriptions_to_annotate.csv` corresponds to one generated description for one attack step.

## Columns to inspect

Use these columns:

- `case_id`
- `attack_step_id`
- `description`
- `evidence_chunk_ids`
- `cited_chunk_text`
- `source_documents`
- `system_support_level`
- `system_status`

Fill these columns:

- `annotator_id`
- `annotator_faithfulness`
- `annotator_specificity`
- `annotator_grounded`
- `annotator_notes`

## `annotator_faithfulness`

Question: Does the description accurately reflect the cited evidence?

Allowed values:

- `faithful`
- `partially_faithful`
- `unfaithful`

Use `faithful` when the description accurately summarizes the cited chunks without adding unsupported details.

Use `partially_faithful` when the description is mostly correct but adds, omits, or overstates some detail.

Use `unfaithful` when the description contradicts the evidence, describes a different step, or relies mainly on unsupported information.

## `annotator_specificity`

Question: Is the description specific enough to be useful?

Allowed values:

- `specific`
- `generic`

Use `specific` when the description includes concrete technical details from the evidence.

Use `generic` when the description is broadly true but could apply to many unrelated attacks.

Examples:

- “The malware creates a Windows service to restart after reboot.” → `specific`
- “The attacker establishes persistence on the system.” → `generic`

## `annotator_grounded`

Question: How strongly do the cited chunks support the description?

Allowed values:

- `explicit`
- `inferred`
- `unsupported`

Use `explicit` when the cited chunks directly state the main description.

Use `inferred` when the description is a reasonable technical inference from the cited chunks.

Use `unsupported` when the cited chunks do not support the description or no chunks are cited.

## Notes

Use `annotator_notes` to explain borderline cases, missing evidence, or descriptions that are faithful but too generic.