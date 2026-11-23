# Condition Annotation Guidelines

These guidelines define how to annotate generated pre-conditions and post-conditions for attack-step enrichment.

Each row in `conditions_to_annotate.csv` corresponds to one generated condition for one attack step.

## Columns to inspect

Use these columns when annotating:

- `case_id`
- `attack_step_id`
- `condition_type`
- `category`
- `statement`
- `evidence_chunk_ids`
- `cited_chunk_text`
- `source_documents`
- `system_support_level`
- `system_directness`
- `system_status`
- `verifier_rationale`

Fill these columns:

- `annotator_id`
- `annotator_valid`
- `annotator_grounded`
- `annotator_directness`
- `annotator_redundant`
- `annotator_category_correct`
- `annotator_notes`

## Definitions

A **condition** is a short technical fact associated with an attack step.

A **pre-condition** is a fact that must already hold before the attack step can be executed.

A **post-condition** is a fact or observable state that holds after the attack step succeeds.

A condition should be:

- atomic: it expresses one fact only;
- technical: it describes a concrete technical state, capability, artifact, or dependency;
- step-specific: it is relevant to the attack step, not only to the whole campaign;
- evidence-grounded when the pipeline cites chunks.

## `annotator_valid`

Question: Is the condition technically reasonable for the attack step?

Allowed values:

- `yes`
- `no`
- `unclear`

Use `yes` when the condition is technically meaningful for the given attack step.

Use `no` when the condition is technically wrong, irrelevant, impossible, or describes another step.

Use `unclear` when the row lacks enough context to decide.

Examples:

- Attack step: create a Windows service. Condition: “The attacker can execute commands on the host.” → `yes`
- Attack step: create a Windows service. Condition: “Sensitive files are exfiltrated.” → `no`
- Condition depends on ambiguous evidence or missing context → `unclear`

## `annotator_grounded`

Question: Do the cited chunks support the condition?

Allowed values:

- `yes`
- `partially`
- `no`

Use `yes` when the cited chunk directly supports the statement.

Use `partially` when the statement is a plausible inference, but not directly stated.

Use `no` when the cited chunks do not support the statement, cited chunks are missing, or the condition relies on outside knowledge.

For No-RAG outputs, use `no` unless an evidence chunk is actually cited and shown.

## `annotator_directness`

Question: Is the condition specific to the attack step?

Allowed values:

- `direct`
- `indirect`
- `campaign_level`

Use `direct` when the condition is specific to the attack step.

Use `indirect` when it is relevant context but not a direct requirement or result of the step.

Use `campaign_level` when the condition describes the whole intrusion, malware family, actor, or campaign rather than the specific step.

Examples:

- “The attacker can execute commands on the host.” for service creation → `direct`
- “The malware has previously been delivered to the victim.” for persistence → `indirect`
- “The victim organization has been compromised.” → `campaign_level`

## `annotator_redundant`

Question: Does this condition duplicate another condition for the same attack step?

Allowed values:

- `yes`
- `no`

Use `yes` when another condition for the same `run_id` and `attack_step_id` expresses the same fact.

Minor wording differences do not make two conditions distinct.

Example redundant pair:

- “The attacker can execute commands on the host.”
- “The attacker has command execution on the target system.”

## `annotator_category_correct`

Question: Is the assigned category appropriate for the statement?

Allowed values:

- `yes`
- `no`
- `unclear`

Allowed categories are:

- `environment`
- `privilege`
- `dependency`
- `filesystem`
- `registry`
- `process`
- `network`
- `identity`
- `log_telemetry`
- `attacker_capability`
- `other`

Examples:

- “The attacker has administrator privileges.” category `privilege` → `yes`
- “A registry run key exists.” category `registry` → `yes`
- “A network connection to C2 exists.” category `filesystem` → `no`

## Notes

Use `annotator_notes` for short explanations, especially for `unclear`, `partially`, or disagreement-prone cases.