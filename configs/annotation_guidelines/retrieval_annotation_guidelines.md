# Retrieval Annotation Guidelines

These guidelines define how to annotate retrieved evidence chunks.

Each row in `retrieval_to_annotate.csv` corresponds to one retrieved chunk for one attack step.

## Columns to inspect

Use these columns:

- `case_id`
- `attack_step_id`
- `retriever`
- `chunking_strategy`
- `rank`
- `chunk_id`
- `document_id`
- `source_path`
- `chunk_text`
- `query`

Fill these columns:

- `annotator_id`
- `annotator_relevant`
- `annotator_evidence_type`
- `annotator_notes`

## `annotator_relevant`

Question: Is the retrieved chunk useful evidence for the attack step?

Allowed values:

- `yes`
- `partially`
- `no`

Use `yes` when the chunk directly supports, explains, or instantiates the attack step.

Use `partially` when the chunk provides useful context but does not directly support the step.

Use `no` when the chunk is irrelevant, too generic, or unrelated to the step.

## `annotator_evidence_type`

Question: What kind of evidence does the chunk provide?

Allowed values:

- `direct_step_evidence`
- `supporting_context`
- `campaign_context`
- `irrelevant`

Use `direct_step_evidence` when the chunk directly describes the technique, behavior, artifact, or command associated with the attack step.

Use `supporting_context` when the chunk helps interpret the step but does not directly describe it.

Use `campaign_context` when the chunk describes the broader intrusion, actor, malware, or campaign.

Use `irrelevant` when the chunk does not help evaluate the attack step.

## Notes

Use `annotator_notes` to explain borderline cases, especially when marking `partially` or `supporting_context`.