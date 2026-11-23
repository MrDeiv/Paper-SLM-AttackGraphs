# Annotation Codebook

## Task

You are annotating outputs produced by a CTI enrichment pipeline.  Each attack
step is shown as a small group: first the enriched description, followed by its
related preconditions and postconditions.  The grouping is intentional and is
meant to improve readability.  The attack-step groups themselves are randomized.

Annotators must work independently.  Use only the cited evidence chunks shown in the
`evidence_texts` column when judging evidence support.  Do not use external knowledge
unless the criterion explicitly asks about general meaningfulness or atomicity.

## Row types

- `description`: an enriched attack-step description.
- `condition`: a generated precondition or postcondition.

## Condition labels

`condition_validity`: 2 = valid, 1 = partially valid, 0 = invalid.

`evidence_support`: 2 = directly supported, 1 = inferentially supported, 0 = unsupported.

`directness`: 2 = direct, 1 = indirect but relevant, 0 = campaign-level or unrelated.

`meaningfulness`: 1 = meaningful, 0 = generic.

`atomicity`: 1 = atomic, 0 = compound.

`type_validity`: 2 = correct type, 1 = ambiguous type, 0 = wrong type.

`non_redundancy`: 2 = non-redundant, 1 = partially overlapping, 0 = redundant.

Use `condition_comment` for ambiguous or difficult cases.

## Description labels

`description_faithfulness`: 2 = faithful, 1 = partially faithful, 0 = not faithful.

`description_evidence_support`: 2 = supported, 1 = partially supported or unclear, 0 = unsupported.

`description_specificity`: 2 = specific, 1 = partially specific, 0 = generic.

`unsupported_claims`: 2 = no unsupported claims, 1 = minor unsupported claims, 0 = major unsupported claims.

Use `description_comment` for ambiguous or difficult cases.

## Not applicable fields

Cells marked `NA` are not applicable and must not be edited.
