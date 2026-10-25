# Toy retrieval lesson: relevance and recall
Authored classroom fixture, version 2. Not a published experiment.

A query asks for evidence. The retriever ranks source chunks. A high score is a ranking signal, not proof of truth or current applicability. Source-level Recall@k in this starter is the number of distinct required source IDs represented among the top k chunks divided by the total number of required source IDs. Multiple chunks from one source count only once in the numerator.

For a hypothetical query needing sources A and B, returning A and A gives recall one half. Returning A and B gives recall one. When the required source set is empty, report recall as not applicable rather than inventing a denominator. Compare the same query cases, relevance judgments, and chunk collection when evaluating two retrieval methods. Keep the held-out reference answers separate from the index.
