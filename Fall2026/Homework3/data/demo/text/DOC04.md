# Toy experiment: separate development and test
Authored classroom fixture, version 1.

Use development cases to choose a prompt, retrieval method, chunk size, or retrieval count. For a controlled comparison, change one component and hold the remaining settings fixed. Freeze the chosen configuration and source snapshot before the final test. Reference answers and grading notes must not be supplied to the model or retriever. The evidence documents needed to answer a test question can and should remain available in the knowledge base.

Report unsuccessful episodes and costs, rather than only successful demonstrations. Comparing a direct call with RAG and an agent changes information access and computation as well as control flow; it is an end-to-end comparison, not proof that orchestration alone caused a gain. More confidence in a topic is not identical to demonstrated understanding.
