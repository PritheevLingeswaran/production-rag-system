# Decisions

## Thin orchestration (explicit code)
We implement retrieval + generation explicitly instead of a heavy LangChain graph to:
- keep runtime behavior predictable and debuggable
- avoid framework lock-in
- make evaluation straightforward

## Chroma as default
Chroma is persisted locally and supports metadata alongside vectors with minimal work.
FAISS is supported as an optional backend for lower latency.

## Strict refusal policy
In production, wrong answers are more damaging than refusals. The model is forced to:
- cite chunk ids for claims
- refuse when citations are missing/invalid
- refuse when evidence is insufficient
