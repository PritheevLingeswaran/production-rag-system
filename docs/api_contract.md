# API contract

## POST /query

Request:
```json
{"query":"string","top_k":8,"filter":{"source":"optional substring"},"rewrite_query":true}
```

Response:
```json
{"answer":"string","confidence":0.0,"sources":[{"chunk_id":"...","source":"...","page":1,"score":0.5,"text":"..."}],"refusal":{"is_refusal":false,"reason":""},"metrics":{...}}
```
