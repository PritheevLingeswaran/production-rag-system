# Security

This is a runnable baseline. For production:
- use a secrets manager, not `.env`
- add auth + rate limiting
- implement document ACL/tenant enforcement in retrieval
- harden against prompt injection (already strict, but add moderation if needed)
