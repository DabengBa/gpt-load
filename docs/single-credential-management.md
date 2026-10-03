# Single credential management API

Each group owns at most one upstream credential. Deleting it leaves an empty
group that cannot serve requests until configured again. Credential IDs remain
stable identity references in request logs and billing; cross-group health
counts and scheduling still aggregate multiple groups.

All paths below are relative to `/api`.

| Method | Path | Contract |
| --- | --- | --- |
| GET | `/groups/:group_id/credential` | `{credential: item \| null, observation: observation \| null}`; empty groups return both null |
| POST | `/groups/:group_id/credential` | Configure an empty API-key group with `{credential: string}`; returns `{group_id, credential_id}` |
| PUT | `/groups/:group_id/credential` | Edit the current API key with `{credential: string}` |
| DELETE | `/groups/:group_id/credential` | Delete the current credential, preserving the group |
| POST | `/groups/:group_id/credential/connect` | Connect one subscription stage with `{staged_credential_id: string}`; returns `{group_id, credential_id}` |
| POST | `/groups` | Create with one `credential` or `staged_credential_id`; returns `{group_id, group_name, credential_id}` |

Single-account actions use the same singular resource with `/test`, `/refresh`,
`/reveal`, `/download`, `/restore`, `/observation-refresh`, or `/reset-credits/consume`.
Existing methods and action-specific response bodies apply. Creation,
configuration and connection retain their idempotency-key requirements.

Group summary and collection items expose `credential_configured: boolean` and
`credential_status: status | null`. An empty group has false/null. These fields
replace counts inside a group; dashboard aggregate counts remain meaningful.

An occupied group rejects a second credential, including an identical API key.
Multiple text lines, including duplicate lines, are invalid. One structured JSON
credential may span lines. Subscription reauthorization must match the existing
account identity and its eligible authorization state; another account is rejected.

Plural credential collection routes, pagination/filter contracts, batch deletion,
download-all and multi-value import bodies are removed without compatibility aliases.
