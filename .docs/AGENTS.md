# Documentation Schema

This repository keeps current documentation in three owner families:

- `.docs/db/` contains product semantics only and is validated by the semantic document compiler.
- `.docs/tech/` owns implementation, architecture, operations, and calculation facts.
- `.docs/adr/` is reserved for hard-to-reverse decisions with explicit trade-offs.

Keep source material immutable. A document has one owner; link to another owner instead of copying its facts. Generated semantic database output under `.docs/db/dist/` is ignored and must be rebuilt with the checked-in compiler.

`.docs/README.md` is the project brief. `.docs/PROJECT_HISTORY.md` is a selective
record of shipped architectural changes, not a complete delivery history.
Record significant product changes and their reasons in `.docs/product-changelog/`,
linking evidence, current owners and delivery records rather than duplicating
them. Preserve new original user communications under `.docs/communications/`;
distinguish original statements and context from interpretations and unconfirmed
opinions. Existing `.docs/tech/briefs/` are immutable historical evidence, not
current technical owners. Do not move or rewrite raw sources.

Run documentation validation from the repository root with:

- `pnpm --dir web run docs:check`
- `pnpm --dir web run docs:build`

For read-only discovery use `bun run docs:tree`, `bun run docs:search -- "<concept>"`,
`bun run docs:index -- --topic <topic>`, and `bun run docs:show -- <path>`
using paths relative to `.docs/` as returned by the index.
Search results are candidates; verify ownership and current facts in source and tests.

Technical owners use front matter with `description`, `kind`, and `topic`. Add narrow `code.paths` entries only for durable implementation boundaries. Do not treat retrieval metadata or generated output as a second source of truth.
