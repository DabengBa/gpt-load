# Documentation Schema

This repository keeps current documentation in three owner families:

- `.docs/db/` contains product semantics only and is validated by the semantic document compiler.
- `.docs/tech/` owns implementation, architecture, operations, and calculation facts.
- `.docs/adr/` is reserved for hard-to-reverse decisions with explicit trade-offs.

Keep source material immutable. A document has one owner; link to another owner instead of copying its facts. Generated semantic database output under `.docs/db/dist/` is ignored and must be rebuilt with the checked-in compiler.

Run documentation validation from the repository root with:

- `pnpm --dir web run docs:check`
- `pnpm --dir web run docs:build`

Technical owners use front matter with `description`, `kind`, and `topic`. Add narrow `code.paths` entries only for durable implementation boundaries. Do not treat retrieval metadata or generated output as a second source of truth.
