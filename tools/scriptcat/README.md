# GPT-Load ScriptCat Import

`gpt-load-channel-import.user.js` transfers an API key, Base URL, group name and
provider website from a Sub2API, New API or To-Tokens key page into GPT-Load's
new-channel import form. It does not submit the form or call either service.

How the credential is stored, claimed once, expired and pre-filled is owned by
[ScriptCat Channel Import Handoff](../../.docs/tech/scriptcat-channel-import.md).
This file only covers installing and adapting the userscript.

## Install And Use

Install the `.user.js` file in ScriptCat. Open a supported key page, activate
`导入到 GPT-Load`, enter a group name, then confirm the API key, Base URL and
GPT-Load address before submitting. The first use defaults to
`https://gptl.tanyaleoallen.cloud`; later uses reuse the last accepted origin.

## Adapt To A Deployment

The userscript recognizes provider key pages through two places that must stay
in agreement:

- the `@match` block in the metadata header, which decides whether the
  userscript runs on a page at all;
- `isSourceRoute()` in the script body, which decides whether the panel is
  offered on that page.

The recognized source routes are `/console/token*`, `/console/tokens*`,
`/api-keys*`, `/user/api-keys*` and `https://totokens.cc/keys*`. GPT-Load's
destination is the `/import` path; the script sets that path itself and adds
only the `mode=new` and `scriptcat_handoff` parameters. A GPT-Load deployment
that serves the import page elsewhere needs a script edit, not a configuration
option.

For a self-hosted provider or gateway with different paths, add the source
pattern to `@match` and to `isSourceRoute()` together, and keep both as narrow
as the deployment allows. Removing a pattern from only one of them leaves a
script that runs but never offers the panel, or a panel that never appears.

## Develop

Open the `.user.js` file in VS Code with the ScriptCat extension installed,
enable ScriptCat's VS Code development connection, run `scriptcat.autoTarget`
once, then save to sync edits to the browser. Alternatively, edit the script in
ScriptCat directly.

For local UI testing, run `pnpm --dir web run dev`. It serves on the Vite
default port and proxies API requests to `http://127.0.0.1:3001`; start the
GPT-Load API there or set `VITE_DEV_PROXY_TARGET` to its address. Enter the local
Vite URL, for example `http://localhost:5173`, as the GPT-Load address in the
panel.
