# GPT-Load ScriptCat Import

`gpt-load-channel-import.user.js` transfers an API key, Base URL, group name and
provider website from a Sub2API, New API, To-Tokens or Sidrune key page into GPT-Load's
new-channel import form. It does not submit the form or call either service.

How the credential is stored, claimed once, expired and pre-filled is owned by
[ScriptCat Channel Import Handoff](../../.docs/tech/scriptcat-channel-import.md).
This file only covers installing and adapting the userscript.

## Install And Use

Install the `.user.js` file in ScriptCat. Open a key page using a recognized
route, activate
`导入到 GPT-Load`, enter a group name, then confirm the API key, Base URL and
GPT-Load address before submitting. The GPT-Load address starts blank; later
uses reuse the last accepted origin.

## Matching And Security

The generic `*://*/*` match is required because the script cannot know which
provider owns a key page, and narrow host/path rules fail on new hosts or SPA
navigation. The script is injected on every HTTP(S) page, but shows its widget
only on `/console/token*`, `/keys*`, `/api-keys*` and `/user/api-keys*` routes;
login pages remain excluded. On SPAs the widget follows `popstate` and URL/path
changes.

Because the userscript has broad page access, install it only if you accept that
ScriptCat runs it on every HTTP(S) site you visit. The import panel does not
submit credentials; check the provider Base URL and GPT-Load address before
confirming. The GPT-Load address starts blank (the last accepted origin is
remembered), and the suggested provider Base URL is the current site origin.
The destination is the `/import` path with only `mode=new` and
`scriptcat_handoff` parameters.

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
