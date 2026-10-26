# GPT-Load ScriptCat Import

`gpt-load-channel-import.user.js` transfers an API key, Base URL, group name,
and provider website from a Sub2API, New API, or To-Tokens key page into
GPT-Load's new-channel import form. It does not submit the form or make API
requests to either service.

## Use

Install the `.user.js` file in ScriptCat. A `导入到 GPT-Load` control appears
on supported API-key pages. Open it, enter a group name, and confirm or edit
the API key, Base URL, and GPT-Load address. The provider website is set to the
current page's origin. Select the complete key before opening the panel if it
is not shown in page text. The first use defaults to
`https://gptl.tanyaleoallen.cloud`; later uses reuse the last selected origin.
The default Base URL is the current site root; verify it for the selected
service endpoint and add any required path.

The script's match list includes common `/console/token`, `/api-keys`, and
`/user/api-keys` routes, `https://totokens.cc/keys`, plus GPT-Load's `/import`
route. For self-hosted installations with different paths, add the source and
GPT-Load match patterns in ScriptCat's script editor. Keep the host/path
patterns as narrow as the deployment allows.

## Development

Open this `.user.js` file in VS Code with the ScriptCat extension installed.
Enable ScriptCat's VS Code development connection, run `scriptcat.autoTarget`
once, then save to sync edits to the browser. Alternatively, edit the script
in ScriptCat directly.

For local GPT-Load UI testing, run the Web Vite server with
`pnpm --dir web run dev`. It serves on the Vite default port and proxies API
requests to `http://127.0.0.1:3001`; start the GPT-Load API there or set
`VITE_DEV_PROXY_TARGET` to its address. In the source-page panel, enter the
local Vite URL, such as `http://localhost:5173`.

## Credential Handling

The key, Base URL, group name, and provider website are stored in a one-use
transfer in ScriptCat's shared value store, bound to the exact GPT-Load origin
entered in the panel. A non-secret transfer ID in the destination URL selects
the matching transfer; credentials and provider details are never put in the
URL. The destination script copies the values to same-origin `sessionStorage`
and removes the ScriptCat entry. The
source tab removes an abandoned transfer after one minute while it remains
open; a later transfer attempt also clears expired entries. GPT-Load accepts
the payload for up to 15 minutes to allow for authentication, consumes it once,
and removes it before rendering the draft. A cleanup timer removes an unconsumed
payload when its TTL elapses if the destination document stays open. If a full
reload interrupts the timer, opening `/import` again cleans an expired payload;
closing the tab clears its `sessionStorage`. The user must review and submit
the import form manually.

ScriptCat's storage is local to the browser profile. Do not sync or share the
value store while a transfer is pending. If the source tab is closed before
the destination opens, the plaintext ScriptCat value can remain until another
transfer attempt or a matching destination page cleans it up.
