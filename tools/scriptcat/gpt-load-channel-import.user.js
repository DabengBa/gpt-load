// ==UserScript==
// @name         GPT-Load channel import
// @namespace    https://github.com/DabengBa/gpt-load
// @version      1.1.0
// @description  Send group details and API credentials to GPT-Load's import form.
// @match        *://*/*
// @run-at       document-start
// @grant        GM_openInTab
// @grant        GM.getValue
// @grant        GM.setValue
// @grant        GM.deleteValue
// @grant        GM.listValues
// @storageName  gpt-load-channel-import
// ==/UserScript==

(async () => {
  "use strict";

  const transferPrefix = "pending-import:";
  const targetOriginKey = "gpt-load-origin";
  const sessionKey = "gpt-load.scriptcat-import-payload";
  const transferTtlMs = 60_000;
  const sessionTtlMs = 15 * 60_000;
  const currentURL = new URL(location.href);

  if (
    currentURL.pathname === "/import" ||
    currentURL.pathname.startsWith("/import/")
  ) {
    const search = new URLSearchParams(currentURL.search);
    const transferId = search.get("scriptcat_handoff");
    try {
      const sessionValue = sessionStorage.getItem(sessionKey);
      if (sessionValue !== null) {
        const existing = JSON.parse(sessionValue);
        if (typeof existing.createdAt === "number") {
          const cleanupDelay = existing.createdAt + sessionTtlMs - Date.now();
          if (cleanupDelay <= 0) {
            sessionStorage.removeItem(sessionKey);
          } else {
            window.setTimeout(() => {
              if (sessionStorage.getItem(sessionKey) === sessionValue) {
                sessionStorage.removeItem(sessionKey);
              }
            }, cleanupDelay);
          }
        }
      }
    } catch {
      // Storage may be blocked; the import page remains available manually.
    }
    if (!transferId || !/^[a-z0-9-]{1,64}$/u.test(transferId)) return;

    const transferKey = `${transferPrefix}${transferId}`;
    try {
      await navigator.locks.request(`${sessionKey}:${transferId}`, async () => {
        const pending = await GM.getValue(transferKey);
        if (!pending || typeof pending !== "object") return;

        if (
          typeof pending.createdAt !== "number" ||
          pending.createdAt > Date.now() ||
          typeof pending.apiKey !== "string" ||
          typeof pending.baseURL !== "string" ||
          typeof pending.groupName !== "string" ||
          typeof pending.providerUrl !== "string"
        ) {
          await GM.deleteValue(transferKey);
          return;
        }
        if (Date.now() - pending.createdAt > transferTtlMs) {
          await GM.deleteValue(transferKey);
          return;
        }
        if (pending.targetOrigin !== location.origin) {
          return;
        }

        const sessionValue = JSON.stringify({
          version: 2,
          createdAt: Date.now(),
          apiKey: pending.apiKey,
          baseURL: pending.baseURL,
          groupName: pending.groupName,
          providerUrl: pending.providerUrl,
        });
        sessionStorage.setItem(sessionKey, sessionValue);
        window.setTimeout(() => {
          if (sessionStorage.getItem(sessionKey) === sessionValue) {
            sessionStorage.removeItem(sessionKey);
          }
        }, sessionTtlMs);
        await GM.deleteValue(transferKey);
        window.dispatchEvent(new Event("gpt-load:scriptcat-import"));
      });
    } catch {
      // Manual import remains available when Web Locks or storage are unavailable.
    }
    return;
  }

  function isSourceRoute() {
    const path = location.pathname;
    return (
      path.startsWith("/console/token") ||
      path.startsWith("/keys") ||
      path.startsWith("/api-keys") ||
      path.startsWith("/user/api-keys")
    );
  }

  function isReadySourcePage() {
    if (!isSourceRoute() || !document.body) return false;
    if (
      /(?:^|\/)(?:login|signin|sign-in|auth)(?:\/|$)/iu.test(location.pathname)
    ) {
      return false;
    }
    const pageText = document.body.innerText ?? "";
    if (!pageText.trim()) return false;
    const hasPasswordField = Boolean(
      document.querySelector('input[type="password"]'),
    );
    const looksLikeLogin =
      /(?:登录|欢迎回来|\blog\s?in\b|\bsign\s?in\b)/iu.test(pageText);
    return !(hasPasswordField && looksLikeLogin);
  }

  function suggestedAPIKey() {
    const selected = window.getSelection()?.toString().trim() ?? "";
    if (selected) return selected;
    const visibleText = document.body?.innerText ?? "";
    return visibleText.match(/\bsk-(?:fx-)?[A-Za-z0-9_-]{8,}\b/u)?.[0] ?? "";
  }

  function createWidget() {
    const host = document.createElement("div");
    host.id = "gpt-load-scriptcat-widget";
    const shadow = host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = `
      :host { all: initial; color-scheme: light; }
      :host([hidden]) { display: none !important; }
      * { box-sizing: border-box; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
      button, input { font: inherit; }
      button { cursor: pointer; }
      .launcher, .panel { position: fixed; z-index: 2147483647; }
      .launcher {
        right: 20px; bottom: 20px; min-height: 44px; padding: 0 16px;
        border: 1px solid #0b544a; border-radius: 7px; background: #0e6b5d;
        color: #fff; font-size: 14px; font-weight: 650;
        box-shadow: 0 4px 14px rgb(13 37 32 / 22%);
        transition: background-color 140ms ease, box-shadow 140ms ease;
      }
      .launcher:hover { background: #09594e; box-shadow: 0 6px 18px rgb(13 37 32 / 25%); }
      .launcher:focus-visible, .close:focus-visible, .reveal:focus-visible, input:focus-visible, .submit:focus-visible {
        outline: 3px solid #e09b36; outline-offset: 2px;
      }
      .panel {
        right: 20px; bottom: 20px; width: min(360px, calc(100vw - 24px));
        max-height: min(620px, calc(100vh - 24px)); overflow: auto;
        border: 1px solid #cbd8d3; border-radius: 8px; background: #fbfcfb;
        color: #172824; box-shadow: 0 12px 34px rgb(13 37 32 / 24%);
      }
      .panel[hidden], .launcher[hidden] { display: none !important; }
      .header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 18px 18px 10px; }
      h2 { margin: 0; font-size: 17px; line-height: 1.3; font-weight: 680; }
      .close { display: grid; place-items: center; width: 30px; height: 30px; margin: -4px -5px 0 0; border: 1px solid transparent; border-radius: 6px; background: transparent; color: #40534d; }
      .close:hover { border-color: #d5e0dc; background: #f0f4f2; }
      .close svg { width: 16px; height: 16px; }
      .notice { margin: 0 18px 14px; color: #536760; font-size: 12px; line-height: 1.5; }
      form { display: grid; gap: 12px; padding: 0 18px 18px; }
      .field { display: grid; gap: 5px; min-width: 0; }
      label { color: #293c36; font-size: 12px; font-weight: 650; }
      input { width: 100%; min-width: 0; height: 40px; padding: 0 10px; border: 1px solid #bdcbc5; border-radius: 5px; background: #fff; color: #172824; font-size: 13px; }
      input:hover { border-color: #879e95; }
      input:focus { border-color: #0e6b5d; outline: none; box-shadow: 0 0 0 2px rgb(14 107 93 / 16%); }
      .key-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px; }
      .reveal { min-width: 52px; padding: 0 9px; border: 1px solid #bdcbc5; border-radius: 5px; background: #f2f5f3; color: #334740; font-size: 12px; }
      .reveal:hover { background: #e8efeb; }
      .submit { min-height: 42px; margin-top: 2px; border: 1px solid #0b544a; border-radius: 6px; background: #0e6b5d; color: #fff; font-size: 13px; font-weight: 650; }
      .submit:hover:not(:disabled) { background: #09594e; }
      .submit:disabled { cursor: wait; opacity: .65; }
      .status { min-height: 18px; margin: -4px 0 0; color: #536760; font-size: 12px; line-height: 1.45; }
      .status[data-state="error"] { color: #a33324; }
      .status[data-state="success"] { color: #176449; }
      @media (max-width: 480px) {
        .launcher { right: 12px; bottom: 12px; }
        .panel { right: 12px; bottom: 12px; width: calc(100vw - 24px); max-height: calc(100vh - 24px); }
      }
      @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: 0.01ms !important; } }
    `;
    const surface = document.createElement("div");
    surface.innerHTML = `
      <button class="launcher" type="button" aria-expanded="false" aria-controls="sc-panel">导入到 GPT-Load</button>
      <section class="panel" id="sc-panel" aria-labelledby="sc-title" hidden>
        <div class="header">
          <h2 id="sc-title">导入 API Key</h2>
          <button class="close" type="button" aria-label="关闭导入面板" title="关闭">
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
          </button>
        </div>
        <p class="notice">供应商官网使用当前站点。请在 GPT-Load 检查信息后手动提交。</p>
        <form novalidate>
          <div class="field">
            <label for="sc-api-key">API Key</label>
            <div class="key-row">
              <input id="sc-api-key" type="password" autocomplete="off" spellcheck="false" required>
              <button class="reveal" type="button" aria-label="显示 API Key">显示</button>
            </div>
          </div>
          <div class="field">
            <label for="sc-group-name">分组名</label>
            <input id="sc-group-name" type="text" autocomplete="organization" maxlength="256" required>
          </div>
          <div class="field">
            <label for="sc-base-url">Base URL</label>
            <input id="sc-base-url" type="url" inputmode="url" autocomplete="url" required>
          </div>
          <div class="field">
            <label for="sc-target-url">GPT-Load 地址</label>
            <input id="sc-target-url" type="url" inputmode="url" placeholder="https://gpt-load.example.com" required>
          </div>
          <button class="submit" type="submit">打开导入页</button>
          <p class="status" role="status" aria-live="polite"></p>
        </form>
      </section>
    `;
    shadow.append(style, surface);

    const launcher = shadow.querySelector(".launcher");
    const panel = shadow.querySelector(".panel");
    const close = shadow.querySelector(".close");
    const form = shadow.querySelector("form");
    const apiKeyInput = shadow.querySelector("#sc-api-key");
    const groupNameInput = shadow.querySelector("#sc-group-name");
    const baseURLInput = shadow.querySelector("#sc-base-url");
    const targetURLInput = shadow.querySelector("#sc-target-url");
    const reveal = shadow.querySelector(".reveal");
    const submit = shadow.querySelector(".submit");
    const status = shadow.querySelector(".status");
    let fieldsInitialized = false;

    const setStatus = (message, state = "") => {
      status.textContent = message;
      status.dataset.state = state;
    };
    const setOpen = (open) => {
      launcher.hidden = open;
      launcher.setAttribute("aria-expanded", String(open));
      panel.hidden = !open;
      if (open) {
        if (!fieldsInitialized) {
          apiKeyInput.value = suggestedAPIKey();
          groupNameInput.value = location.hostname;
          baseURLInput.value = location.origin;
          fieldsInitialized = true;
          setStatus("");
        }
        apiKeyInput.focus();
        GM.getValue(targetOriginKey).then(
          (savedOrigin) => {
            if (!targetURLInput.value && savedOrigin) {
              targetURLInput.value = savedOrigin;
            }
          },
        );
      } else {
        launcher.focus();
      }
    };

    launcher.addEventListener("click", () => setOpen(true));
    close.addEventListener("click", () => setOpen(false));
    reveal.addEventListener("click", () => {
      const visible = apiKeyInput.type === "text";
      apiKeyInput.type = visible ? "password" : "text";
      reveal.textContent = visible ? "显示" : "隐藏";
      reveal.setAttribute("aria-label", `${visible ? "显示" : "隐藏"} API Key`);
    });
    shadow.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) setOpen(false);
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const apiKey = apiKeyInput.value.trim();
      const groupName = groupNameInput.value.trim();
      const baseURL = baseURLInput.value.trim();
      const target = targetURLInput.value.trim();
      if (!apiKey || !groupName || !baseURL || !target) {
        setStatus(
          "请填写 API Key、分组名、Base URL 和 GPT-Load 地址。",
          "error",
        );
        return;
      }

      let parsedBaseURL;
      let targetURL;
      try {
        parsedBaseURL = new URL(baseURL);
        targetURL = new URL(target);
        if (
          !["http:", "https:"].includes(parsedBaseURL.protocol) ||
          !["http:", "https:"].includes(targetURL.protocol)
        ) {
          throw new Error("invalid URL");
        }
      } catch {
        setStatus("请输入有效的 HTTP 或 HTTPS 地址。", "error");
        return;
      }

      submit.disabled = true;
      setStatus("正在准备导入页…");
      try {
        for (const name of await GM.listValues()) {
          if (!name.startsWith(transferPrefix)) continue;
          const stale = await GM.getValue(name);
          if (
            !stale ||
            typeof stale.createdAt !== "number" ||
            stale.createdAt > Date.now() ||
            Date.now() - stale.createdAt > transferTtlMs
          ) {
            await GM.deleteValue(name);
          }
        }
        targetURL.pathname = "/import";
        targetURL.search = "?mode=new";
        targetURL.hash = "";
        const transferId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        const transferKey = `${transferPrefix}${transferId}`;
        targetURL.searchParams.set("scriptcat_handoff", transferId);
        await GM.setValue(transferKey, {
          createdAt: Date.now(),
          targetOrigin: targetURL.origin,
          apiKey,
          baseURL: parsedBaseURL.href.replace(/\/$/u, ""),
          groupName,
          providerUrl: location.origin,
        });
        await GM.setValue(targetOriginKey, targetURL.origin);
        window.setTimeout(async () => {
          await GM.deleteValue(transferKey);
        }, transferTtlMs);
        GM_openInTab(targetURL.href, { active: true });
        setStatus("导入页已打开。请检查信息并手动提交。", "success");
      } catch {
        setStatus("交接失败。请检查 ScriptCat 权限后重试。", "error");
      } finally {
        submit.disabled = false;
      }
    });

    document.documentElement.append(host);
    return host;
  }

  let widget = null;
  let mutationTimer = null;
  function updateWidget() {
    if (widget && !widget.isConnected) widget = null;
    if (!isSourceRoute()) {
      if (widget) widget.hidden = true;
      return;
    }
    if (!widget && isReadySourcePage()) widget = createWidget();
    if (widget) widget.hidden = !isReadySourcePage();
  }

  function startWidget() {
    if (!document.documentElement) {
      window.setTimeout(startWidget, 50);
      return;
    }
    updateWidget();
    const observer = new MutationObserver(() => {
      if (mutationTimer !== null) window.clearTimeout(mutationTimer);
      mutationTimer = window.setTimeout(updateWidget, 120);
    });
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
    window.addEventListener("popstate", updateWidget);
    window.setInterval(updateWidget, 600);
  }

  startWidget();
})();
