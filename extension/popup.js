let currentTabUrl = "";

// Duplicated from background.js (same reasoning as escapeHtml being
// duplicated across every context in this extension - no bundler, so
// small stable constants/helpers get copied rather than imported. Keep
// these two in sync with background.js if either changes.
const IGNORED_SCHEMES = ["chrome:", "chrome-extension:", "about:", "edge:", "brave:", "file:"];
const CACHE_TTL_MS = 10 * 60 * 1000;
function isCacheFresh(entry) {
  return entry && Date.now() - entry.timestamp < CACHE_TTL_MS;
}

// result.note is backend-controlled today (low
// risk), but one compromised/misconfigured backend away from XSS inside
// the extension's own privileged popup context. Same escape pattern as
// app/main.py's embedded pages.
function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
}

function renderStatus(result) {
  const area = document.getElementById("statusArea");
  if (!result) {
    document.body.className = "state-neutral";
    area.innerHTML = `<span class="badge unknown">Not yet checked</span>`;
    return;
  }
  // Backend can answer status="invalid" with verdict=null (e.g. a host
  // with no dot, like an intranet name or localhost). This used to fall
  // through to result.verdict.toUpperCase() -> TypeError: the popup died
  // mid-render with a red border and a stale badge. Render it as its own
  // neutral state instead.
  if (result.status === "invalid" || !result.verdict) {
    document.body.className = "state-neutral";
    area.innerHTML = `
      <span class="badge unknown">Can't check</span>
      <div class="helper-text">${escapeHtml(result.message || "This address can't be assessed.")}</div>
    `;
    return;
  }
  const isSafe = result.verdict === "safe";
  const cls = isSafe ? "safe" : "unsafe";
  document.body.className = isSafe ? "state-safe" : "state-unsafe";
  // 2026-07 audit fix: this only ever rendered result.note, which is
  // ONLY set for the typosquat stage - a blocklist or model-flagged
  // unsafe result showed just "UNSAFE" with no explanation at all.
  // result.reason is the backend's plain-language "why", set for every
  // unsafe stage (see app/main.py's _unsafe_reason) - prefer it, falling
  // back to note for defense-in-depth against an older/misconfigured
  // backend that predates the reason field.
  const explanation = result.reason || result.note;
  area.innerHTML = `
    <span class="badge ${cls}">${result.verdict.toUpperCase()}</span>
    ${explanation ? `<div class="note">${escapeHtml(explanation)}</div>` : ""}
  `;
}

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTabUrl = tab.url || "";
  document.getElementById("siteName").textContent = tab.title || "";
  document.getElementById("currentUrl").textContent = currentTabUrl;

  // Found via real user testing: checking the popup right after typing a
  // new URL and hitting Enter can catch the tab BEFORE navigation
  // commits - tab.url is still the previous page (often chrome://newtab/)
  // even though the address bar already shows the new URL. Show that
  // plainly instead of silently querying a cache keyed by the wrong URL.
  if (IGNORED_SCHEMES.some((s) => currentTabUrl.startsWith(s))) {
    document.body.className = "state-neutral";
    document.getElementById("statusArea").innerHTML = `
      <span class="badge unknown">Uncheckable page</span>
      <div class="helper-text">Internal browser pages and new tabs cannot be scanned by the extension.</div>
    `;
    document.getElementById("checkBtn").disabled = true;
    const { autoCheckEnabled } = await chrome.storage.sync.get("autoCheckEnabled");
    document.getElementById("autoCheckToggle").checked = autoCheckEnabled !== false;
    return;
  }

  const { urlCache } = await chrome.storage.session.get("urlCache");
  const cached = urlCache && urlCache[currentTabUrl];
  // Found via real user testing: this used to render ANY cached entry
  // with no freshness check, unlike every other read path in the
  // extension - a stale/expired result could be shown as if current.
  renderStatus(isCacheFresh(cached) ? cached : null);

  const { autoCheckEnabled } = await chrome.storage.sync.get("autoCheckEnabled");
  document.getElementById("autoCheckToggle").checked = autoCheckEnabled !== false;
}

document.getElementById("checkBtn").addEventListener("click", () => {
  const btn = document.getElementById("checkBtn");
  btn.disabled = true;
  btn.textContent = "Checking...";
  chrome.runtime.sendMessage({ type: "MANUAL_CHECK", url: currentTabUrl }, (response) => {
    btn.disabled = false;
    btn.textContent = "Check this page";
    if (response && response.ok) {
      renderStatus(response.result);
    } else {
      document.getElementById("statusArea").innerHTML =
        `<span class="badge unknown">Error - is the backend reachable?</span>`;
    }
  });
});

document.getElementById("autoCheckToggle").addEventListener("change", (e) => {
  chrome.storage.sync.set({ autoCheckEnabled: e.target.checked });
});

init();
