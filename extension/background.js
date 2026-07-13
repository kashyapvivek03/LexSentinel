// background.js
// ==============
// Core logic. Intercepts top-level navigations via webNavigation (NOT the
// address bar itself - no extension can read keystrokes typed there; this
// fires the moment the browser starts navigating to a URL, which is the
// practical equivalent and early enough to warn before the page loads).
//
// HONEST LIMITATION: Manifest V3 does not allow extensions to synchronously
// block navigation while an async check runs (that capability - blocking
// webRequest - was removed in MV3 for most extensions). So this checks and
// redirects AS FAST AS POSSIBLE after navigation starts, rather than
// guaranteeing zero content ever renders. In practice, with a warm backend,
// this is usually fast enough that nothing perceptible loads - but on a
// slow/cold-started free-tier backend, a brief flash of the real page is
// possible before the redirect lands. This is a platform constraint, not a
// bug - the same constraint every MV3 safe-browsing-style extension faces.

const DEFAULT_SETTINGS = {
  backendUrl: "https://phishing-detector-f65f.onrender.com",
  autoCheckEnabled: true,
};

const CACHE_TTL_MS = 10 * 60 * 1000;      // 10 min: re-check a site periodically
const MAX_CACHE_ENTRIES = 500;            // see setCacheEntry below
const IGNORED_SCHEMES = ["chrome:", "chrome-extension:", "about:", "edge:", "brave:", "file:"];

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  const toSet = {};
  for (const [k, v] of Object.entries(DEFAULT_SETTINGS)) {
    if (existing[k] === undefined) toSet[k] = v;
  }
  if (Object.keys(toSet).length) await chrome.storage.sync.set(toSet);
});

async function getSettings() {
  const stored = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function getCache() {
  const { urlCache } = await chrome.storage.session.get("urlCache");
  return urlCache || {};
}
async function setCacheEntry(url, entry) {
  const cache = await getCache();
  cache[url] = { ...entry, timestamp: Date.now() };

  // urlCache accumulated one entry per distinct
  // URL visited, TTL-checked on READ but never evicted on WRITE.
  // chrome.storage.session has a ~10MB quota; heavy browsing eventually
  // makes storage.session.set() start throwing, which then breaks
  // caching entirely (every navigation silently re-fetches). Prune
  // expired entries every write, and cap at MAX_CACHE_ENTRIES (drop
  // oldest first) as a hard backstop even if TTL pruning somehow isn't
  // enough (e.g. clock skew, extremely high-volume browsing in one
  // 10-minute window).
  const now = Date.now();
  for (const [cachedUrl, cachedEntry] of Object.entries(cache)) {
    if (now - cachedEntry.timestamp >= CACHE_TTL_MS) delete cache[cachedUrl];
  }
  const entries = Object.entries(cache);
  if (entries.length > MAX_CACHE_ENTRIES) {
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp); // oldest first
    const toDrop = entries.length - MAX_CACHE_ENTRIES;
    for (let i = 0; i < toDrop; i++) delete cache[entries[i][0]];
  }

  await chrome.storage.session.set({ urlCache: cache });
}
function isCacheFresh(entry) {
  return entry && Date.now() - entry.timestamp < CACHE_TTL_MS;
}

function stripForPrivacy(urlStr) {
  // every top-level navigation - full URL,
  // INCLUDING query strings (which routinely contain search terms,
  // session tokens, password-reset links, document IDs) - was POSTed to
  // the backend. This is a real privacy exposure: it amounts to sending
  // meaningful fragments of the user's browsing activity to a
  // third-party server on every single page load.
  //
  // Implementing the review's recommended option 1: strip query string
  // and fragment before sending, keep only origin + path. The model's
  // path-based features (suspicious keywords, path depth, etc.) still
  // work; query-string-based signal is lost for the EXTENSION's checks
  // specifically (the standalone web checker at "/" is unaffected - a
  // user pasting a full URL there is giving explicit, one-time consent
  // to check exactly what they typed, a different privacy posture than
  // silently checking every navigation).
  //
  // TODO (deferred, review option 2, ~half a day): ship the allowlist +
  // a local Bloom filter of popular domains inside the extension and
  // only call the backend for domains that miss it. Most browsing is
  // top-1000 sites - this would eliminate ~95% of backend calls entirely,
  // which is strictly better for privacy than stripping query strings
  // from the calls that still happen. Not done in this change set.
  try {
    const u = new URL(urlStr);
    return u.origin + u.pathname;
  } catch {
    return urlStr; // unparseable - fall through, backend has its own guard
  }
}

async function checkUrlWithBackend(url, backendUrl) {
  const privacySafeUrl = stripForPrivacy(url);
  const res = await fetch(`${backendUrl}/api/check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: privacySafeUrl }),
  });
  if (!res.ok) throw new Error(`Backend returned ${res.status}`);
  return res.json();
}

// chrome.action.* and chrome.tabs.* calls below are all keyed on a tabId
// captured before an async backend round-trip (checkUrlWithBackend). By
// the time that call resolves, the user may have already closed the tab
// or navigated it elsewhere, making the tabId stale - both APIs reject
// with "No tab with id: N" in that case (visible as an unhandled promise
// rejection in chrome://extensions's Errors page). That's an expected
// race, not a bug: there's nothing useful to do with a badge/redirect for
// a tab that no longer exists, so swallow it here rather than let it
// surface as a spurious error.
async function safeTabCall(promise) {
  try {
    await promise;
  } catch (err) {
    if (!String(err && err.message).includes("No tab with id")) throw err;
  }
}

async function setBadge(tabId, verdict) {
  if (verdict === "unsafe") {
    await safeTabCall(chrome.action.setBadgeText({ tabId, text: "!" }));
    await safeTabCall(chrome.action.setBadgeBackgroundColor({ tabId, color: "#dc2626" }));
  } else if (verdict === "safe") {
    await safeTabCall(chrome.action.setBadgeText({ tabId, text: "" })); // clear - safe is the quiet default
  } else {
    await safeTabCall(chrome.action.setBadgeText({ tabId, text: "?" }));
    await safeTabCall(chrome.action.setBadgeBackgroundColor({ tabId, color: "#6b7280" }));
  }
}

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return; // only top-level page loads, not iframes/ads
  const url = details.url;
  if (IGNORED_SCHEMES.some((s) => url.startsWith(s))) return;

  const settings = await getSettings();
  if (!settings.autoCheckEnabled) return;

  const cache = await getCache();
  const cached = cache[url];
  if (isCacheFresh(cached)) {
    setBadge(details.tabId, cached.verdict);
    if (cached.verdict === "unsafe") redirectToWarning(details.tabId, url, cached);
    return;
  }

  try {
    const result = await checkUrlWithBackend(url, settings.backendUrl);
    await setCacheEntry(url, result);
    setBadge(details.tabId, result.verdict);
    if (result.verdict === "unsafe") redirectToWarning(details.tabId, url, result);
  } catch (err) {
    // Backend unreachable (cold start, offline, misconfigured URL) - fail
    // OPEN, not closed: don't block browsing just because the checker is
    // down. Badge shows "?" so it's visible rather than silently swallowed.
    console.warn("Phishing checker: backend unreachable", err);
    setBadge(details.tabId, null);
  }
});

function redirectToWarning(tabId, blockedUrl, result) {
  const params = new URLSearchParams({
    url: blockedUrl,
    stage: result.stage || "",
    confidence: result.confidence != null ? result.confidence : "",
    note: result.note || "",
    // 2026-07 audit fix: this was already being built and passed through
    // to warning.html, but warning.js never read note/stage/confidence
    // and warning.html had no element to show them - the block page only
    // ever displayed a generic hardcoded sentence, so a user never saw
    // WHY a site was flagged. `reason` (backend's plain-language "why",
    // set for every unsafe stage - see app/main.py's _unsafe_reason) is
    // now read and rendered by warning.js/warning.html.
    reason: result.reason || "",
    // Set whenever the backend could identify a real brand this URL
    // resembles - from the typosquat stage directly, or as an advisory
    // guess for blocklist/model-stage results (see app/main.py's
    // CheckResponse.legit_domain and _advisory_legit_domain). Lets
    // warning.js offer "go to the real site" instead of ever sending the
    // user to the flagged URL itself - there is no button path that
    // navigates to blockedUrl.
    legit_domain: result.legit_domain || "",
  });
  const warningUrl = chrome.runtime.getURL(`warning.html?${params.toString()}`);
  safeTabCall(chrome.tabs.update(tabId, { url: warningUrl }));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "MANUAL_CHECK") {
    // Found via real user testing: this handler had no scheme filtering,
    // unlike onBeforeNavigate. A manual check triggered while a tab was
    // on a browser-internal page (chrome://newtab/, chrome://settings,
    // etc.) sent that URL to the backend, which - never seeing anything
    // like it in training - confidently called it unsafe, and cached it.
    if (IGNORED_SCHEMES.some((s) => message.url.startsWith(s))) {
      sendResponse({ ok: false, error: "This type of page cannot be checked." });
      return true;
    }
    getSettings().then(async (settings) => {
      try {
        const result = await checkUrlWithBackend(message.url, settings.backendUrl);
        await setCacheEntry(message.url, result);
        sendResponse({ ok: true, result });
      } catch (err) {
        sendResponse({ ok: false, error: String(err) });
      }
    });
    return true; // async response
  }
});
