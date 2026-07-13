const params = new URLSearchParams(window.location.search);
const blockedUrl = params.get("url") || "";

document.getElementById("blockedUrl").textContent = blockedUrl;

// 2026-07 audit fix: background.js has always built `reason` (and
// note/stage/confidence before it) into these query params, but nothing
// here ever read them - the block page showed only a generic hardcoded
// sentence, never the actual reason a site was flagged. `.textContent`,
// not innerHTML, so this is safe against a reason string containing
// markup without needing a separate escape helper (same end result as
// popup.js's escapeHtml(), just via the DOM API directly since this is
// the only dynamic text on this page).
const reason = params.get("reason") || "";
if (reason) {
  const reasonBox = document.getElementById("reasonBox");
  reasonBox.textContent = reason;
  reasonBox.classList.add("visible");
}

function goBack() {
  if (window.history.length > 1) {
    window.history.back();
  } else {
    window.close();
  }
}

// There is deliberately no path from this page to the flagged URL itself
// - once a site is flagged unsafe, the only useful actions are going back
// or (for a typosquat, where we know which real brand is being
// impersonated) going to that brand's actual site. See app/main.py's
// CheckResponse.legit_domain and core/typosquat.py's find_typosquat_match.
//
// legit_domain is only ever set by background.js from the backend's
// response (see redirectToWarning), and background.js only ever sets it
// from a typosquat-stage result, where find_typosquat_match() guarantees
// it's a literal entry from config/allowlist.json - never text built out
// of the flagged URL. Still, this page treats it as untrusted input (one
// compromised/misconfigured backend away from something unexpected - same
// posture as popup.js's escapeHtml comment) and validates it looks like a
// plain hostname before ever using it in a navigation target.
const HOSTNAME_RE = /^[a-z0-9]([a-z0-9-]{0,62}\.)+[a-z]{2,}$/i;
const legitDomainParam = params.get("legit_domain") || "";
const legitDomain = HOSTNAME_RE.test(legitDomainParam) ? legitDomainParam : "";

const actionBtn = document.getElementById("actionBtn");
if (legitDomain) {
  actionBtn.textContent = `Go to ${legitDomain}`;
  actionBtn.addEventListener("click", () => {
    window.location.href = `https://${legitDomain}`;
  });
} else {
  actionBtn.textContent = "Close warning";
  actionBtn.addEventListener("click", goBack);
}

document.getElementById("backBtn").addEventListener("click", goBack);
