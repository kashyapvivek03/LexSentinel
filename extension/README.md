# Phishing URL Checker - Browser Extension

## What each file does
- `manifest.json` — Manifest V3 config: permissions, background worker, popup, icons
- `background.js` — the actual logic: intercepts navigation, calls the backend, caches results, redirects to the warning page
- `warning.html` / `warning.js` — the interstitial shown when a site is flagged unsafe
- `popup.html` / `popup.js` — click the toolbar icon to see the current page's status or trigger a manual check
- `options.html` / `options.js` — set the backend URL here

## Load it (personal use - no Chrome Web Store needed)
1. Open `chrome://extensions`
2. Toggle **Developer mode** on (top right)
3. Click **Load unpacked**
4. Select this `extension` folder
5. Click the extension's icon → **Settings** (or right-click the icon → Options) and set your backend URL (see main README for deploying it)

## Honest limitation worth knowing
Manifest V3 doesn't let extensions pause navigation while an async check runs (that capability was removed for most extensions). This checks and redirects **as fast as possible** after navigation starts, not with a hard guarantee that zero content ever renders. With a warm backend this is usually imperceptible; with a cold-started free-tier backend, you may occasionally see a brief flash of the real page before the redirect lands. This is a platform constraint every Manifest V3 safe-browsing-style extension faces, not something specific to this build.

## Testing it
Try navigating to `https://www.sbl.co.in/` (should redirect to the warning page - typosquat detection) and a real site like `https://www.google.com/` (should load normally, badge stays clear).
