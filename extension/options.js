async function load() {
  const { backendUrl } = await chrome.storage.sync.get("backendUrl");
  document.getElementById("backendUrl").value = backendUrl || "https://phishing-detector-f65f.onrender.com";
}

document.getElementById("saveBtn").addEventListener("click", async () => {
  let url = document.getElementById("backendUrl").value.trim().replace(/\/+$/, "");
  // Validate before saving: a non-URL value (typo, missing scheme) used
  // to be stored silently, after which every check quietly failed open -
  // badge "?" on every site with no hint the settings were the problem.
  let valid = false;
  try {
    const u = new URL(url);
    valid = u.protocol === "http:" || u.protocol === "https:";
  } catch {}
  const errMsg = document.getElementById("errMsg");
  if (!valid) {
    errMsg.style.display = "inline";
    return;
  }
  errMsg.style.display = "none";
  await chrome.storage.sync.set({ backendUrl: url });
  const msg = document.getElementById("savedMsg");
  msg.style.display = "inline";
  setTimeout(() => (msg.style.display = "none"), 1500);
});

load();
