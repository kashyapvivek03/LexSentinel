"""
core/typosquat.py
==================
Catches the exact gap flagged during testing: "sbl.co.in" (one character
off from "sbi.co.in") was scored SAFE at 0.06% phishing probability,
because it's structurally clean - no suspicious keywords, no odd TLD, no
obfuscation. The ONLY way to catch it is recognizing it's a near-miss of a
protected brand's domain. That's a high-precision pattern, better done as
an explicit, deterministic rule than hoped for from a handful of ML
training examples - false positives here (flagging a genuinely unrelated
domain) are cheap to avoid with a tight distance threshold, and false
negatives (missing a typosquat) are exactly what this exists to prevent.

Reuses the allowlist as the protected-brand reference set, so there's one
list to maintain, not two (see core/lists.py's staleness-risk fix - the
same principle applies here: don't hardcode a second brand list in code).

IMPORTANT - how brand cores are extracted:
The original version took host.split(".")[0] - the first DNS label of the
WHOLE hostname - as "the core" to compare, with no awareness of where the
registrable domain actually starts. This caused a real, confirmed bug:
"mail.chase.com" extracted core "mail", which sits 1 edit from the
allowlisted "gmail.com"'s core "gmail" - flagging any company's ordinary
mail subdomain as a Gmail impersonation attempt.

The fix is NOT to adopt a general public-suffix-list library (tldextract
was tried and rejected: its bundled PSL doesn't recognize "bank.in" as a
compound suffix, so it would mis-parse our OWN allowlist entry
"icici.bank.in" as domain="bank"/subdomain="icici" - backwards). Instead,
since we always know the EXACT structure of each protected domain we're
comparing against (they're literal strings in config/allowlist.json), we
size-match: take the host's last N labels, where N is THAT SPECIFIC
protected domain's own label count. This correctly extracts "chase" (not
"mail") when comparing "mail.chase.com" against 2-label "gmail.com", and
correctly leaves "icici.bank.in" (3 labels) alone when comparing against
other 3-label protected domains, without needing any general suffix data.

This also fixes a second, related gap: a brand name used as a SUBDOMAIN
PREFIX to impersonate ("irs.mynewsblog.net") wasn't caught, because the
old distance-based check required distance EXACTLY 1 for short cores,
excluding an exact match (distance 0). The new subdomain-prefix check
looks at exactly this case: any label BEFORE the size-matched suffix that
EXACTLY equals a protected core is flagged immediately - high precision,
since no legitimate business coincidentally names a subdomain "irs" or
"paypal".
"""
from __future__ import annotations
import re
import unicodedata
from core.lists import _load, is_allowlisted  # reuse the same cached JSON loader
from core.wordplay import normalize_confusables, count_confusable_chars, GENERIC_SUSPICIOUS_TERMS
from core.features import COMMON_TLDS, _safe_urlparse


# Protected cores that are ALSO common, legitimate subdomain labels on
# unrelated real domains: product names deployed across many organizations
# (outlook.live.com is Microsoft's own webmail; office.<company>.com and
# zoom.<university>.edu are standard IT setups) and geographic prefixes
# (usa.philips.com, usa.visa.com, usa.kaspersky.com are all real). Found
# via confirmed live false positives: for these cores, an exact
# prefix-label match alone is NOT the high-precision evidence the
# subdomain-prefix check assumes for e.g. "paypal" or "irs" - so Check 1
# additionally requires a corroborating signal (suspicious term in the
# path/query or in another host label, or an unusual TLD) before flagging.
# A real attack essentially always pairs the brand label with a lure word
# or cheap TLD ("outlook.secure-signin.top"); a bare, clean host is far
# more likely a legitimate product/regional subdomain.
# (Cores shorter than 3 chars - "id" from id.me, "x" from x.com - are
# excluded from ALL checks by the length guard in find_typosquat_match;
# id.atlassian.com and x.<anything> were confirmed false positives too.)
GENERIC_PREFIX_CORES = {"usa", "outlook", "office", "zoom"}


def _other_host_labels_suspicious(host: str, matched_core: str) -> bool:
    """True if any host label/sub-token OTHER than the matched brand label
    is a generic suspicious term after confusable normalization - e.g.
    'outlook.secure-signin.com' -> ['secure', 'signin'] both hit. Used as
    corroboration for GENERIC_PREFIX_CORES, complementing
    _has_corroborating_signal (which only looks at path/query + TLD and
    misses lure words placed in the domain itself)."""
    tokens = re.split(r"[.\-_]", normalize_confusables(host))
    return any(t in GENERIC_SUSPICIOUS_TERMS for t in tokens if t and t != matched_core)


def _damerau_levenshtein(a: str, b: str) -> int:
    """Like Levenshtein, but an adjacent-character transposition (e.g.
    'flipkart' -> 'filpkart', swapping 'li' to 'il') counts as ONE edit,
    not two. Plain Levenshtein misses exactly this class of typosquat -
    found via testing: 'filpkart.com' scored safe because standard edit
    distance to 'flipkart' is 2, past the distance-1 threshold, even
    though a human reads it as a single obvious swap."""
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    d = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        d[i][0] = i
    for j in range(len_b + 1):
        d[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[len_a][len_b]


def _levenshtein_no_transposition(a: str, b: str) -> int:
    """Plain Levenshtein (substitution/insertion/deletion only, no
    transposition credit) - used ALONGSIDE _damerau_levenshtein to detect
    whether a transposition was actually involved in reaching a given
    distance. If this is higher than the Damerau distance, a transposition
    was used - see _has_corroborating_signal's caller for why that
    matters: a transposition-involving distance-2 match (filpcart ->
    flipkart: swap + substitute) is a much stronger deliberate-typosquat
    signal than a pure-substitution distance-2 match (redfin/reddit,
    shopify/spotify - two coincidentally-placed different letters), so
    only the latter needs a corroborating keyword/TLD before flagging."""
    if a == b:
        return 0
    len_b = len(b)
    prev = list(range(len_b + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len_b
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _normalize_host(url: str) -> str:
    if "://" not in url:
        url = "http://" + url
    host = _safe_urlparse(url).hostname or ""
    # NFKC normalization folds fullwidth Unicode forms (e.g. 'ａ'-'ｚ') to
    # their ASCII equivalents essentially for free - closes a real gap
    # (fullwidth Unicode wasn't in HOMOGLYPH_MAP) without hand-maintaining
    # a bigger confusables table.
    host = unicodedata.normalize("NFKC", host)
    return host[4:] if host.startswith("www.") else host


def _size_matched_core_and_prefix(host: str, protected: str) -> tuple[str, list[str]] | None:
    """For a SPECIFIC protected domain, splits the host into (a) the core
    to compare for registrable-domain typosquatting, sized to match that
    protected domain's own label count, and (b) any leading labels beyond
    that, checked separately for subdomain-prefix abuse. Returns None if
    the host has fewer labels than the protected domain (can't meaningfully
    compare)."""
    host_labels = host.split(".")
    protected_labels = protected.split(".")
    n = len(protected_labels)
    if len(host_labels) < n:
        return None
    suffix_slice = host_labels[-n:]
    prefix_labels = host_labels[:-n] if len(host_labels) > n else []
    return suffix_slice[0], prefix_labels


def _has_corroborating_signal(url: str, host: str) -> bool:
    """For the fuzzy distance-2 branch only (5+ char cores): require a
    suspicious keyword in the path/query OR an unusual TLD alongside the
    near-miss, before treating it as a match. Found via 100K-URL
    evaluation: without this, coincidental collisions between distinct
    real companies whose names happen to sit distance-2 apart
    (redfin.com/reddit.com, shopify.com/spotify.com, slate.com/slack.com,
    usbank.com/yesbank.in) get flagged with no other evidence. Every
    actual reported attack in this distance class (filpkart.com,
    filpcart.com) carried a suspicious path or was tested standalone
    without this gate, but a REAL attack of this shape would essentially
    always pair the near-miss domain with a credential-harvesting path or
    a free/unusual TLD - a bare, path-less near-miss with an ordinary TLD
    and no suspicious wording is far more likely a coincidental real
    company than an active attack."""
    parsed = _safe_urlparse(url if "://" in url else "http://" + url)
    path_query = f"{parsed.path or ''} {parsed.query or ''}".lower()
    normalized_path_query = normalize_confusables(path_query)
    has_keyword = any(
        term in path_query or term in normalized_path_query
        for term in GENERIC_SUSPICIOUS_TERMS
    )
    tld = host.split(".")[-1] if "." in host else host
    has_unusual_tld = tld not in COMMON_TLDS
    return has_keyword or has_unusual_tld


def find_typosquat_match(url: str, max_distance: int = 2, require_corroboration: bool = True) -> str | None:
    """Returns the protected domain this URL suspiciously resembles, or
    None. Two independent checks, both against every protected domain:
    (1) registrable-core typosquat (near-miss of the actual domain), and
    (2) subdomain-prefix abuse (protected brand name used as a label
    before the real registrable domain, e.g. 'irs.mynewsblog.net').

    require_corroboration=True (default) is the verdict-deciding mode: a
    distance-2, non-transposition near-miss (e.g. 'arnazon' vs 'amazon')
    needs a suspicious path/query keyword or unusual TLD alongside it
    before this counts as a match - see _has_corroborating_signal's
    docstring for why (avoids flagging coincidental real-company
    collisions like redfin/reddit). That gate exists to protect the
    unsafe/safe VERDICT from false positives.

    require_corroboration=False skips that gate for Check 2's fuzzy
    branch only (Check 1's GENERIC_PREFIX_CORES gate and Check 4's
    combosquat gate are unaffected - those protect against different,
    still-live FP classes). Intended ONLY for suggesting a redirect target
    on a URL some OTHER stage has already independently flagged unsafe
    (see app/main.py's advisory legit_domain lookup) - at that point the
    site is being blocked regardless, so a looser brand-resemblance check
    just changes which real site gets suggested, never whether the user
    is blocked."""
    host = _normalize_host(url)
    if not host:
        return None
    if is_allowlisted(url):
        return None  # a real protected domain is never "its own typosquat"

    # TODO (deferred - a secondary recommendation from this project's own
    # design notes):
    # at Tranco-Top-1M allowlist scale, reusing the whole allowlist as the
    # typosquat reference set means every check does a million-entry loop
    # with an O(len^2) edit-distance DP each - seconds per URL. Keep a
    # separate, small protected-brands list (top ~500 by attack value)
    # instead, once the allowlist actually grows that large. Not done now
    # since the current allowlist (~80 entries) doesn't need it yet, and
    # splitting the reference set is a real design decision (which ~500
    # brands?) better made when it's actually load-bearing.
    protected_domains = _load("allowlist")["domains"]

    for protected in protected_domains:
        if host == protected:
            continue  # exact match is legitimate, not a typosquat
        match = _size_matched_core_and_prefix(host, protected)
        if match is None:
            continue
        host_core, prefix_labels = match
        protected_core = protected.split(".")[0]

        # too short/generic to ever compare safely by ANY method (e.g. "co",
        # "id", "x"). Must come BEFORE Check 1: this guard used to sit after
        # it, so any host with an ordinary "id." or "x." subdomain was
        # flagged as impersonating id.me / x.com - confirmed live on
        # id.atlassian.com (Atlassian's real login domain).
        if len(protected_core) < 3:
            continue

        # --- Check 1: subdomain-prefix abuse (exact match only - high
        # precision, catches "irs.mynewsblog.net" style impersonation).
        # For cores that are also common legitimate subdomain labels
        # (GENERIC_PREFIX_CORES above), a corroborating signal is required
        # - otherwise outlook.live.com / usa.philips.com / zoom.<edu> get
        # flagged just for existing. ---
        if any(label == protected_core for label in prefix_labels):
            if protected_core not in GENERIC_PREFIX_CORES:
                return protected
            if (_has_corroborating_signal(url, host)
                    or _other_host_labels_suspicious(host, protected_core)):
                return protected

        # --- Check 2: registrable-core typosquat (fuzzy, same rules as
        # before, now operating on the CORRECTLY size-matched core).
        # This branch used to `continue` past Check 3 on
        # every branch, making Check 3 structurally unreachable for any
        # protected core of length <=4 (sbi, rbi, irs, usa, nih, ajio,
        # jio) - confirmed live: '5b!' normalizes exactly to 'sbi' under
        # this file's own LEETSPEAK_MAP but was silently missed. Fixed by
        # falling through to Check 3 instead of skipping to the next
        # protected domain. ---
        if len(protected_core) <= 4:
            if len(host_core) == len(protected_core):
                if _damerau_levenshtein(host_core, protected_core) == 1:
                    return protected
        else:
            if abs(len(host_core) - len(protected_core)) <= 1:
                distance = _damerau_levenshtein(host_core, protected_core)
                if distance == 1:
                    return protected  # distance-1 on a long core is unambiguous, no gate needed
                if distance <= max_distance:
                    if not require_corroboration:
                        return protected
                    transposition_involved = (
                        _levenshtein_no_transposition(host_core, protected_core) > distance
                    )
                    if transposition_involved or _has_corroborating_signal(url, host):
                        return protected

        # --- Check 3: leetspeak/homoglyph normalized exact match - now
        # ALWAYS reached (see comment above), regardless of what Check 2
        # decided for this protected domain. ---
        if count_confusable_chars(host_core) > 0:
            if normalize_confusables(host_core) == protected_core:
                return protected

        # --- Check 4: combosquatting - brand name embedded as ONE
        # sub-token within a longer hyphenated compound, e.g.
        # "paypal-secure-verify.com" or "g00gle-signin.com" or
        # "arnazon-orders.com". Found via real-user testing: every check
        # above treats host_core as one indivisible string, so a compound
        # label is always far longer than a bare protected core and gets
        # excluded by the length-difference guard before any real
        # comparison happens - "paypal-secure-verify" (20 chars) vs
        # "paypal" (6 chars) never even reaches a distance calculation.
        #
        # A sub-token match is WEAKER evidence than a whole-domain match
        # (a bare exact match to the whole registrable domain core is
        # rare/deliberate; a common word showing up as ONE part of a
        # compound is not - "apple", "usa", and "jio" are all protected
        # cores that are also ordinary words, and "team-usa-sports.org"
        # or "apple-pie-recipes.com" must not be flagged just for
        # containing them). So corroboration (a suspicious term as
        # another sub-token, or an unusual TLD) is REQUIRED here even for
        # an exact sub-token match - unlike Check 2's distance-1 case,
        # which is unambiguous enough to skip the gate.
        if "-" in host_core or "_" in host_core:
            sub_tokens = re.split(r"[-_]", host_core)
            # A suspicious word as ANOTHER sub-token in the SAME compound
            # label is the most direct corroboration there is - found via
            # testing that _has_corroborating_signal alone (path/query +
            # TLD) misses this class entirely, since "paypal-secure-verify.com"
            # has no path and an ordinary .com TLD; the suspicious words
            # are sitting right there in the domain, not the path.
            other_tokens_suspicious = any(
                t.lower() in GENERIC_SUSPICIOUS_TERMS for t in sub_tokens
            )
            for token in sub_tokens:
                if len(protected_core) <= 4:
                    # Short cores (usa, irs, jio, sbi...) are also common
                    # words/abbreviations - stay strict, same as Check 2.
                    near_miss = (len(token) == len(protected_core)
                                 and _damerau_levenshtein(token, protected_core) == 1)
                else:
                    near_miss = (abs(len(token) - len(protected_core)) <= 1
                                 and _damerau_levenshtein(token, protected_core) <= max_distance)
                token_matches = (
                    token == protected_core
                    or near_miss
                    or (count_confusable_chars(token) > 0
                        and normalize_confusables(token) == protected_core)
                )
                if token_matches and (other_tokens_suspicious or _has_corroborating_signal(url, host)):
                    return protected

    return None
