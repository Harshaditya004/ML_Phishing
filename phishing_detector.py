import math
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "phishing_urls_dataset.csv"  # columns: id, url, label (1 = phishing), class
MODEL_FILE = "phishing_random_forest_model.pkl"
FEATURE_FILE = "phishing_url_features.csv"
RANDOM_STATE = 42
PHISHING_THRESHOLD = 0.70  # phishing score at or above this -> PHISHING
LEGIT_THRESHOLD = 0.40     # phishing score at or below this -> LEGITIMATE

# Allowlist of well-known legitimate registrable domains (demo safeguard).
TRUSTED_DOMAINS = {
    "youtube.com",
    "google.com",
    "microsoft.com",
    "github.com",
    "amazon.in",
    "amazon.com",
    "steampowered.com",
    "wikipedia.org",
    "wikimedia.org",
    "bbc.com",
    "bbc.co.uk",
    "stackoverflow.com",
    "stackexchange.com",
    "apple.com",
    "mozilla.org",
    "python.org",
    "linkedin.com",
    "reddit.com",
    "netflix.com",
    "paypal.com",
    "flipkart.com",
    "nytimes.com",
    "cloudflare.com",
}

# Hosts under a trusted domain where ANYONE can publish content. Phishing pages
# really are hosted here (the dataset has ~47 such google.com examples), so these
# are never auto-trusted and always go through the ML model.
USER_CONTENT_HOSTS = {
    "docs.google.com",
    "sites.google.com",
    "drive.google.com",
    "forms.google.com",
    "storage.googleapis.com",
    "script.google.com",
    "raw.github.com",
    "gist.github.com",
    "pages.github.com",
}

# Demo blocklist (project rule, not evidence of phishing).
BLOCKED_DOMAINS = {
    "123movies.com",
    "123movies.net",
    "123movies.to",
    "123moviesfree.net",
}

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "confirmation", "password", "payment",
    "billing", "bank", "wallet", "authenticate", "authentication", "restore",
    "suspended", "urgent", "claim", "free", "prize",
]

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
IP_RE = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


# ============================================================
# DOMAIN RULES
# ============================================================

def get_hostname(url):
    """Parse and normalise a hostname, including inputs without a scheme."""
    candidate = str(url).strip()
    if not SCHEME_RE.match(candidate):
        candidate = "https://" + candidate
    try:
        hostname = urlparse(candidate).hostname
    except ValueError:
        return ""
    return hostname.lower().rstrip(".") if hostname else ""


def domain_matches(hostname, domain):
    """Match a domain itself or a true subdomain, not a lookalike suffix."""
    return hostname == domain or hostname.endswith("." + domain)


def is_trusted_domain(url):
    hostname = get_hostname(url)
    if not hostname:
        return False
    if any(domain_matches(hostname, h) for h in USER_CONTENT_HOSTS):
        return False
    return any(domain_matches(hostname, d) for d in TRUSTED_DOMAINS)


def is_blocked_domain(url):
    hostname = get_hostname(url)
    return bool(hostname) and any(domain_matches(hostname, d) for d in BLOCKED_DOMAINS)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def split_url(url):
    """Return (url_without_scheme, hostname, path, query, port_present).

    The training corpus stores URLs WITHOUT a scheme, so the scheme is removed
    here. Otherwise a user typing "https://x.com/a" would get different length
    and slash features than the model saw in training.
    """
    url = str(url).strip()
    url = SCHEME_RE.sub("", url)
    try:
        parsed = urlparse("http://" + url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        has_port = parsed.port is not None
    except ValueError:  # defanged / malformed hosts, bad ports, bad IPv6
        host_part = url.split("/", 1)[0].split("?", 1)[0]
        hostname = host_part.split("@")[-1].lower()
        rest = url[len(host_part):]
        path = rest.split("?", 1)[0]
        query = rest.split("?", 1)[1] if "?" in rest else ""
        has_port = False
    return url, hostname.lower(), path, query, has_port


def extract_features(url):
    url, hostname, path, query, has_port = split_url(url)

    words = re.findall(r"[a-zA-Z0-9]+", url.lower())
    joined = " ".join(words)
    host_words = " ".join(re.findall(r"[a-zA-Z0-9]+", hostname))
    path_words = " ".join(re.findall(r"[a-zA-Z0-9]+", (path + " " + query).lower()))

    n = max(len(url), 1)
    digit_count = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    is_ip = bool(IP_RE.fullmatch(hostname))
    labels = hostname.split(".") if hostname else []
    tld = labels[-1] if len(labels) > 1 else ""
    host_digits = sum(c.isdigit() for c in hostname)
    last_segment = path.rstrip("/").rsplit("/", 1)[-1]

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),

        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersands": url.count("&"),
        "num_at": url.count("@"),
        "num_percent": url.count("%"),

        "num_digits": digit_count,
        "num_letters": letter_count,

        "has_ip_address": int(is_ip),
        "has_port": int(has_port),
        "subdomain_count": 0 if is_ip else max(0, len(labels) - 2),
        "path_depth": len([x for x in path.split("/") if x]),

        "digit_ratio": digit_count / n,
        "letter_ratio": letter_count / n,
        "special_char_ratio": sum(not c.isalnum() for c in url) / n,

        "suspicious_word_count": sum(w in joined for w in SUSPICIOUS_WORDS),
        "suspicious_in_host": sum(w in host_words for w in SUSPICIOUS_WORDS),
        "suspicious_in_path": sum(w in path_words for w in SUSPICIOUS_WORDS),
        "has_login": int("login" in joined or "signin" in joined),
        "has_verify": int("verify" in joined or "verification" in joined),
        "has_account": int("account" in joined),
        "has_password": int("password" in joined),
        "has_payment": int("payment" in joined or "billing" in joined),
        "has_security": int("security" in joined or "secure" in joined),

        "has_at_symbol": int("@" in url),
        "has_double_slash_path": int("//" in path),
        "has_hex_encoding": int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
        "has_long_hostname": int(len(hostname) > 30),
        "has_long_url": int(len(url) > 75),

        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "max_word_length": max((len(w) for w in words), default=0),

        # Added features
        "url_entropy": entropy(url),
        "hostname_entropy": entropy(hostname),
        "hostname_digit_count": host_digits,
        "hostname_hyphens": hostname.count("-"),
        "longest_label_length": max((len(x) for x in labels), default=0),
        "tld_length": len(tld),
        "starts_with_www": int(hostname.startswith("www.")),
        "path_has_script_ext": int(bool(re.search(r"\.(php|asp|aspx|jsp|cgi|exe|zip)$", last_segment, re.I))),
        "path_has_html_ext": int(bool(re.search(r"\.(html?|shtml)$", last_segment, re.I))),
        "path_is_root": int(path in ("", "/")),
        "last_segment_length": len(last_segment),
    }


# ============================================================
# DATA
# ============================================================

def load_dataset():
    try:
        df = pd.read_csv(DATASET)
    except FileNotFoundError:
        print(f"\nERROR: '{DATASET}' not found. Put it in the same folder as this script.")
        return None

    if not {"url", "label"}.issubset(df.columns):
        print("\nERROR: Dataset must contain 'url' and 'label' columns.")
        return None

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Same URL appearing in both train and test inflates every metric.
    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    if len(df) < before:
        print(f"Removed {before - len(df)} duplicate URLs ({before} -> {len(df)}).")
    return df


# ============================================================
# TRAIN
# ============================================================

def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def train_model():
    df = load_dataset()
    if df is None:
        return None

    print("\n" + "=" * 60)
    print("       PHISHING URL DETECTION MODEL")
    print("=" * 60)
    print("\nDataset size (after de-duplication):", len(df))

    print("\nExtracting URL features...")
    X = pd.DataFrame([extract_features(u) for u in df["url"]])
    y = df["label"]
    print("Number of features:", X.shape[1])

    # Group by hostname so the same site never appears in both train and test.
    groups = df["url"].map(get_hostname).replace("", np.nan).fillna(df["url"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))
    print("\nTraining Random Forest...")
    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("   MODEL RESULTS (host-grouped hold-out, no leakage)")
    print("=" * 60)
    acc = accuracy_score(y_test, pred)
    print(f"\nAccuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision : {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1-score  : {f1_score(y_test, pred, zero_division=0):.4f}")
    print("\nConfusion Matrix (rows = actual, cols = predicted):\n")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, target_names=["Legitimate", "Phishing"], zero_division=0))

    print("\n" + "=" * 60)
    print("   10-FOLD GROUPED CROSS-VALIDATION")
    print("=" * 60)
    scores = cross_val_score(make_model(), X, y, cv=GroupKFold(n_splits=10),
                             groups=groups, scoring="accuracy")
    print("\nFold accuracies:", np.round(scores, 4))
    print(f"Mean accuracy : {scores.mean():.4f}")
    print(f"Std deviation : {scores.std():.4f}")

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print("\n" + "=" * 60)
    print("         TOP IMPORTANT FEATURES")
    print("=" * 60 + "\n")
    print(importance.head(10).to_string(index=False))

    # Final model uses all data; the metrics above come from the hold-out/CV.
    final_model = make_model().fit(X, y)
    joblib.dump({"model": final_model, "feature_names": list(X.columns)}, MODEL_FILE)

    pd.concat([df.reset_index(drop=True), X.reset_index(drop=True)], axis=1).to_csv(
        FEATURE_FILE, index=False
    )
    print("\nModel saved as:", MODEL_FILE)
    print("Feature dataset saved as:", FEATURE_FILE)
    return {"model": final_model, "feature_names": list(X.columns)}


def load_saved_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("\nNo trained model found. Please train the model first.")
        return None


# ============================================================
# PREDICT
# ============================================================

def _banner():
    print("\n" + "=" * 60)
    print("                 URL RESULT")
    print("=" * 60 + "\n")


def predict_url(bundle, url):
    """Return (label, note) and print the result."""
    if is_blocked_domain(url):
        _banner()
        print("URL        :", url)
        print("Prediction : BLOCKLISTED")
        print("Confidence : Domain blocklist rule (not an ML score; not verified as phishing)")
        return "BLOCKLISTED", None

    if is_trusted_domain(url):
        _banner()
        print("URL        :", url)
        print("Prediction : LEGITIMATE")
        print("Confidence : Trusted-domain rule (not an ML confidence score)")
        return "LEGITIMATE", None

    model, names = bundle["model"], bundle["feature_names"]
    features = pd.DataFrame([extract_features(url)]).reindex(columns=names, fill_value=0)
    p_phish = float(model.predict_proba(features)[0][1])

    # Short homepage-style URLs carry little signal, so the middle band is
    # reported as uncertain instead of forcing a yes/no answer.
    if p_phish >= PHISHING_THRESHOLD:
        result, confidence = "PHISHING", p_phish * 100
    elif p_phish <= LEGIT_THRESHOLD:
        result, confidence = "LEGITIMATE", (1 - p_phish) * 100
    else:
        result, confidence = "UNCERTAIN", None

    _banner()
    print("URL        :", url)
    print("Prediction :", result)
    if confidence is None:
        print(f"Confidence : Phishing score {p_phish * 100:.1f}% (between {LEGIT_THRESHOLD:.0%} and "
              f"{PHISHING_THRESHOLD:.0%}); verify manually before trusting this link")
    else:
        print(f"Confidence : {confidence:.2f}%")
    return result, confidence


# ============================================================
# MENU
# ============================================================

def ask_and_predict(bundle):
    url = input("\nEnter URL to test: ").strip()
    if url:
        predict_url(bundle, url)
    else:
        print("Please enter a URL.")


def main():
    bundle = None
    while True:
        print("\n" + "=" * 60)
        print("        MACHINE LEARNING PHISHING DETECTOR")
        print("=" * 60)
        print("\n1. Train model\n2. Test a URL\n3. Train model and test a URL\n4. Exit\n")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            bundle = train_model()
        elif choice == "2":
            bundle = bundle or load_saved_model()
            if bundle is not None:
                ask_and_predict(bundle)
        elif choice == "3":
            bundle = train_model()
            if bundle is not None:
                ask_and_predict(bundle)
        elif choice == "4":
            print("\nExiting program...")
            break
        else:
            print("\nInvalid choice.\nPlease enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_model()
    else:
        main()
