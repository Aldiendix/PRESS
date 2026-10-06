"""Download Apex text-clustering round data, leaderboards and revealed code via the public dashboard proxy."""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

B = "https://apex.macrocosmos.ai/api/mainnet"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CID = 10


def get(path, raw=False, **params):
    url = f"{B}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404):
                raise RuntimeError(f"{url}: HTTP {e.code} {e.read()[:200]!r}") from None
            err = e
            time.sleep(2 + 3 * attempt)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"{url}: {err}")


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data if isinstance(data, (bytes, str)) else json.dumps(data, indent=1))


def fetch_code(hotkey, rnd, ver):
    out, idx = "", 0
    while True:
        r = get("dashboard/code", competition_id=CID, round_number=rnd, hotkey=hotkey, version=ver, start_idx=idx)
        chunk = r.get("code") or ""
        out += chunk
        nxt = r.get("next_idx")
        if not chunk or nxt in (None, -1) or nxt <= idx:
            return out, r
        idx = nxt


def fetch_file(sub_id, kind, name):
    out, idx = "", 0
    while idx is not None:
        r = get("dashboard/file", submission_id=sub_id, file_type=kind, file_name=name, start_idx=idx, reverse="false")
        out += r.get("data") or ""
        idx = (r.get("pagination") or {}).get("next_start_idx")
    return out


def fetch_round(rnd, top_n=5):
    d = os.path.join(ROOT, f"round_{rnd:04d}")
    miners = get(f"dashboard/competitions/{CID}/miners", round_number=rnd)["miners"]
    subs = get(f"dashboard/competitions/{CID}/submissions", round_number=rnd)["miners"]
    save(os.path.join(d, "miners.json"), miners)
    save(os.path.join(d, "submissions.json"), subs)
    ranked = sorted([s for s in subs if s.get("score") is not None], key=lambda s: -s["score"])
    print(f"round {rnd}: {len(subs)} submissions; top:", [(s["hotkey"][:8], s["version"], round(s["score"], 4)) for s in ranked[:top_n]])
    keys = None
    for s in ranked[: max(top_n, 1)]:
        tag = f"{s['hotkey'][:8]}_v{s['version']}"
        det = get(f"dashboard/competitions/{CID}/submissions/{s['hotkey']}/{rnd}/{s['version']}")
        save(os.path.join(d, "subs", tag, "detail.json"), det)
        keys = keys or (det.get("eval_metadata") or {}).get("details", {}).get("data_keys")
        if top_n and det.get("revealed"):
            try:
                code, _ = fetch_code(s["hotkey"], rnd, s["version"])
                save(os.path.join(d, "subs", tag, "solution.py"), code)
            except Exception as e:  # noqa: BLE001
                print("  code fail", tag, e)
        for kind, paths in ((det.get("eval_file_paths") or {}) if top_n else {}).items():
            for p in paths if isinstance(paths, list) else [paths]:
                fn = os.path.basename(p)
                try:
                    save(os.path.join(d, "subs", tag, fn), fetch_file(det["id"], kind, fn))
                except Exception as e:  # noqa: BLE001
                    print("  file fail", tag, fn, str(e)[:100])
    for i, k in enumerate(keys or []):
        fn = os.path.basename(k)
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            save(p, get("download", raw=True, key=k, filename=fn))
        print("  data", fn, os.path.getsize(p))


if __name__ == "__main__":
    save(os.path.join(ROOT, "competition.json"), get(f"dashboard/competitions/{CID}"))
    save(os.path.join(ROOT, "top_scores.json"), get(f"dashboard/competitions/{CID}/top-scores"))
    args = sys.argv[1:]
    top_n = 0 if "--data-only" in args else 6
    for r in [int(a) for a in args if a.isdigit()]:
        try:
            fetch_round(r, top_n)
        except Exception as e:  # noqa: BLE001
            print("round", r, "failed:", str(e)[:200])
