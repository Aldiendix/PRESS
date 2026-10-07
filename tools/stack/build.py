"""Build the combined single-file submission (tree cut -> language split -> singleton ranker -> link rule) with the
ANS / enumerative table container.  Drop-in for tools/build.py (same --table/--fw/--fc/--d/--idf/--social/--title/
--refine/--out arguments; --mark / --rejoin are accepted but must be 0).

  python build.py --table cache/W_x --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.2)" \
      --title "(35, 0.4, 4, 80, 0.3)" --refine "(90, 10.0, 0.75)" --out solution.py           (stage 1, readable template)
  python build.py --table cache/W_x --mini --out solution.py                                  (stage 2, mini_template.py)

Other switches: --rank_social / --rank_title "None" (density singletons, v4.1), --lang 0, --link 0, --bits 19
(code points U+40000 + (v ^ mask), mask chosen so that no noncharacter is written; server acceptance untested),
--format rice --template ref_template.py (old Rice container, for equivalence tests only).
The built file is imported and its decoded table is compared bit for bit with the source table.
"""
import argparse, ast, importlib.util, io, lzma, os, sys, tokenize, unicodedata
from math import comb
import numpy as np

LIMIT, M_BITS, CH = 50000, 14, 256
M = 1 << M_BITS
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = {15: 0x4E00, 19: 0x40000}
# weights17.json keys social_all / arxiv_all (17 features in the order of the template), 2 significant digits
RANK_SOCIAL = "(-.098,.14,.86,.2,.37,-2,-.44,.078,-1.1,.37,-1.5,-1.6,2.7,.73,-15,.14,-.26)"
RANK_TITLE = "(3.2,-.33,2.8,-.014,-.065,.41,.12,-.56,.44,.63,-8.2,-.51,-.46,-.35,4.2,.29,.29)"


def strip(src):
    """Drop comments, docstrings and blank lines and join continuation lines (keeps the code itself untouched)."""
    out, prev = [], tokenize.INDENT
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, t in enumerate(toks):
        if t.type == tokenize.COMMENT:
            continue
        if t.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT) and toks[i + 1].type == tokenize.NEWLINE and t.string.startswith(('"""', "'''")):
            out.append((tokenize.NAME, "pass", t.start, t.end, t.line)); prev = t.type; continue
        out.append(t); prev = t.type
    code = tokenize.untokenize(out)
    lines = [l.rstrip() for l in code.split("\n")]
    lines = [l for l in lines if l.strip() and l.strip() != "pass"]
    lines = [" " * ((len(l) - len(l.lstrip(" "))) // 4) + l.lstrip(" ") for l in lines]   # 1-space indentation
    code = "\n".join(lines) + "\n"
    depth, cont = 0, set()
    for t in tokenize.generate_tokens(io.StringIO(code).readline):
        if t.type == tokenize.OP:
            depth += (t.string in "([{") - (t.string in ")]}")
        elif t.type == tokenize.NL and depth > 0:
            cont.add(t.start[0])
    joined = []
    for i, l in enumerate(lines):
        if i in cont:
            a, b = joined[-1], l.lstrip(" ")
            joined[-1] = a + ("" if a[-1] in "([{" or b[0] in ")]}" else " ") + b
        else:
            joined.append(l)
    out = "\n".join(joined) + "\n"
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(code)), "joining lines changed the code"
    return out


def quantise(cnt):
    """Integer frequencies summing to M, >= 1 wherever cnt > 0."""
    cnt = np.asarray(cnt, np.float64); p = cnt / cnt.sum(); f = np.maximum(np.floor(p * M), (cnt > 0)).astype(np.int64)
    while f.sum() != M:
        d = M - f.sum()
        gain = np.where(cnt > 0, cnt * (np.log2(f + np.sign(d)) - np.log2(f)) if d > 0 else np.where(f > 1, -cnt * (np.log2(f) - np.log2(np.maximum(f - 1, 1))), -np.inf), -np.inf)
        f[int(np.argmax(gain))] += int(np.sign(d))
    return f


def container(q, s16, rl):
    """Bit string: 16b #symbols | 14b frequencies | D x 16b float16 column scales | 8b (min level + 128) |
    per 256-row chunk: 20b length L, L-bit ANS state.  Row symbol y = 4 * count + (level - min level), 0 for an empty row."""
    F, D = q.shape; assert F % CH == 0
    nzm = q != 0; kr = nzm.sum(1); lmin = int(rl[kr > 0].min()); lev = np.where(kr > 0, rl - lmin, 0); assert lev.max() <= 3 and lev.min() >= 0
    y = np.where(kr > 0, 4 * kr + lev, 0)
    f = quantise(np.bincount(y, minlength=4 * (int(kr.max()) + 1))); c = np.concatenate([[0], np.cumsum(f)[:-1]])
    st = [len(f), 16]
    def put(v, w): assert 0 <= int(v) < 1 << w; st[0] = st[0] << w | int(v); st[1] += w
    for v in f: put(v, 14)
    for v in s16.view(np.uint16): put(v, 16)
    put(lmin + 128, 8)
    for a in range(0, F, CH):
        x = 1
        for i in range(a + CH - 1, a - 1, -1):          # ANS is last-in first-out: encode rows (and fields) in reverse
            kk = int(kr[i])
            if kk:
                cols = np.flatnonzero(nzm[i]); sg = 0
                for t, j in enumerate(cols): sg |= int(q[i, j] < 0) << t
                x = ((x << kk) | sg) * comb(D, kk) + sum(comb(int(j), t + 1) for t, j in enumerate(cols))
            fy, cy = int(f[y[i]]), int(c[y[i]]); x = (x // fy) * M + cy + (x % fy)
        put(x.bit_length(), 20); put(x, x.bit_length())
    return st[0], st[1]


def ans_chars(q, s16, rl, bits):
    n, nb = container(q, s16, rl); k = -(-nb // bits); n <<= k * bits - nb; mk = (1 << bits) - 1
    g = [(n >> (bits * (k - 1 - i))) & mk for i in range(k)]; mask = 0
    if bits == 19:   # a group becomes a noncharacter U+xFFFE / U+xFFFF iff its bits 1..15 equal ~mask's: use a pattern that does not occur
        used = {v >> 1 & 0x7FFF for v in g}; mask = (~next(p for p in range(1, 0x8000) if p not in used) & 0x7FFF) << 1
    return "".join(chr(BASE[bits] + (v ^ mask)) for v in g), nb, mask


def rice_chars(q, s, rows_r):
    sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "..", "tools")]; from pack import to_chars
    nz = q != 0; pos = np.flatnonzero(nz.ravel()); gaps = np.diff(pos, prepend=-1) - 1; k = len(pos); best = None
    for rice in range(1, 9):
        quo, rem = gaps >> rice, gaps & ((1 << rice) - 1)
        un = np.ones(int(quo.sum()) + k, np.uint8); un[np.cumsum(quo + 1) - 1] = 0
        rbits = ((rem[:, None] >> np.arange(rice - 1, -1, -1)) & 1).astype(np.uint8).ravel()
        ub = np.packbits(un).tobytes(); body = ub + np.packbits(rbits).tobytes()
        if best is None or len(body) < len(best[1]): best = (rice, body, len(ub))
    rice, body, ul = best
    raw = k.to_bytes(4, "big") + ul.to_bytes(4, "big") + body + np.packbits(q.ravel()[pos] < 0).tobytes() + s.astype(np.float16).tobytes()
    raw += lzma.compress(np.round(2 * np.log2(rows_r)).astype(np.int8).tobytes(), preset=9 | lzma.PRESET_EXTREME)
    return to_chars(raw), rice, len(raw) * 8


def settings(p, w):
    t = ast.literal_eval(p)
    if len(t) == 5 and w != "None":
        p = p.strip()[:-1].rstrip(", ") + ", " + w + ")"; t = ast.literal_eval(p)
    assert len(t) == 5 or (len(t) == 6 and len(t[5]) == 17), "settings: 5 numbers, plus the 17 ranker weights (the templates compute 17 features)"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--fw", type=int, default=8192); ap.add_argument("--fc", type=int, default=16384)
    ap.add_argument("--d", type=int, default=96); ap.add_argument("--idf", type=int, default=1); ap.add_argument("--mark", type=int, default=0); ap.add_argument("--rejoin", default="0")
    ap.add_argument("--refine", default="(90, 10.0, 0.75)"); ap.add_argument("--social", default="(20, 0.4, 4, 120, 0.2)"); ap.add_argument("--title", default="(35, 0.4, 4, 80, 0.3)")
    ap.add_argument("--rank_social", default=RANK_SOCIAL); ap.add_argument("--rank_title", default=RANK_TITLE)
    ap.add_argument("--lang", type=int, default=1); ap.add_argument("--link", type=int, default=20)
    ap.add_argument("--bits", type=int, default=15, choices=(15, 19)); ap.add_argument("--format", default="ans", choices=("ans", "rice"))
    ap.add_argument("--mini", action="store_true", help="stage 2: mini_template.py (all settings hard-coded in it)")
    ap.add_argument("--template", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.mark == 0 and float(a.rejoin) == 0, "the template has no MARK / REJOIN code"
    q = np.load(a.table + ".q.npy"); s = np.load(a.table + ".s.npy").astype(np.float32); r = np.load(a.table + ".r.npy")
    assert q.shape == (a.fw + a.fc, a.d), q.shape
    s16 = s.astype(np.float16); rl = np.round(2 * np.log2(r)).astype(np.int64)
    name = "mini_template.py" if a.mini else "solution_template.py"   # looked up next to this file, then in ../src (tools/ layout)
    tpl = open(a.template or next(x for x in (os.path.join(HERE, name), os.path.join(HERE, "..", "src", name)) if os.path.exists(x)), encoding="utf8").read()
    if a.format == "rice":
        assert a.bits == 15 and not a.mini; tab, rice, nb = rice_chars(q, s, r); mask = 0
    else:
        tab, nb, mask = ans_chars(q, s16, rl, a.bits)
    base = str(BASE[a.bits]) + ("^%d" % mask if mask else "")
    if a.mini:
        assert (a.fw, a.fc, a.d, a.idf) == (8192, 16384, 96, 1), "mini_template.py is hard-coded for the 24,576 x 96 table"
        src, sub = tpl.rstrip("\n") + "\n", (("__BITS__", a.bits), ("__BASE__", base), ("__TABLE__", tab))
    else:
        src = strip(tpl)
        sub = [("__FW__", a.fw), ("__FC__", a.fc), ("__D__", a.d), ("__IDF__", a.idf), ("__P_SOCIAL__", settings(a.social, a.rank_social)),
               ("__P_TITLE__", settings(a.title, a.rank_title)), ("__REFINE__", a.refine), ("__LANG__", a.lang), ("__LINK__", a.link), ("__TABLE__", tab)]
        sub += [("__ROWS__", 1), ("__RICE__", rice)] if a.format == "rice" else [("__BITS__", a.bits), ("__BASE__", base)]
    for k, v in sub:
        assert src.count(k) == (2 if a.mini and k == "__BITS__" else 1), k + ": unexpected number of occurrences in the template"
        src = src.replace(k, str(v))
    assert "__" not in src.replace("__name__", "").replace("__main__", ""), "unfilled placeholder"
    open(a.out, "w", encoding="utf8").write(src)
    n = len(src)
    spec = importlib.util.spec_from_file_location("built", a.out); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ref = q.astype(np.float32) * s16.astype(np.float32) * np.exp2(np.where((q != 0).any(1), rl, 0).astype(np.float32) / 2)[:, None]
    W = mod.table(); ok = np.array_equal(W, ref) if a.format == "ans" else bool(np.array_equal((W != 0), (q != 0)))
    bad = sum((ord(c) & 0xFFFE) == 0xFFFE or 0xD800 <= ord(c) < 0xE000 for c in src)
    stable = all(unicodedata.normalize(f, src) == src for f in ("NFC", "NFKC"))
    print(f"{a.out}: {n} chars (limit {LIMIT}, free {LIMIT - n}), code {n - len(tab)}, table {len(tab)} chars = {nb} bits at {a.bits} bits/char, non-zero {float((q != 0).mean()):.4f}, "
          f"utf-8 {len(src.encode())} bytes, decoded table identical: {ok}, noncharacters {bad}, normalisation-stable {stable}")
    assert ok, "decoded table differs"
    assert bad == 0 and (stable or a.bits == 15), "unsafe characters"
    assert n <= LIMIT or os.environ.get("FORCE"), "over the character limit"


if __name__ == "__main__":
    main()
