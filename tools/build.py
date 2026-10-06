"""Build a single-file submission: pack a trained ternary table into src/solution_template.py.

Usage: python tools/build.py --table cache/W_x --fw 8192 --fc 4096 --d 48 --out dist/press_v1.py
       (expects <table>.q.npy = ternary int8 matrix, <table>.s.npy = column scales, optional <table>.r.npy = row scales)
"""
import argparse, ast, io, lzma, os, sys, tokenize
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack import compress, to_chars

LIMIT = 50000


def strip(src):
    """Drop comments, docstrings and blank lines (keeps the code itself untouched)."""
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
    ast.parse(code)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--fw", type=int, required=True); ap.add_argument("--fc", type=int, required=True)
    ap.add_argument("--d", type=int, required=True); ap.add_argument("--idf", type=int, default=0); ap.add_argument("--mark", type=int, default=0); ap.add_argument("--refine", default="None")
    ap.add_argument("--social", default="(20, 0.4, 4, 80, 0.2)"); ap.add_argument("--title", default="(35, 0.4, 4, 80, 0.3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q = np.load(a.table + ".q.npy"); s = np.load(a.table + ".s.npy").astype(np.float32)
    assert q.shape == (a.fw + a.fc, a.d), q.shape
    nz = q != 0
    pos = np.flatnonzero(nz.ravel()); gaps = np.diff(pos, prepend=-1) - 1; k = len(pos)
    best = None
    for rice in range(1, 9):   # Golomb-Rice: unary quotient + `rice` remainder bits per gap
        quo, rem = gaps >> rice, gaps & ((1 << rice) - 1)
        un = np.ones(int(quo.sum()) + k, np.uint8); un[np.cumsum(quo + 1) - 1] = 0
        rbits = ((rem[:, None] >> np.arange(rice - 1, -1, -1)) & 1).astype(np.uint8).ravel()
        ub = np.packbits(un).tobytes()
        body = ub + np.packbits(rbits).tobytes()
        if best is None or len(body) < len(best[1]):
            best = (rice, body, len(ub))
    rice, body, ul = best
    raw = k.to_bytes(4, "big") + ul.to_bytes(4, "big") + body + np.packbits(q.ravel()[pos] < 0).tobytes() + s.astype(np.float16).tobytes()
    rows = os.path.exists(a.table + ".r.npy")
    if rows:
        r = np.load(a.table + ".r.npy"); raw += lzma.compress(np.round(2 * np.log2(r)).astype(np.int8).tobytes(), preset=9 | lzma.PRESET_EXTREME)
    comp = raw
    here = os.path.dirname(os.path.abspath(__file__))
    src = strip(open(os.path.join(here, "..", "src", "solution_template.py"), encoding="utf8").read())
    for k, v in (("__FW__", a.fw), ("__FC__", a.fc), ("__D__", a.d), ("__IDF__", a.idf), ("__ROWS__", int(rows)), ("__P_SOCIAL__", a.social),
                 ("__P_TITLE__", a.title), ("__RICE__", rice), ("__MARK__", a.mark), ("__REFINE__", a.refine), ("__TABLE__", to_chars(comp))):
        src = src.replace(k, str(v))
    open(a.out, "w", encoding="utf8").write(src)
    n = len(src)
    print(f"{a.out}: {n} chars (limit {LIMIT}), table {len(comp) / 1024:.1f} KB = {len(to_chars(comp))} chars, code {n - len(to_chars(comp))} chars, nonzero {nz.mean():.3f}")
    assert n <= LIMIT or os.environ.get("FORCE"), "over the character limit"


if __name__ == "__main__":
    main()
