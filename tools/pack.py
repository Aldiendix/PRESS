"""Binary <-> text packing used to embed model tables inside the single-file submission.

The competition limit is 50,000 *characters*, so every character carries 15 bits:
code point = 0x4E00 + value (0..32767), a contiguous BMP range (CJK, Yi, Hangul) with no quotes,
backslashes, surrogates or line breaks.
"""
import lzma

BASE = 0x4E00


def to_chars(b: bytes) -> str:
    n = int.from_bytes(b, "big"); nbits = len(b) * 8; k = -(-nbits // 15); n <<= k * 15 - nbits
    return chr(BASE + (len(b) >> 15)) + chr(BASE + (len(b) & 32767)) + "".join(chr(BASE + ((n >> (15 * (k - 1 - i))) & 32767)) for i in range(k))


def from_chars(s: str) -> bytes:
    ln = (ord(s[0]) - BASE) << 15 | (ord(s[1]) - BASE); n = 0
    for c in s[2:]:
        n = n << 15 | (ord(c) - BASE)
    k = len(s) - 2
    return (n >> (k * 15 - ln * 8)).to_bytes(ln, "big")


def compress(b: bytes) -> bytes:
    best = None
    for lc in (0, 1, 2, 3, 4):
        for lp in (0, 1, 2):
            for pb in (0, 1, 2):
                try:
                    c = lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": lc, "lp": lp, "pb": pb, "dict_size": 1 << 24}])
                except lzma.LZMAError:
                    continue
                if best is None or len(c) < len(best[0]):
                    best = (c, lc, lp, pb)
    return best


if __name__ == "__main__":
    import os
    for n in (0, 1, 7, 1000, 4097):
        b = os.urandom(n); assert from_chars(to_chars(b)) == b
    print("pack roundtrip ok")
