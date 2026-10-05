#!/usr/bin/env python3
"""mk_draft_vocab.py GGUF BASE.bin OUT.bin [ASCII_BELOW]: a draft vocabulary (int32 token ids) for QWFN_MTP_DRAFT_VOCAB =
BASE (e.g. Strata's English+code subset) + every token holding a non-ASCII Latin character (or a byte-level piece of
one) + the pure-ASCII tokens with an id below ASCII_BELOW (BPE ids follow merge order: lower = more frequent; default
0 = none). All-ASCII-Latin would add ~100K tokens, most of the vocabulary. Characters counted as Latin: ASCII, Latin-1 Supplement and Latin Extended
A/B (Spanish and the other Western European languages: accents, n-tilde), general punctuation, and the byte-level
pieces of those characters (a token can hold half of a UTF-8 sequence). Token strings come from the GGUF's
tokenizer.ggml.tokens (GPT-2 byte-level BPE: each byte mapped to a printable code point)."""
import struct, sys

def read_tokens(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, = struct.unpack("<I", f.read(4)); nt, nkv = struct.unpack("<QQ", f.read(16))
    def s(): n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
    SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            return [val(et) for _ in range(n)]
        fmt = "<" + SZ[t]; return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    for _ in range(nkv):
        k = s(); t, = struct.unpack("<I", f.read(4)); v = val(t)
        if k == "tokenizer.ggml.tokens": return v
    raise SystemExit("no tokenizer.ggml.tokens")

def byte_decoder():   # GPT-2's bytes_to_unicode, inverted
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def latin_ok(ch):
    c = ord(ch)
    return (c < 0x80) or (0xA0 <= c <= 0x24F) or (0x2000 <= c <= 0x206F) or c in (0x20AC,)

def main():
    gguf, base, out = sys.argv[1:4]
    ascii_below = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    toks = read_tokens(gguf)
    dec = byte_decoder()
    ids = list(struct.unpack("<%di" % (len(open(base, "rb").read()) // 4), open(base, "rb").read()))
    have = set(ids); added = 0
    for i, t in enumerate(toks):
        if i in have: continue
        try: b = bytes(dec[ch] for ch in t)
        except KeyError: continue   # a special / added token outside the byte map
        try:
            txt = b.decode("utf-8")
            ok = len(txt) > 0 and all(latin_ok(ch) for ch in txt) and \
                 (any(0xA0 <= ord(ch) <= 0x24F for ch in txt) or (i < ascii_below and all(ord(ch) < 0x80 for ch in txt)))
        except UnicodeDecodeError:
            # a piece of a multi-byte character: keep it when every lead byte starts a Latin-1/Latin-Extended
            # character (0xC2-0xC9: U+0080-U+027F) and the rest are continuation bytes
            ok = all(0x80 <= x <= 0xBF or 0xC2 <= x <= 0xC9 or x < 0x80 for x in b)
        if ok: ids.append(i); have.add(i); added += 1
    open(out, "wb").write(struct.pack("<%di" % len(ids), *ids))
    print(f"{base}: {len(ids) - added} ids + {added} Latin-script (ASCII below {ascii_below}) = {len(ids)} of {len(toks)} -> {out}")

main()
