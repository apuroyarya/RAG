"""Tests for the embedding layer. Runs against the fake backend by default.

    python scripts/test_embedding.py              # fake, instant
    EMBED_BACKEND=sentence_transformers python scripts/test_embedding.py

With the real backend this also checks that semantically related Bengali text
scores higher than unrelated text - the one property the fake backend cannot
have, and the one that actually matters for retrieval.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.embedding import get_embedder, normalize, pack, unpack

failures = []


def check(label, cond, detail=""):
    print(f"{'  ok  ' if cond else ' FAIL '} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(label)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_serialization():
    vec = [0.5, -0.25, 0.125, 0.0]
    back = unpack(pack(vec))
    check("float32 round-trip preserves values", back == vec, str(back))
    check("packed size is 4 bytes per float", len(pack(vec)) == len(vec) * 4)
    check("empty vector round-trips", unpack(pack([])) == [])

    # a long vector, since misaligned slicing is the realistic failure
    long_vec = [i / 1000 for i in range(1024)]
    packed = pack(long_vec)
    check("1024-float vector packs to 4096 bytes", len(packed) == 4096)
    check("slicing a concatenated buffer recovers each vector",
          unpack((packed + packed)[4096:]) == unpack(packed))


def test_normalize():
    check("normalize gives unit length",
          abs(sum(x * x for x in normalize([3.0, 4.0])) - 1.0) < 1e-6)
    check("normalize handles the zero vector", normalize([0.0, 0.0]) == [0.0, 0.0])


def test_backend():
    backend = os.environ.get("EMBED_BACKEND", "fake")
    embedder = get_embedder(backend)
    ok, why = embedder.available()
    if not ok:
        print(f"\n  backend {backend!r} unavailable: {why}")
        print("  skipping backend tests")
        return
    print(f"\n  backend: {embedder.model_id}")

    texts = ["দৈনিক দেবযজ্ঞ বিধি", "দৈনিক দেবযজ্ঞ বিধি", "সম্পূর্ণ ভিন্ন একটি বিষয়"]
    vectors = embedder.embed(texts)

    check("one vector per input", len(vectors) == len(texts))
    check("vectors have consistent width",
          len({len(v) for v in vectors}) == 1, str(len(vectors[0])))
    check("vectors are unit length",
          all(abs(sum(x * x for x in v) - 1.0) < 1e-4 for v in vectors))
    check("identical text gives identical vectors", vectors[0] == vectors[1])
    check("different text gives different vectors", vectors[0] != vectors[2])
    check("dim property matches actual width", embedder.dim == len(vectors[0]))
    check("empty input list is handled", embedder.embed([]) == [])

    if backend == "fake":
        print("      (fake backend: no semantic check - hashed vectors carry "
              "no meaning)")
        return

    # the property that actually matters, and that only a real model has
    related = "দেবযজ্ঞের নিয়ম ও বিধি"
    unrelated = "ক্রিকেট খেলার নিয়মাবলী"
    base, rel, unrel = embedder.embed(["দৈনিক দেবযজ্ঞ বিধি", related, unrelated])
    s_rel, s_unrel = dot(base, rel), dot(base, unrel)
    check("related Bengali scores above unrelated",
          s_rel > s_unrel, f"related={s_rel:.3f} unrelated={s_unrel:.3f}")

    # cross-lingual, which is what phases 2-3 depend on
    en = embedder.embed(["rules and procedures of the daily fire ritual"])[0]
    s_cross = dot(base, en)
    print(f"      cross-lingual bn/en similarity: {s_cross:.3f} "
          f"(informational - phase 1 is monolingual)")


for fn in (test_serialization, test_normalize, test_backend):
    print(f"\n{fn.__name__}")
    fn()

print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nall passed")
sys.exit(1 if failures else 0)
