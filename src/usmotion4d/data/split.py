"""Frozen subject-disjoint split of the 50 TUS-REC training subjects (40 train / 10 dev). The 3 public-validation subjects (050-052)
are never part of it. Manifest = JSON + sha256, versioned in configs/splits/."""
import glob, hashlib, json, os, random

ROOT = "/mnt/hdd/vcivale/DATASETS/TUS-REC2024/"
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "splits", "split_v1.json")


def make_manifest(path=MANIFEST, seed=17, n_dev=10, root=ROOT):
    ids = sorted(os.path.basename(p) for p in glob.glob(root + "[0-4][0-9][0-9]"))
    assert len(ids) == 50, len(ids)
    random.Random(seed).shuffle(ids)
    m = dict(version=1, seed=seed, train=sorted(ids[n_dev:]), dev=sorted(ids[:n_dev]), note="subject-disjoint; official test (32 subjects) never used")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(json.dumps(m, indent=1))
    return m


def load_manifest(path=MANIFEST):
    raw = open(path, "rb").read()
    m = json.loads(raw)
    m["sha256"] = hashlib.sha256(raw).hexdigest()
    assert not set(m["train"]) & set(m["dev"]) and len(m["train"]) == 40 and len(m["dev"]) == 10
    return m
