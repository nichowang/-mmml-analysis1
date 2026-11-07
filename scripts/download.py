"""Stream Seamless Interaction Naturalistic train archives and keep only the
.npz (SMPL-H / movement) and .json (transcript + VAD) members.

Video (.mp4) and audio (.wav) are skipped, so nothing large is written to disk.

Usage: python download.py OUT_DIR BATCH ARCHIVE [ARCHIVE ...]
"""
import os
import sys
import tarfile
import time
import urllib.request

BASE = ("https://huggingface.co/datasets/facebook/seamless-interaction/"
        "resolve/main/naturalistic/train/{b:04d}/{a:04d}.tar")
KEEP = (".npz", ".json")


def fetch(out_dir, batch, archive):
    dst = os.path.join(out_dir, f"{batch:04d}_{archive:04d}")
    done = os.path.join(dst, ".done")
    if os.path.exists(done):
        print("skip", dst, flush=True)
        return
    os.makedirs(dst, exist_ok=True)
    url = BASE.format(b=batch, a=archive)
    n = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(KEEP):
                    continue
                f = tf.extractfile(m)
                with open(os.path.join(dst, os.path.basename(m.name)), "wb") as g:
                    g.write(f.read())
                n += 1
    open(done, "w").close()
    print(f"{dst}: kept {n} files", flush=True)


def fetch_retry(out_dir, batch, archive, tries=6):
    """A dropped connection leaves no .done marker, so the archive is simply redone."""
    for t in range(tries):
        try:
            return fetch(out_dir, batch, archive)
        except Exception as e:  # network errors, truncated tar stream
            wait = 30 * 2 ** t
            print(f"archive {archive} attempt {t + 1} failed: {e!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"archive {archive} FAILED after {tries} attempts", flush=True)


if __name__ == "__main__":
    out, batch = sys.argv[1], int(sys.argv[2])
    for a in map(int, sys.argv[3:]):
        fetch_retry(out, batch, a)
