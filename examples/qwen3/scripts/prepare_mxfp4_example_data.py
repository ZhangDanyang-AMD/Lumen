"""Fetch and verify the fixed Alpaca source used by numbered example 13."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError("Alpaca source changed; refusing to silently change the benchmark data")
    rows = json.loads(payload)
    manifest = {"url": URL, "source_sha256": SHA256}
    for name, split in (("train", rows[:-512]), ("validation", rows[-512:])):
        data = "".join(json.dumps(row) + "\n" for row in split).encode()
        path = args.output_dir / (name + ".jsonl")
        path.write_bytes(data)
        manifest[name] = {"rows": len(split), "sha256": hashlib.sha256(data).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
