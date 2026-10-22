from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_DIR = str(DATA_DIR.parent / "cache")


def download(destination, **options):
    if (destination / "state.json").exists():
        print(f"Already downloaded: {destination}")
        return

    print(f"Downloading: {destination}")
    dataset = load_dataset(
        split="train",
        cache_dir=CACHE_DIR,
        **options,
    )
    dataset.save_to_disk(str(destination))
    print(f"Saved {len(dataset):,} rows")


def main():
    # FTRACE-TREx: abstracts and queries.
    for subset in ("abstracts", "queries"):
        download(
            DATA_DIR / "ftrace" / subset,
            path="ekinakyurek/ftrace",
            revision="refs/convert/parquet",
            data_files=f"{subset}/train/*.parquet",
        )

    # T-REx: full facts, evaluation sample, and all sentence shards.
    base = "https://storage.googleapis.com/tda-resources/2410.17413/public"

    for name, filename in {
        "facts": "trex_facts.jsonl",
        "queries": "trex_facts_sample.jsonl",
    }.items():
        download(
            DATA_DIR / "trex" / name,
            path="json",
            data_files=f"{base}/{filename}",
        )

    for shard in range(20):
        download(
            DATA_DIR / "trex" / "sentences" / f"shard_{shard:02d}",
            path="json",
            data_files=(
                f"{base}/trex_sentences.jsonl-{shard:05d}-of-00020"
            ),
        )


if __name__ == "__main__":
    main()