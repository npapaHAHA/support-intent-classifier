from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


BANKING77 = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
)


def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        destination = data_dir / f"banking-{split}.csv"
        if destination.is_file() and destination.stat().st_size:
            print(f"Using {destination}")
            continue

        partial = destination.with_suffix(".part")
        try:
            with urlopen(f"{BANKING77}/{split}.csv", timeout=60) as response, partial.open("wb") as output:
                copyfileobj(response, output)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        print(f"Downloaded {destination}")


if __name__ == "__main__":
    main()
