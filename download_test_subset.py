"""
Download only the annotations.csv from Slovo zip (without downloading 16 GB),
then selectively download N test-split videos.

Usage:
    python download_test_subset.py --n 200 --out slovo_test/
"""

import argparse
import io
import os
import struct
import urllib.request
import zipfile

SLOVO_ZIP_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
)
BASE_VIDEO_URL = (  # individual video URL pattern (same server, flat structure)
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo/"
)


# ---------------------------------------------------------------------------
# Step 1: extract annotations.csv by reading only the ZIP central directory
# ---------------------------------------------------------------------------

def http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def get_zip_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_annotations_csv(url: str) -> bytes:
    """Stream only the ZIP end-of-central-directory + central directory,
    find annotations.csv offset, then fetch only that file entry."""
    print("[1/3] Getting ZIP file size …")
    size = get_zip_size(url)
    print(f"      ZIP size: {size / 1e9:.2f} GB")

    # Read last 64 KB — enough to contain end-of-central-directory record
    tail = http_range(url, max(0, size - 65536), size - 1)

    # Parse using Python's zipfile (trick: feed a fake file-like object)
    buf = io.BytesIO(tail)

    # Locate EOCD signature 0x06054b50
    sig = b"\x50\x4b\x05\x06"
    idx = tail.rfind(sig)
    if idx == -1:
        raise RuntimeError("EOCD signature not found — ZIP may use ZIP64")

    eocd = tail[idx:]
    # EOCD layout: sig(4) disk(2) start_disk(2) entries_disk(2) entries(2)
    #              cd_size(4) cd_offset(4) comment_len(2)
    (_, _, _, num_entries, cd_size, cd_offset, _) = struct.unpack_from(
        "<4sHHHHII H", eocd
    )
    real_cd_offset = size - len(tail) + idx - cd_size

    print(f"[2/3] Fetching central directory ({cd_size / 1e3:.1f} KB, {num_entries} entries) …")
    cd_data = http_range(url, real_cd_offset, real_cd_offset + cd_size - 1)

    # Build minimal fake ZIP to let Python parse the central directory
    fake_zip = cd_data + eocd
    # Fix cd_offset in EOCD to 0 (relative to our fake buffer)
    eocd_fixed = eocd[:16] + struct.pack("<I", len(cd_data)) + eocd[20:]
    fake_zip = cd_data + eocd_fixed

    zf = zipfile.ZipFile(io.BytesIO(fake_zip))

    # Find annotations.csv
    csv_name = None
    for info in zf.infolist():
        if info.filename.endswith("annotations.csv"):
            csv_name = info.filename
            local_header_offset = info.header_offset
            compress_size = info.compress_size
            break

    if csv_name is None:
        raise RuntimeError("annotations.csv not found in ZIP central directory")

    # Local file header is 30 bytes + filename + extra; fetch a chunk to parse
    print(f"[3/3] Downloading {csv_name} ({compress_size / 1e3:.1f} KB) …")
    header_chunk = http_range(url, local_header_offset, local_header_offset + 299)
    # Local header: sig(4) ver(2) flags(2) method(2) mod_time(2) mod_date(2)
    #               crc(4) comp_size(4) uncomp_size(4) fname_len(2) extra_len(2)
    fname_len, extra_len = struct.unpack_from("<HH", header_chunk, 26)
    data_offset = local_header_offset + 30 + fname_len + extra_len

    compressed = http_range(url, data_offset, data_offset + compress_size - 1)

    info2 = zf.getinfo(csv_name)
    return zipfile.ZipExtFile(
        io.BytesIO(compressed), "r", info2, None, True
    ).read()


# ---------------------------------------------------------------------------
# Step 2: download individual video files
# ---------------------------------------------------------------------------

def download_videos(video_ids: list[str], out_dir: str, base_url: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    downloaded = []
    for i, vid_id in enumerate(video_ids, 1):
        fname = f"{vid_id}.mp4"
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            downloaded.append(dest)
            continue
        url = base_url + fname
        print(f"  [{i}/{len(video_ids)}] {fname} … ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"{size_kb:.0f} KB")
            downloaded.append(dest)
        except Exception as e:
            print(f"FAILED ({e})")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="Number of test videos to download (default 200)")
    parser.add_argument("--out", default="slovo_test",
                        help="Output directory for videos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # -- Get annotations.csv --
    csv_path = os.path.join(args.out, "annotations.csv")
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(csv_path):
        print(f"annotations.csv already exists at {csv_path}, skipping download.")
    else:
        try:
            csv_bytes = fetch_annotations_csv(SLOVO_ZIP_URL)
            with open(csv_path, "wb") as f:
                f.write(csv_bytes)
            print(f"Saved annotations.csv ({len(csv_bytes) / 1e3:.1f} KB)")
        except Exception as e:
            print(f"ZIP-stream approach failed: {e}")
            print("Trying direct URL fallback …")
            # Sometimes the server has a separate CSV
            fallback = SLOVO_ZIP_URL.replace("slovo.zip", "annotations.csv")
            try:
                urllib.request.urlretrieve(fallback, csv_path)
                print("Downloaded via fallback URL.")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                print("\nPlease download annotations.csv manually from Kaggle:")
                print("  https://www.kaggle.com/datasets/kapitanov/slovo")
                return

    # -- Parse annotations --
    import csv
    import random

    random.seed(args.seed)
    test_rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["train"].strip().lower() in ("false", "0"):
                test_rows.append(row)

    print(f"\nTest split: {len(test_rows)} videos total")
    sample = random.sample(test_rows, min(args.n, len(test_rows)))
    print(f"Sampling {len(sample)} videos …\n")

    video_ids = [r["attachment_id"] for r in sample]
    labels = {r["attachment_id"]: r["text"] for r in sample}

    # -- Download videos --
    downloaded = download_videos(video_ids, args.out, BASE_VIDEO_URL)
    print(f"\nDownloaded {len(downloaded)}/{len(sample)} videos to '{args.out}/'")

    # -- Save label map --
    label_map_path = os.path.join(args.out, "labels.csv")
    with open(label_map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attachment_id", "text"])
        for vid_id in video_ids:
            writer.writerow([vid_id, labels[vid_id]])
    print(f"Label map saved to {label_map_path}")


if __name__ == "__main__":
    main()
