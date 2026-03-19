"""Convert SP-1024 tokenized shards to byte-level shards.

Reads fineweb10B_sp1024 shards, decodes to text, encodes as raw UTF-8 bytes,
writes new shards in fineweb10B_byte256 format.

Usage: python data/convert_to_bytes.py [--train-shards 80]
"""
import argparse
import numpy as np
import sentencepiece as spm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=int, default=80)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file="data/tokenizers/fineweb_1024_bpe.model")
    src_dir = Path("data/datasets/fineweb10B_sp1024")
    dst_dir = Path("data/datasets/fineweb10B_byte256")
    dst_dir.mkdir(parents=True, exist_ok=True)

    SHARD_MAGIC = 20240520
    SHARD_VERSION = 1

    def read_shard(path):
        header = np.fromfile(path, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)
        return tokens

    def write_shard(path, byte_tokens):
        header = np.zeros(256, dtype="<i4")
        header[0] = SHARD_MAGIC
        header[1] = SHARD_VERSION
        header[2] = len(byte_tokens)
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(byte_tokens.astype("<u2").tobytes())

    def convert_shard(src_path, dst_path):
        tokens = read_shard(src_path)
        # Decode SP tokens → text → UTF-8 bytes
        text = sp.decode(tokens.tolist())
        raw_bytes = text.encode("utf-8")
        byte_tokens = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint16)
        write_shard(dst_path, byte_tokens)
        return len(tokens), len(byte_tokens)

    # Convert val shards
    val_files = sorted(src_dir.glob("fineweb_val_*.bin"))
    print(f"Converting {len(val_files)} val shards...")
    total_sp = 0
    total_bytes = 0
    for vf in val_files:
        dst = dst_dir / vf.name
        n_sp, n_byte = convert_shard(vf, dst)
        total_sp += n_sp
        total_bytes += n_byte
        print(f"  {vf.name}: {n_sp:,} SP tokens → {n_byte:,} bytes ({n_byte/n_sp:.2f}x)")

    # Convert train shards
    train_files = sorted(src_dir.glob("fineweb_train_*.bin"))[:args.train_shards]
    print(f"\nConverting {len(train_files)} train shards...")
    for i, tf in enumerate(train_files):
        dst = dst_dir / tf.name
        n_sp, n_byte = convert_shard(tf, dst)
        total_sp += n_sp
        total_bytes += n_byte
        if (i + 1) % 10 == 0 or i + 1 == len(train_files):
            print(f"  [{i+1}/{len(train_files)}] {tf.name}: {n_sp:,} → {n_byte:,} bytes")

    print(f"\nDone. Total: {total_sp:,} SP tokens → {total_bytes:,} bytes ({total_bytes/total_sp:.2f}x expansion)")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
