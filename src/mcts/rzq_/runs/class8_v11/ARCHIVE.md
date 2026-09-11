# Large result files

Three files exceed GitHub ordinary Git file limits and are stored as lossless gzip copies. Original files remain available locally and are ignored by Git. All other formal v11 results are tracked directly.

To restore missing originals, run these commands from this directory. gzip refuses to overwrite existing files by default.

    gzip -dk final_snapshot.pkl.gz checkpoint.pkl.gz decisions.jsonl.gz
    sha256sum -c SHA256SUMS

The hashes refer to the uncompressed originals; decompression was verified before upload.
