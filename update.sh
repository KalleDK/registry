#!/usr/bin/env bash
find . -name 'fetcher*.toml' -print0 | while IFS= read -r -d '' file; do
    dir=$(realpath $(dirname "$file"))
    docker run --rm -v $dir:/app fetcher fetcher --config $(basename "$file")
done