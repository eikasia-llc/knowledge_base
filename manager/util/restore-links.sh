#!/bin/bash

cd "$(dirname "$(realpath "$0")")/../.."

echo "Restoring links. pwd: $(pwd)"

for f in manager/*/*.py; do
    dir=$(dirname "$f")
    base=$(basename "$f" .py)
    ln -vsf ../../manager/util/sh2py3.sh "bin/$(basename "$dir")/$base"
done

ls -l --color=always bin/*
