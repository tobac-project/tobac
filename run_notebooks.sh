#!/usr/bin/env bash
find . -type f -name '*.ipynb' -not -path "*/.ipynb_checkpoints/*" > nbfiles.txt
cat nbfiles.txt

while IFS= read -r nbpath; do
  jupyter nbconvert --inplace --ClearMetadataPreprocessor.enabled=True --clear-output $nbpath
  jupyter nbconvert --to notebook --inplace --execute $nbpath
done < nbfiles.txt