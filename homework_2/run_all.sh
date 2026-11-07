#!/bin/bash
set -e
cd "$(dirname "$0")"
for nb in hw2_q1 hw2_q2 hw2_q3; do
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb
done
