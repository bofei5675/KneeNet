main.py
---
Used pretrained model to make prediction, and output bbox files
Used `scripts/annote.lsf` to submite jobs


draw_figure2.py
---
Draw a figure with predicted bbox of given month
```
python draw_figure2.py 96m &
```
dataset_generation/
---
Generate figures that has whole knee x-ray image with predicted bbox
Generate dataset (h5 files) based on the summary file.
