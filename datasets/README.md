# Datasets Directory

This directory is for training data used to build custom gesture recognition models.

## Structure
```
datasets/
├── fist/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── open_palm/
│   ├── img1.jpg
│   └── ...
└── ... (one folder per gesture class)
```

## Data Collection
1. Use the Camera window to capture hand crops
2. Or collect images from public datasets (e.g., ASL Alphabet on Kaggle)
3. Organize into gesture-named folders
4. Run `python train_model.py --data_dir datasets --epochs 50`

## Notes
- Images should be 150x150px hand crops (with padding)
- Augment data with rotations, flips, brightness changes for better accuracy
- Target ≥99% validation accuracy for production use
