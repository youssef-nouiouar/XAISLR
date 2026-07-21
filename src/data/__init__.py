# src.data — data pipeline (scan, dedup, split, dataset).
# Keep this module import-light: dataset.py pulls in torch/torchvision, but
# dedup/split/scan must stay torch-free so prepare_data.py runs without a GPU.
