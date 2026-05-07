python -m scripts.resize_chest_cts_and_masks \
  --raw-root datasets/rexgrounding_original \
  --out-root datasets/rexgrounding \
  --masks-dir-name masks \
  --target-shape 256 256 128

python -m scripts.prepare_chest_text_controlnet_dataset \
  --raw-root datasets/rexgrounding \
  --out-root datasets/rexgrounding_processed \
  --masks-dir-name masks \
  --target-label 23 \
  --val-frac 0.25

python -m scripts.diff_model_create_training_data \
   -e configs/environment_create_chest_text_embeddings.json \
   -c configs/config_maisi_diff_model_rflow-ct.json \
   -t configs/config_network_rflow_text.json \
   -g 1

python -m scripts.train_controlnet \
  -t configs/config_network_rflow_text.json \
  -c configs/config_maisi_controlnet_train_chest_text.json \
  -e configs/environment_maisi_controlnet_train_chest_text.json \
  -g 1