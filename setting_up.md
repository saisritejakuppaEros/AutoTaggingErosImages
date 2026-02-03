conda create -n rnd_autotag_v2 python=3.12
conda activate rnd_autotag_v2
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install pyarrow numpy pillow tqdm torch torchvision opencv-python accelerate ultralytics pandas
pip install transformers
pip install streamlit
pip install autofaiss

conda activate rnd_autotag

<!-- export CUDA_VISIBLE_DEVICES=3
python vllm_segmentation.py --start 0 --end 10 
python compute_clip_embeddings.py --start 0 --end 10 --batch_size 32 --output_dir ./clip_embeddings
python images_segmentation.py --folder 00290 --gpu 0
python image_tagging.py --test_images_dir /data/corerndimage/image_tagging/target_dataset


 -->



# to do segmentation on all the images


python vllm_segmentation.py --start 0 --end 10 --gpu 3 \
  --dataset-dir /data/corerndimage/image_tagging/Dataset/internet_indian_dataset \
  --tags-dir /data/corerndimage/image_tagging/Dataset/vlm_tags \
  --output-dir /data/corerndimage/image_tagging/Dataset/segmentation_output \
  --tmp-dir ./tmp/my_seg \
  --min-area 150 \
  --model sam3.pt

# segmentation for images folder type over webdataset
python images_segmentation.py --folder 00290 --dataset-dir /data/corerndimage/image_tagging/target_dataset --output-dir /data/corerndimage/image_tagging/Dataset/segmentation_output --min-area 100 --model sam3.pt --gpu 0

# computing the embddings
python compute_clip_embeddings.py --start 0 --end 10 --dataset-dir /data/corerndimage/image_tagging/Dataset/internet_indian_dataset --segmentation-dir /data/corerndimage/image_tagging/Dataset/segmentation --tmp-dir ./tmp/tmp_clip --output_dir ./clip_embeddings

# test on the target dataset
python image_tagging.py --source_embeddings ./clip_embeddings --source_parquet /data/corerndimage/image_tagging/Dataset/internet_indian_dataset --test_seg_dir /data/corerndimage/image_tagging/Dataset/segmentation_output --test_images_dir /data/corerndimage/image_tagging/target_dataset --output_dir /data/corerndimage/image_tagging/test_tags --index_dir /data/corerndimage/image_tagging/faiss_index