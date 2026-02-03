
install the version of vllm, where you can run the qwen image 3

if you are on h200 use this , 


the whole goal is to tag the objects present in the images to do check the closet tags it can get based on objects.


```
conda activate teja_pomato


tmux new-session -s vllm_server_0 
export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":20,"video":0}' \
  --allowed-local-media-path /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp


tmux new-session -s vllm_server_1
export CUDA_VISIBLE_DEVICES=3
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":20,"video":0}' \
  --allowed-local-media-path /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp


tmux new-session -s vllm_server_2
export CUDA_VISIBLE_DEVICES=4
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":20,"video":0}' \
  --allowed-local-media-path /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp


tmux new-session -s vllm_server_3
export CUDA_VISIBLE_DEVICES=5
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8003 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":20,"video":0}' \
  --allowed-local-media-path /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp

tmux new-session -s vllm_server_4

export CUDA_VISIBLE_DEVICES=6
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8004 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":20,"video":0}' \
  --allowed-local-media-path /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation/tmp


tmux new-session -s tag_gpu_0
cd /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation
conda activate teja_pomato
python vllm_tag_images.py --start 0 --end 70 --port 8000

tmux new-session -s tag_gpu_1
cd /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation
conda activate teja_pomato
python vllm_tag_images.py --start 70 --end 140 --port 8001

tmux new-session -s tag_gpu_2
cd /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation
conda activate teja_pomato
python vllm_tag_images.py --start 140 --end 210 --port 8002

tmux new-session -s tag_gpu_3
cd /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation
conda activate teja_pomato
python vllm_tag_images.py --start 210 --end 280 --port 8003

tmux new-session -s tag_gpu_4
cd /mnt/data0/data0/mouryesh/T2I/tagging/object_tagging/pipeline_generation
conda activate teja_pomato
python vllm_tag_images.py --start 280 --end 350 --port 8004
```