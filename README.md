<h1 style="font-size: 3em;">SDformer: Similarity-driven Discrete Transformer For Time Series Generation</h1>

# Usage
## Install the environment using yaml file
~~~
conda env create -f environment.yaml
~~~
## Train Stage 1
~~~
python train_vq.py --batch-size 128 --width 512 --lr 1e-4 --total-iter 100000 --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_energy --dataname energy --vq-act relu --quantizer ema_reset_sim --exp-name VQVAE --window-size 24 --commit 0.001 --gpu 0 
~~~

## Train Stage 2

### SDformer-ar
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_sdformer_ar.py --exp-name ARTM --batch-size 4096 --num-layers 1 --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512 --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset_sim --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

### SDformer-m
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_sdformer_m.py --exp-name MTM --batch-size 4096 --num-layers 1  --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512  --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset_sim --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

## Test
### SDformer-ar
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_sdformer_ar.py --exp-name ARTM_test --if-test --resume-trans ./output/output_energy/ARTM/net_best_ds.pth --batch-size 4096 --num-layers 1 --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512 --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset_sim --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

### SDformer-m
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_sdformer_m.py --exp-name MTM_test  --if-test --resume-trans ./output/output_energy/MTM/net_best_ds.pth --batch-size 4096 --num-layers 1  --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512  --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset_sim --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

## Eval
### SDformer-ar
~~~
python eval.py --dir ./output/output_energy/ARTM_test
~~~

### SDformer-m
~~~
python eval.py --dir ./output/output_energy/MTM_test
~~~

