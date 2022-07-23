#!/bin/sh

python train.py --seed=1 --dataset=cifar100 --eval_early_stopped_model --attack=fgsm --eps=8 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0.2 --lr_max=0.30 --n_final_eval=1000 --model_name=c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1
python eval_cifar100c.py --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1
python eval.py --dataset=cifar100 --pgd_loss_func=cw --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1 --eval_results_filename=eval_results_cw.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1 --eps 12 --eval_results_filename=eps12_eval_results.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1 --eps 4 --eval_results_filename=eps4_eval_results.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atkfgsm_ep30_gradalign02_seed1 --eps 16 --eval_results_filename=eps16_eval_results.txt