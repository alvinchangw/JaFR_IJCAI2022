#!/bin/sh

python train.py --seed=1 --dataset=cifar100 --eval_early_stopped_model --lr_max=0.1 --attack=none --eps=8 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0 --n_final_eval=1000 --low_freq_bias_lambda=0.001 --track_grad_sparsity_loss --track_grad_l2_norm_loss --track_low_freq_bias_loss --model_name=c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1
python eval_cifar100c.py --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1
python eval.py --dataset=cifar100 --pgd_loss_func=cw --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1 --eval_results_filename=eval_results_cw.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1 --eps 12 --eval_results_filename=eps12_eval_results.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1 --eps 4 --eval_results_filename=eps4_eval_results.txt
python eval.py --dataset=cifar100 --model_dir=models/c100rn18_eps8_atknone_ep30_nogradalign_lowfreq0001_trackallgradloss_seed1 --eps 16 --eval_results_filename=eps16_eval_results.txt