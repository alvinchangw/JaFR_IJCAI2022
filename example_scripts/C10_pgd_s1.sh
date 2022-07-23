#!/bin/sh

python train.py --seed=1 --dataset=cifar10 --eval_early_stopped_model --attack=pgd --eps=8 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0 --lr_max=0.30 --n_final_eval=1000 --track_grad_sparsity_loss --track_grad_l2_norm_loss --track_low_freq_bias_loss --model_name=c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1
python eval_cifar10c.py --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1
python eval.py --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1
python eval.py --pgd_loss_func=cw --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1 --eval_results_filename=cw_eval_results.txt
python eval.py --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1 --eps 16 --eval_results_filename=eps16_eval_results.txt
python eval.py --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1 --eps 4 --eval_results_filename=eps4_eval_results.txt
python eval.py --model_dir=models/c10rn18_eps8_atkpgd10_ep30_trackallgradloss_seed1 --eps 12 --eval_results_filename=eps12_eval_results.txt