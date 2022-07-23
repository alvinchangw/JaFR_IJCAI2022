 JaFR_IJCAI2022
This is our Pytorch implementation of JaFR. 

**How Does Frequency Bias Affect the Robustness of Neural Image Classifiers against Common Corruption and Adversarial Perturbations?**<br>
*Alvin Chan, Yew-Soon Ong, Clement Tan*<br>
https://arxiv.org/abs/2205.04533

TL;DR: We propose JaFR to more directly study the frequency bias of a model through the lens of its Jacobians and its implication to model robustness.


## Requirements
- Python 3.8.1 on Linux
- PyTorch 1.4


## Dataset
See instructions [here](https://github.com/hendrycks/robustness) to download the CIFAR-10-C and CIFAR-100-C corruption datasets.


# Examples
Folder containing scripts to train and evaluate models, with the parameters reported in the paper: `example_scripts/`.  

## Training
Train model with JaFR and FGSM, with the parameters reported in the paper:  
```bash
python train.py  --model_name=model_folder_name --dataset=cifar10 --eval_early_stopped_model --attack=fgsm --eps=8 --attack_init=zero --epochs=30 --grad_align_cos_lambda=0 --lr_max=0.30 --low_freq_bias_lambda=0.001  --seed=1
```  
After training, the COCON block's weights will be saved in `models/model_folder_name`.

### Training Key Arguments
`--model_name` : path of folder where model weights, logs and parameters are saved  
`--attack` : type of adversarial training
`--dataset`: dataset to train model on  
`--low_freq_bias_lambda`: weight of JaFR in training    


## Evaluation
Evaluate model with adversarial examples (PGD with eps value of 8):  
```bash
python eval.py --model_dir=models/model_folder_name --eps 8 --eval_results_filename=eval_results.txt
```  
Evaluation output text file (`eval_results.txt`) will be saved in `models/model_folder_name`.

### Evaluation Key Arguments
`--model_dir` : path of folder where model weights, logs and parameters are saved  
`--eps` : eps of PGD/FGSM attack
`--dataset`: type of image dataset   
`--pgd_loss_func`: type of loss function used to compute adversarial examples    


## Citation
If you find our repository useful, please consider citing our paper:

```
@article{chan2022does,
  title={How Does Frequency Bias Affect the Robustness of Neural Image Classifiers against Common Corruption and Adversarial Perturbations?},
  author={Chan, Alvin and Ong, Yew-Soon and Tan, Clement},
  journal={arXiv preprint arXiv:2205.04533},
  year={2022}
}
```


## Acknowledgements
Code is adapted from:
- https://github.com/tml-epfl/understanding-fast-adv-training