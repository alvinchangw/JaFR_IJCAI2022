import argparse
import os
import utils
# import apex.amp as amp
import numpy as np
import torch
import torch.utils.data as td
import time
import data
import models
from utils import rob_acc, analyze_corruption_fourier_and_freq_bias
import glob

import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])

    parser.add_argument('--analysis_dir', default='analysis/c10_corruption',
                        type=str, help='model dir name')
    parser.add_argument('--eval_results_filename', default='corruption_eval_results.txt',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--eval_json_results_filename', default='corruption_eval_results.json',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--batch_size_eval', default=1000, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')

    # CIFAR-10-C
    parser.add_argument('--severity', default='all', choices=['1', '2', '3', '4', '5', 'all'], type=str)
    parser.add_argument('--corruption', default='all', type=str, help='type of image corruption')


    return parser.parse_args()

data_paths = {
    'brightness': "./data/CIFAR-10-C/brightness.npy",
    'contrast': "./data/CIFAR-10-C/contrast.npy",
    'defocus_blur': "./data/CIFAR-10-C/defocus_blur.npy",
    'elastic_transform': "./data/CIFAR-10-C/elastic_transform.npy",
    'fog': "./data/CIFAR-10-C/fog.npy",
    'frost': "./data/CIFAR-10-C/frost.npy",
    'gaussian_blur': "./data/CIFAR-10-C/gaussian_blur.npy",
    'gaussian_noise': "./data/CIFAR-10-C/gaussian_noise.npy",
    'glass_blur': "./data/CIFAR-10-C/glass_blur.npy",
    'impulse_noise': "./data/CIFAR-10-C/impulse_noise.npy",
    'jpeg_compression': "./data/CIFAR-10-C/jpeg_compression.npy",
    'motion_blur': "./data/CIFAR-10-C/motion_blur.npy",
    'pixelate': "./data/CIFAR-10-C/pixelate.npy",
    'saturate': "./data/CIFAR-10-C/saturate.npy",
    'shot_noise': "./data/CIFAR-10-C/shot_noise.npy",
    'snow': "./data/CIFAR-10-C/snow.npy",
    'spatter': "./data/CIFAR-10-C/spatter.npy",
    'speckle_noise': "./data/CIFAR-10-C/speckle_noise.npy",
    'zoom_blur': "./data/CIFAR-10-C/zoom_blur.npy"
}

label_path = "./data/CIFAR-10-C/labels.npy"

args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
n_cls = 2 if 'binary' in args.dataset else 10

os.makedirs(args.analysis_dir, exist_ok = True)

if args.severity == 'all':
    severities = [i for i in range(1,6)]
else:
    severities = [int(args.severity)]

if args.corruption == 'all':
    corruptions = [key for key in data_paths]
else:
    corruptions = [args.corruption]

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# eps, pgd_alpha, pgd_alpha_rr = eps / 255, pgd_alpha / 255, pgd_alpha_rr / 255

eval_y = np.load(label_path)
eval_y = np.int64(eval_y)

results_dict = {}

# Clean evaluation
clean_eval_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                shuffle=False, data_augm=False)
eval_results_output = '\n Corruption difference fourier analysis'


for corruption in corruptions:
    print("Analyzing {} corruption".format(corruption))
    eval_x = np.load(data_paths[corruption])
    eval_x = np.float32(eval_x)
    eval_x = eval_x/255

    results_dict[corruption] = {}

    # current_corruption_errors = []
    for severity in severities:
        print("Analyzing #{} severity".format(severity))

        # CIFAR-10-C data loading
        eval_x_match_severity = eval_x[(severity-1)* 10000:severity* 10000]

        eval_y_match_severity = eval_y[(severity-1)* 10000:severity* 10000]

        eval_dataset = td.TensorDataset(torch.tensor(eval_x_match_severity).permute(0, 3, 1, 2), torch.tensor(eval_y_match_severity, dtype=torch.long))
        eval_batches = td.DataLoader(eval_dataset, batch_size=args.batch_size_eval) 

        time_start = time.time()

        cor_diff_low_freq_bias_value = analyze_corruption_fourier_and_freq_bias(clean_eval_batches, eval_batches, analysis_output_dir=args.analysis_dir, output_dir_suffix="-{}-{}".format(corruption, severity))

        print('{} corruption, # {} severity: cor_diff_low_freq_bias_value={}'.format(corruption, severity, cor_diff_low_freq_bias_value))
        eval_results_output += '\n {} corruption, # {} severity: cor_diff_low_freq_bias_value={}'.format(corruption, severity, cor_diff_low_freq_bias_value)

        results_dict[corruption][severity] = {"cor_diff_low_freq_bias_value": cor_diff_low_freq_bias_value.cpu().numpy().tolist()}
        time_elapsed = time.time() - time_start
    

results_path = os.path.join(args.analysis_dir, args.eval_results_filename)
with open(results_path, 'w') as f:
    f.write(eval_results_output)


output_dict_file = os.path.join(args.analysis_dir, args.eval_json_results_filename)
with open(output_dict_file, 'w') as f:
    json.dump(results_dict, f)