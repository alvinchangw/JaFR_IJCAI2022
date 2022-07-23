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
from utils import rob_acc
import glob

import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='tinyimagenet', choices=['tinyimagenet'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet34', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])

    parser.add_argument('--model_dir', default='models/c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad',
                        type=str, help='model dir name')
    parser.add_argument('--model_filename', default=None,
                        type=str, help='model filename')
    parser.add_argument('--eval_results_filename', default='corruption_eval_results.txt',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--eval_json_results_filename', default='corruption_eval_results.json',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model in [fc, cnn])')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#filters on each conv layer (for model==fc)')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--early_stopped_model', action='store_true', help='eval the best model according to pgd_acc evaluated every k iters (typically, k=200)')

    # Tinyimagenet-C
    parser.add_argument('--severity', default='all', choices=['1', '2', '3', '4', '5', 'all'], type=str)
    parser.add_argument('--corruption', default='all', type=str, help='type of image corruption')


    return parser.parse_args()

data_paths = {
    'brightness': "./data/tinyimagenet-C/brightness",
    'contrast': "./data/tinyimagenet-C/contrast",
    'defocus_blur': "./data/tinyimagenet-C/defocus_blur",
    'elastic_transform': "./data/tinyimagenet-C/elastic_transform",
    'fog': "./data/tinyimagenet-C/fog",
    'frost': "./data/tinyimagenet-C/frost",
    'gaussian_blur': "./data/tinyimagenet-C/gaussian_blur",
    'gaussian_noise': "./data/tinyimagenet-C/gaussian_noise",
    'glass_blur': "./data/tinyimagenet-C/glass_blur",
    'impulse_noise': "./data/tinyimagenet-C/impulse_noise",
    'jpeg_compression': "./data/tinyimagenet-C/jpeg_compression",
    'motion_blur': "./data/tinyimagenet-C/motion_blur",
    'pixelate': "./data/tinyimagenet-C/pixelate",
    'saturate': "./data/tinyimagenet-C/saturate",
    'shot_noise': "./data/tinyimagenet-C/shot_noise",
    'snow': "./data/tinyimagenet-C/snow",
    'spatter': "./data/tinyimagenet-C/spatter",
    'speckle_noise': "./data/tinyimagenet-C/speckle_noise",
    'zoom_blur': "./data/tinyimagenet-C/zoom_blur"
}


args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
n_cls = 200


if args.severity == 'all':
    severities = [i for i in range(1,6)]
else:
    severities = [int(args.severity)]

if args.corruption == 'all':
    corruptions = [key for key in data_paths]
else:
    corruptions = [args.corruption]

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn)
model = model.cuda()

if args.model_filename is None:
    for name in glob.glob(os.path.join(args.model_dir, "*.pth")): 
        print(name)
        model_path = name
else:
    model_path = os.path.join(args.model_dir, args.model_filename)

print('Evaluating model @ {}'.format(model_path))
eval_results_output = 'Evaluating model @ {}'.format(model_path)

print('Evaluating corruptions: ', corruptions)
print('     severities: ', severities)

model_dict = torch.load(model_path)

if args.early_stopped_model:
    model.load_state_dict(model_dict['best'])
else:
    model.load_state_dict(model_dict['last'] if 'last' in model_dict else model_dict)
print("Loaded model weights")

opt = torch.optim.SGD(model.parameters(), lr=0)  # needed for backprop only

utils.model_eval(model, half_prec)


results_dict = {}

for corruption in corruptions:
    print("Evaluating {} corruption".format(corruption))

    results_dict[corruption] = {}

    current_corruption_errors = []
    for severity in severities:
        print("Evaluating #{} severity".format(severity))

        # data loading
        img_folder_dir = os.path.join(data_paths[corruption], str(severity))
        print("img_folder_dir: ", img_folder_dir)
        eval_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False, img_folder_dir=img_folder_dir)

        time_start = time.time()

        acc_corrupt, loss_corrupt, _ = rob_acc(eval_batches, model, 0, 0, opt, half_prec, 0, 1)

        print('{} corruption, # {} severity: acc={:.2%}, loss={:.3f}'.format(corruption, severity, acc_corrupt, loss_corrupt))
        eval_results_output += '\n {} corruption, # {} severity: acc={:.2%}, loss={:.3f}'.format(corruption, severity, acc_corrupt, loss_corrupt)

        results_dict[corruption][severity] = {"acc": acc_corrupt, "loss": loss_corrupt}
        current_corruption_errors.append(acc_corrupt)

        time_elapsed = time.time() - time_start
    
    corruption_mean_error = np.mean(current_corruption_errors)
    print('{} corruption, avg over all severities: acc={:.2%}'.format(corruption, corruption_mean_error))
    eval_results_output += '\n {} corruption, avg over all severities: acc={:.2%}'.format(corruption, corruption_mean_error)
    results_dict[corruption]['mean'] = {"acc": corruption_mean_error}

results_path = os.path.join(args.model_dir, args.eval_results_filename)
with open(results_path, 'w') as f:
    f.write(eval_results_output)


output_dict_file = os.path.join(args.model_dir, args.eval_json_results_filename)
with open(output_dict_file, 'w') as f:
    json.dump(results_dict, f)