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
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    # parser.add_argument('--model_path', default='2020-03-19 23:51:05 dataset=cifar100 model=resnet18 eps=8.0 attack=pgd attack_init=zero fgsm_alpha=1.25 epochs=30 pgd_train_n_iters=7 grad_align_cos_lambda=0.0 seed=1 epoch=30',
    #                     type=str, help='model name')
    parser.add_argument('--model_dir', default='models/c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad',
                        type=str, help='model dir name')
    # parser.add_argument('--model_filename', default='c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad_epoch60.pth',
    #                     type=str, help='model filename')
    parser.add_argument('--model_filename', default=None,
                        type=str, help='model filename')
    parser.add_argument('--eval_results_filename', default='corruption_eval_results.txt',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--eval_json_results_filename', default='corruption_eval_results.json',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    # parser.add_argument('--n_eval', default=256, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model in [fc, cnn])')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#filters on each conv layer (for model==fc)')
    # parser.add_argument('--batch_size_eval', default=1024, type=int, help='batch size for evaluation')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--early_stopped_model', action='store_true', help='eval the best model according to pgd_acc evaluated every k iters (typically, k=200)')

    # CIFAR-100-C
    parser.add_argument('--severity', default='all', choices=['1', '2', '3', '4', '5', 'all'], type=str)
    parser.add_argument('--corruption', default='all', type=str, help='type of image corruption')


    return parser.parse_args()

data_paths = {
    'brightness': "./data/CIFAR-100-C/brightness.npy",
    'contrast': "./data/CIFAR-100-C/contrast.npy",
    'defocus_blur': "./data/CIFAR-100-C/defocus_blur.npy",
    'elastic_transform': "./data/CIFAR-100-C/elastic_transform.npy",
    'fog': "./data/CIFAR-100-C/fog.npy",
    'frost': "./data/CIFAR-100-C/frost.npy",
    'gaussian_blur': "./data/CIFAR-100-C/gaussian_blur.npy",
    'gaussian_noise': "./data/CIFAR-100-C/gaussian_noise.npy",
    'glass_blur': "./data/CIFAR-100-C/glass_blur.npy",
    'impulse_noise': "./data/CIFAR-100-C/impulse_noise.npy",
    'jpeg_compression': "./data/CIFAR-100-C/jpeg_compression.npy",
    'motion_blur': "./data/CIFAR-100-C/motion_blur.npy",
    'pixelate': "./data/CIFAR-100-C/pixelate.npy",
    'saturate': "./data/CIFAR-100-C/saturate.npy",
    'shot_noise': "./data/CIFAR-100-C/shot_noise.npy",
    'snow': "./data/CIFAR-100-C/snow.npy",
    'spatter': "./data/CIFAR-100-C/spatter.npy",
    'speckle_noise': "./data/CIFAR-100-C/speckle_noise.npy",
    'zoom_blur': "./data/CIFAR-100-C/zoom_blur.npy"
}

label_path = "./data/CIFAR-100-C/labels.npy"

args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
n_cls = 2 if 'binary' in args.dataset else 100


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

model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn)
# model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.n_filters_cnn, args.n_hidden_fc)
model = model.cuda()

if args.model_filename is None:
    for name in glob.glob(os.path.join(args.model_dir, "*.pth")): 
        print(name)
        model_path = name
else:
    model_path = os.path.join(args.model_dir, args.model_filename)

print('Evaluating model @ {}'.format(model_path))
eval_results_output = 'Evaluating model @ {}'.format(model_path)

# model_dict = torch.load('models/{}.pth'.format(args.model_path))
model_dict = torch.load(model_path)
# from training: torch.save({'last': model.state_dict(), 'best': best_state_dict}, os.path.join('models', model_name, '{}_epoch{}.pth'.format(model_name, epoch)))


if args.early_stopped_model:
    model.load_state_dict(model_dict['best'])
else:
    model.load_state_dict(model_dict['last'] if 'last' in model_dict else model_dict)

opt = torch.optim.SGD(model.parameters(), lr=0)  # needed for backprop only
# if half_prec:
#     model, opt = amp.initialize(model, opt, opt_level="O1")
utils.model_eval(model, half_prec)

# eps, pgd_alpha, pgd_alpha_rr = eps / 255, pgd_alpha / 255, pgd_alpha_rr / 255

eval_y = np.load(label_path)
eval_y = np.int64(eval_y)

results_dict = {}

# Clean evaluation
clean_eval_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                shuffle=False, data_augm=False)
acc_clean, loss_clean, _ = rob_acc(clean_eval_batches, model, 0, 0, opt, half_prec, 0, 1)

print('clean: acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean))
eval_results_output += '\n clean: acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean)

results_dict['clean'] = {"acc": acc_clean, "loss": loss_clean}

for corruption in corruptions:
    print("Evaluating {} corruption".format(corruption))
    eval_x = np.load(data_paths[corruption])
    eval_x = np.float32(eval_x)
    # eval_x = eval_x.permute(0, 3, 1, 2)
    # eval_x = eval_x.reshape(-1, 3, 32, 32)
    eval_x = eval_x/255

    results_dict[corruption] = {}

    current_corruption_errors = []
    for severity in severities:
        print("Evaluating #{} severity".format(severity))

        # CIFAR-100-C data loading
        eval_x_match_severity = eval_x[(severity-1)* 10000:severity* 10000]

        eval_y_match_severity = eval_y[(severity-1)* 10000:severity* 10000]

        eval_dataset = td.TensorDataset(torch.tensor(eval_x_match_severity).permute(0, 3, 1, 2), torch.tensor(eval_y_match_severity, dtype=torch.long))
        eval_batches = td.DataLoader(eval_dataset, batch_size=args.batch_size_eval) 

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