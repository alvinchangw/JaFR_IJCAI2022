import argparse
import os
import utils
# import apex.amp as amp
import numpy as np
import torch
import time
import data
import models
from utils import rob_acc
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'uniform_noise', 'cifar100', 'tinyimagenet'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet34', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--model_dir', default='models/c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad',
                        type=str, help='model dir name')
    parser.add_argument('--model_filename', default=None,
                        type=str, help='model filename')
    parser.add_argument('--eval_results_filename', default='eval_results.txt',
                        type=str, help='evaluation result output filename')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model in [fc, cnn])')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#filters on each conv layer (for model==fc)')
    parser.add_argument('--batch_size_eval', default=128, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--early_stopped_model', action='store_true', help='eval the best model according to pgd_acc evaluated every k iters (typically, k=200)')

    parser.add_argument('--pgd_loss_func', default='xent', type=str, choices=['xent', 'cw'])
    parser.add_argument('--do_fgsm', action='store_true', help='eval fgsm instead of pgd')

    return parser.parse_args()


args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
if args.dataset == 'cifar100':
    n_cls = 100
elif args.dataset == 'tinyimagenet':
    n_cls = 200
else:
    n_cls = 2 if 'binary' in args.dataset else 10

# print("n_cls: ", n_cls)
n_sampling_attack = 40
pgd_attack_iters = 50
pgd_alpha, pgd_alpha_rr, alpha_fgm = args.eps/4, args.eps/4, 300.0
pgd_rr_n_iter, pgd_rr_n_restarts = (50, 10)

if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

model_dict = torch.load(model_path)

results_path = os.path.join(args.model_dir, args.eval_results_filename)

if args.early_stopped_model:
    model.load_state_dict(model_dict['best'])
else:
    model.load_state_dict(model_dict['last'] if 'last' in model_dict else model_dict)

opt = torch.optim.SGD(model.parameters(), lr=0)  # needed for backprop only
utils.model_eval(model, half_prec)

eps, pgd_alpha, pgd_alpha_rr = eps / 255, pgd_alpha / 255, pgd_alpha_rr / 255

eval_batches_all = data.get_loaders(args.dataset, -1, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                    shuffle=False, data_augm=False)
eval_batches = data.get_loaders(args.dataset, n_eval, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                shuffle=False, data_augm=False)

time_start = time.time()

acc_clean, loss_clean, _ = rob_acc(eval_batches, model, 0, 0, opt, half_prec, 0, 1)

print('clean acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean))
eval_results_output += 'clean acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean)

if args.do_fgsm:
    acc_pgd_rr, loss_pgd_rr, delta_pgd_rr = rob_acc(eval_batches, model, eps, eps, opt, half_prec, 1, 1, rs=False)
else:
    acc_pgd_rr, loss_pgd_rr, delta_pgd_rr = rob_acc(eval_batches, model, eps, pgd_alpha_rr, opt, half_prec, pgd_rr_n_iter, pgd_rr_n_restarts, print_fosc=False, pgd_loss_func=args.pgd_loss_func)

time_elapsed = time.time() - time_start

print('[test on {} points] acc_clean {:.2%}, pgd_rr {:.2%} ({:.2f}m)'.format(n_eval, acc_clean, acc_pgd_rr, time_elapsed/60))
eval_results_output += '\n[test on {} points] acc_clean {:.2%}, pgd_rr {:.2%} ({:.2f}m)'.format(n_eval, acc_clean, acc_pgd_rr, time_elapsed/60)


with open(results_path, 'w') as f:
    f.write(eval_results_output)