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
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_binary', 'uniform_noise', 'cifar100'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    # parser.add_argument('--model_path', default='2020-03-19 23:51:05 dataset=cifar10 model=resnet18 eps=8.0 attack=pgd attack_init=zero fgsm_alpha=1.25 epochs=30 pgd_train_n_iters=7 grad_align_cos_lambda=0.0 seed=1 epoch=30',
    #                     type=str, help='model name')
    parser.add_argument('--model_dir', default='models/c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad',
                        type=str, help='model dir name')
    # parser.add_argument('--model_filename', default='c10rn18_eps8_atkpgd7_ep60_nogradalign_gradsparse02_4x4_normch_NLgrad_epoch60.pth',
    #                     type=str, help='model filename')
    parser.add_argument('--model_filename', default=None,
                        type=str, help='model filename')
    parser.add_argument('--eval_results_filename', default='fgsm_eval_results.txt',
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

    parser.add_argument('--fgsm_loss_func', default='xent', type=str, choices=['xent', 'cw'])

    return parser.parse_args()


args = get_args()
eps = args.eps
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_eval = args.n_eval
if args.dataset == 'cifar100':
    n_cls = 100
else:
    n_cls = 2 if 'binary' in args.dataset else 10
# print("n_cls: ", n_cls)
n_sampling_attack = 40
fgsm_attack_iters = 50
fgsm_alpha, fgsm_alpha_rr, alpha_fgm = args.eps/4, args.eps/4, 300.0
fgsm_rr_n_iter, fgsm_rr_n_restarts = (50, 10)

if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

results_path = os.path.join(args.model_dir, args.eval_results_filename)

if args.early_stopped_model:
    model.load_state_dict(model_dict['best'])
else:
    model.load_state_dict(model_dict['last'] if 'last' in model_dict else model_dict)

opt = torch.optim.SGD(model.parameters(), lr=0)  # needed for backprop only
# if half_prec:
#     model, opt = amp.initialize(model, opt, opt_level="O1")
utils.model_eval(model, half_prec)

eps, fgsm_alpha, fgsm_alpha_rr = eps / 255, fgsm_alpha / 255, fgsm_alpha_rr / 255

eval_batches_all = data.get_loaders(args.dataset, -1, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                    shuffle=False, data_augm=False)
eval_batches = data.get_loaders(args.dataset, n_eval, args.batch_size_eval, train_set=True if args.set == 'train' else False,
                                shuffle=False, data_augm=False)

time_start = time.time()

acc_clean, loss_clean, _ = rob_acc(eval_batches, model, 0, 0, opt, half_prec, 0, 1)

print('clean acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean))
eval_results_output += 'clean acc={:.2%}, loss={:.3f}'.format(acc_clean, loss_clean)

acc_fgsm_rr, loss_fgsm_rr, delta_fgsm_rr = rob_acc(eval_batches, model, eps, eps, opt, half_prec, 1, 1, rs=False, print_fosc=False, pgd_loss_func=args.fgsm_loss_func)
time_elapsed = time.time() - time_start

print('[test on {} points] acc_clean {:.2%}, fgsm_rr {:.2%} ({:.2f}m)'.format(n_eval, acc_clean, acc_fgsm_rr, time_elapsed/60))
eval_results_output += '\n[test on {} points] acc_clean {:.2%}, fgsm_rr {:.2%} ({:.2f}m)'.format(n_eval, acc_clean, acc_fgsm_rr, time_elapsed/60)


with open(results_path, 'w') as f:
    f.write(eval_results_output)