import logging
import math
import os
# import apex.amp as amp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt

from autoattack import AutoAttack


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def get_grad_np(model, batches, eps, opt, half_prec, rs=False, cross_entropy=True):
    grad_list = []
    for i, (X, y) in enumerate(batches):
        X, y = X.cuda(), y.cuda()

        if rs:
            delta = get_uniform_delta(X.shape, eps, requires_grad=False)
        else:
            delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        logits = model(clamp(X + delta, 0, 1))

        if cross_entropy:
            loss = F.cross_entropy(logits, y)
        else:
            y_onehot = torch.zeros([len(y), 10]).long().cuda()
            y_onehot.scatter_(1, y[:, None], 1)
            preds_correct_class = (logits * y_onehot.float()).sum(1, keepdim=True)
            margin = preds_correct_class - logits  # difference between the correct class and all other classes
            margin += y_onehot.float() * 10000  # to exclude zeros coming from f_correct - f_correct
            margin = margin.min(1, keepdim=True)[0]
            loss = F.relu(1 - margin).mean()


        loss.backward()
        grad = delta.grad.detach().cpu()
        grad_list.append(grad.numpy())
        delta.grad.zero_()
    grads = np.vstack(grad_list)
    return grads


def get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none', backprop=False, return_delta=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)

    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
        
    if return_delta:
        return grad, delta
    else:
        return grad


def configure_logger(model_name, debug, log_dir='logs'):
    logging.basicConfig(format='%(message)s')  # , level=logging.DEBUG)
    logger = logging.getLogger()
    logger.handlers = []  # remove the default logger

    # add a new logger for stdout
    formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if not debug:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # add a new logger to a log file
        logger.addHandler(logging.FileHandler(os.path.join(log_dir, '{}.log'.format(model_name))))

    return logger


def to_eval_halfprec(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval().half()


def to_train_halfprec(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train().float()


def to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def to_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def model_eval(model, half_prec):
    model.eval()


def model_train(model, half_prec):
    model.train()


def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def get_gaussian_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta = eps * torch.randn(*delta.shape)
    delta.requires_grad = requires_grad
    return delta


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def attack_pgd_training(model, X, y, eps, alpha, opt, half_prec, attack_iters, rs=True, early_stopping=False):
    delta = torch.zeros_like(X).cuda()
    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(clamp(X + delta, 0, 1))
        loss = F.cross_entropy(output, y)

        loss.backward()
        grad = delta.grad.detach()

        if early_stopping:
            idx_update = output.max(1)[1] == y
        else:
            idx_update = torch.ones(y.shape, dtype=torch.bool)
        grad_sign = sign(grad)
        delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
        delta.data = clamp(X + delta.data, 0, 1) - X
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()

    return delta.detach()


def attack_pgd_original_only_xent(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True, loss_func='xent'):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = F.cross_entropy(output, y)

            loss.backward()
            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data = delta + alpha * sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta



def attack_pgd(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True, loss_func='xent'):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if loss_func == 'xent':
                loss = F.cross_entropy(output, y)
            # CW loss function
            elif loss_func == 'cw':
                y_unsqueeze = torch.unsqueeze(y, dim=-1)
                correct_logit = torch.gather(output, 1, y_unsqueeze)

                logit_bias = torch.zeros_like(output)
                if cuda:
                    logit_bias = logit_bias.cuda()

                logit_bias = torch.scatter(logit_bias, 1, y_unsqueeze, value=-float('inf'))
                max_wrong_logit = torch.max(output+logit_bias, 1, keepdim=True)[0]

                loss = -F.relu(correct_logit - max_wrong_logit + 50)
                loss = torch.mean(loss)

            loss.backward()
            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data = delta + alpha * sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta



def rob_acc(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=False, cuda=True, pgd_loss_func='xent'):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=rs,
                               verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda, loss_func=pgd_loss_func)
        if corner:
            pgd_delta = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X
        pgd_delta_proj = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)
    return robust_acc, avg_loss, pgd_delta_np


def rob_acc_autoattack(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=False, cuda=True, pgd_loss_func='xent'):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0

    
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard')
    print('autoattack adversary: ', adversary)

    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()

        x_adv = adversary.run_standard_evaluation(X, y, bs=X.shape[0])

        with torch.no_grad():
            output = model(x_adv)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    return robust_acc, avg_loss
        

def compute_low_freq_bias(fourier_map, max_pow=1, min_pow=-1, epsilon = 1e-05, reduce_type='sum', log_value=True, temperature=1, ignore_first_basis=False):

    if ignore_first_basis:
        fourier_map_to_use = fourier_map.clone()
        fourier_map_to_use[0,0] = 0
    else:
        fourier_map_to_use = fourier_map

    pow_range = torch.linspace(start=max_pow, end=min_pow, steps=fourier_map_to_use.shape[0]).cuda() + epsilon

    if reduce_type == 'sumlog':
        fourier_map_to_log = fourier_map_to_use + epsilon # torch.Size([17, 17])
        fourier_map_log = torch.log(fourier_map_to_log)  # torch.Size([17, 17])

        log_dim0 = fourier_map_log * torch.unsqueeze(pow_range, dim=0) # torch.Size([17, 17])
        sum_log_dim0 = torch.sum(log_dim0, dim=1) # torch.Size([17])

        log_dim1 = fourier_map_log * torch.unsqueeze(pow_range, dim=1) # torch.Size([17, 17])
        sum_log_dim1 = torch.sum(log_dim1, dim=0) # torch.Size([17])

        total_logsum = torch.sum(sum_log_dim0) + torch.sum(sum_log_dim1) # scalar

        return total_logsum
    
    fourier_map_to_pow = fourier_map_to_use + epsilon # torch.Size([17, 17])
    
    pow_dim0 = torch.pow(fourier_map_to_pow, torch.unsqueeze(pow_range, dim=0))
    product_dim0 = torch.prod(pow_dim0, dim=1) # torch.Size([17])
    
    pow_dim1 = torch.pow(fourier_map_to_pow, torch.unsqueeze(pow_range, dim=1))
    product_dim1 = torch.prod(pow_dim1, dim=0) # torch.Size([17])
    
    if reduce_type=='sum': # default
        reduced_fm_product = torch.sum(product_dim0) + torch.sum(product_dim1) # scalar
    elif reduce_type=='product':
        reduced_fm_product = torch.prod(product_dim0) * torch.prod(product_dim1)
        
    if log_value:
        return torch.log(reduced_fm_product/temperature)
    else:
        return reduced_fm_product


def analyze_corruption_fourier_and_freq_bias(clean_batches, cor_batches, half_prec=False, analysis_output_dir="debug_cor_fourier", delta_init='none', cuda=True, output_dir_suffix='', 
                    max_pow=1, min_pow=-1, temperature=1, constant_threshold=1):
    cor_it = iter(cor_batches)
    
    all_cor_diff_freq_norm = []
    all_grad_freq_norm = []
    all_img_freq_norm = []
    saved_ig = []
    for i, (X, y) in enumerate(clean_batches):
        X_cor, y_cor = next(cor_it)
        if cuda:
            X, y = X.cuda(), y.cuda()
            X_cor, y_cor = X_cor.cuda(), y_cor.cuda()
        
        cor_diff = X_cor - X
        
        # compute dft of imgs
        cor_diff_fourier_map = torch.rfft(cor_diff, signal_ndim=2)
        cor_diff_freq_norm = torch.norm(cor_diff_fourier_map, dim=-1)
        all_cor_diff_freq_norm.append(cor_diff_freq_norm)

        # compute dft of imgs
        img_fourier_map = torch.rfft(X, signal_ndim=2)
        img_freq_norm = torch.norm(img_fourier_map, dim=-1)
        all_img_freq_norm.append(img_freq_norm)

    save_dir = os.path.join(analysis_output_dir, 'corruption_fourier_analysis'+output_dir_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # draw fourier map of imgs
    all_cor_diff_freq_norm = torch.cat(all_cor_diff_freq_norm, dim=0)
    mean_cor_diff_freq_norm = torch.mean(all_cor_diff_freq_norm, dim=(0,1))

    # draw fourier map of imgs
    all_img_freq_norm = torch.cat(all_img_freq_norm, dim=0)
    mean_img_freq_norm = torch.mean(all_img_freq_norm, dim=(0,1))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_img_freq_norm[:1+mean_img_freq_norm.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'img_fourier_map.jpeg'))
    torch.save(mean_img_freq_norm, os.path.join(save_dir, 'img_fourier_map.pt'))

    mean_img_freq_norm_zeroconstant = mean_img_freq_norm[:]
    mean_img_freq_norm_zeroconstant[0,0] = 0
    mean_img_freq_norm_thresholdconstant = mean_img_freq_norm[:]
    mean_img_freq_norm_thresholdconstant[0,0] = torch.max((mean_img_freq_norm_zeroconstant[:1+mean_img_freq_norm_zeroconstant.shape[0]//2])) * constant_threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_img_freq_norm_thresholdconstant[:1+mean_img_freq_norm_thresholdconstant.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'img_fourier_map_thresholdconstant.jpeg'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_cor_diff_freq_norm[:1+mean_cor_diff_freq_norm.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'cor_diff_fourier_map.jpeg'))
    torch.save(mean_cor_diff_freq_norm, os.path.join(save_dir, 'cor_diff_fourier_map.pt'))

    # compute fourier map's low frequency bias value
    cor_diff_low_freq_bias_value = compute_low_freq_bias(mean_cor_diff_freq_norm[:1+mean_cor_diff_freq_norm.shape[0]//2], max_pow=1, min_pow=-1, temperature=1)

    mean_cor_diff_freq_norm_zeroconstant = mean_cor_diff_freq_norm[:]
    mean_cor_diff_freq_norm_zeroconstant[0,0] = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_cor_diff_freq_norm_zeroconstant[:1+mean_cor_diff_freq_norm_zeroconstant.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'cor_diff_fourier_map_zeroconstant.jpeg'))

    mean_cor_diff_freq_norm_thresholdconstant = mean_cor_diff_freq_norm[:]
    mean_cor_diff_freq_norm_thresholdconstant[0,0] = torch.max((mean_cor_diff_freq_norm_zeroconstant[:1+mean_cor_diff_freq_norm_zeroconstant.shape[0]//2])) * constant_threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_cor_diff_freq_norm_thresholdconstant[:1+mean_cor_diff_freq_norm_thresholdconstant.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'cor_diff_fourier_map_thresholdconstant.jpeg'))

    return cor_diff_low_freq_bias_value

        
def analyze_save_ig(batches, model, opt, eps, half_prec, model_output_dir, delta_init='none', cuda=True, num_saved_ig=20, output_dir_suffix='', 
                    max_pow=1, min_pow=-1, temperature=1, constant_threshold=1):
    all_img_freq_norm = []
    all_grad_freq_norm = []
    saved_ig = []
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        
        # compute dft of imgs
        img_fourier_map = torch.rfft(X, signal_ndim=2)
        img_freq_norm = torch.norm(img_fourier_map, dim=-1)
        all_img_freq_norm.append(img_freq_norm)

        # compute dft of input gradients
        grad = get_input_grad(model, X, y, opt, eps, half_prec, delta_init=delta_init)
        grad_fourier_map = torch.rfft(grad, signal_ndim=2)
        grad_freq_norm = torch.norm(grad_fourier_map, dim=-1)
        all_grad_freq_norm.append(grad_freq_norm)

        if i * grad.shape[0] < num_saved_ig:
            saved_ig.append(grad)
    
    save_dir = os.path.join(model_output_dir, 'grad_fourier_analysis'+output_dir_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # draw input gradient samples
    if num_saved_ig > 0:
        saved_ig = torch.cat(saved_ig, dim=0)
        for ind in range(num_saved_ig):
            grad = saved_ig[ind]
            for ch_ind, ch_ig in enumerate(grad):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.imshow(ch_ig.cpu(), interpolation='nearest')
                cbar = fig.colorbar(cax)
                fig.savefig(os.path.join(save_dir, 'ig_sample{}ch{}.jpeg'.format(ind, ch_ind)))
        torch.save(saved_ig, os.path.join(save_dir, 'saved_ig.pt'))

    # draw fourier map of imgs
    all_img_freq_norm = torch.cat(all_img_freq_norm, dim=0)
    mean_img_freq_norm = torch.mean(all_img_freq_norm, dim=(0,1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_img_freq_norm[:1+mean_img_freq_norm.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'img_fourier_map.jpeg'))
    torch.save(mean_img_freq_norm, os.path.join(save_dir, 'img_fourier_map.pt'))

    # compute fourier map's low frequency bias value
    img_low_freq_bias_value = compute_low_freq_bias(mean_img_freq_norm[:1+mean_img_freq_norm.shape[0]//2], max_pow=1, min_pow=-1, temperature=1)

    mean_img_freq_norm_zeroconstant = mean_img_freq_norm[:]
    mean_img_freq_norm_zeroconstant[0,0] = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_img_freq_norm_zeroconstant[:1+mean_img_freq_norm_zeroconstant.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'img_fourier_map_zeroconstant.jpeg'))

    mean_img_freq_norm_thresholdconstant = mean_img_freq_norm[:]
    mean_img_freq_norm_thresholdconstant[0,0] = torch.max((mean_img_freq_norm_zeroconstant[:1+mean_img_freq_norm_zeroconstant.shape[0]//2])) * constant_threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_img_freq_norm_thresholdconstant[:1+mean_img_freq_norm_thresholdconstant.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'img_fourier_map_thresholdconstant.jpeg'))


    # draw fourier map of input gradients
    all_grad_freq_norm = torch.cat(all_grad_freq_norm, dim=0)
    mean_grad_freq_norm = torch.mean(all_grad_freq_norm, dim=(0,1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_grad_freq_norm[:1+mean_grad_freq_norm.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'grad_fourier_map.jpeg'))
    torch.save(mean_grad_freq_norm, os.path.join(save_dir, 'grad_fourier_map.pt'))

    grad_low_freq_bias_value = compute_low_freq_bias(mean_grad_freq_norm[:1+mean_grad_freq_norm.shape[0]//2], max_pow=1, min_pow=-1, temperature=1)

    mean_grad_freq_norm[0,0] = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mean_grad_freq_norm[:1+mean_grad_freq_norm.shape[0]//2].cpu(), interpolation='nearest')
    cbar = fig.colorbar(cax)
    fig.savefig(os.path.join(save_dir, 'grad_fourier_map_zeroconstant.jpeg'))

    return grad_low_freq_bias_value, img_low_freq_bias_value


def compute_fourier_map(grad, batch_average=True):
    grad_fourier_map = torch.rfft(grad, signal_ndim=2)
    grad_freq_norm = torch.norm(grad_fourier_map, dim=-1)
    
    # average over channels
    if batch_average:
        chmean_grad_freq_norm = torch.mean(grad_freq_norm, dim=(0,1))
    else:
        chmean_grad_freq_norm = torch.mean(grad_freq_norm, dim=1)

    return chmean_grad_freq_norm

def model_params_to_list(model):
    list_params = []
    model_params = list(model.parameters())
    for param in model_params:
        list_params.append(param.data.clone())
    return list_params


def avg_cos_np(v1, v2):
    norms1 = np.sum(v1 ** 2, (1, 2, 3), keepdims=True) ** 0.5
    norms2 = np.sum(v2 ** 2, (1, 2, 3), keepdims=True) ** 0.5
    cos_vals = np.sum(v1/norms1 * v2/norms2, (1, 2, 3))
    cos_vals[np.isnan(cos_vals)] = 1.0  # to prevent nans (0/0)
    cos_vals[np.isinf(cos_vals)] = 1.0  # to prevent +infs and -infs (x/0, -x/0)
    avg_cos = np.mean(cos_vals)
    return avg_cos


def avg_l2_np(v1, v2=None):
    if v2 is not None:
        diffs = v1 - v2
    else:
        diffs = v1
    diff_norms = np.sum(diffs ** 2, (1, 2, 3)) ** 0.5
    avg_norm = np.mean(diff_norms)
    return avg_norm


def avg_fraction_same_sign(v1, v2):
    v1 = np.sign(v1)
    v2 = np.sign(v2)
    avg_cos = np.mean(v1 == v2)
    return avg_cos


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    return norms


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        n = module.in_features
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
    elif lr_schedule_type == 'piecewise':
        def lr_schedule(t):
            if t / n_epochs < 0.5:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    else:
        raise ValueError('wrong lr_schedule_type')
    return lr_schedule


def backward(loss, opt, half_prec):

    loss.backward()
