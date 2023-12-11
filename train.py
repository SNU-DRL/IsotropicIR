import os
from selectors import EpollSelector
import sys
import argparse
import subprocess
import random
from numpy import True_
from tqdm import tqdm
import torch
import torch.nn.functional as F
import modeling
import data
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from IsoScore import IsoScore
import csv
import time
import math
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

trec_eval_f = 'bin/trec_eval'

def setRandomSeed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args, model, dataset, train_pairs, qrels, valid_run, qrelf):
    _verbose = False
    _logf = os.path.join(args.model_out_dir, 'train.log')
    print(f'learning_rate nonbert={args.non_bert_lr} bert={args.bert_lr}')
    torch.autograd.set_detect_anomaly(True)

    ## freeze_bert
    model_name = type(model).__name__
    if(args.freeze_bert == 2):
        model.freeze_bert()

    ## parameter update setting
    bert_params, non_bert_params, glow_params, nice_params = model.get_params()
    optim_bert_params = {'params': bert_params, 'lr':args.bert_lr}
    optim_non_bert_params = {'params': non_bert_params, 'lr':args.non_bert_lr}
    optim_params=[optim_non_bert_params]
    if args.freeze_bert == 0:
        optim_params.append(optim_bert_params)
        print("adding bert params to optim_params")

    optimizer = torch.optim.Adam(optim_params, weight_decay=args.weight_decay)

    ## scheduler
    warmup_step = args.warmup_epoch * args.batches_per_epoch
    max_step = args.max_epoch * args.batches_per_epoch
    if args.scheduler is not None:
        print("Using linear scheduler...")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, max_step, last_epoch=-1)
    else:
        print("Scheduler is None")
        scheduler = None

    ## training & validation
    logf = open(_logf, "w")
    print(f'max_epoch={args.max_epoch}', file=logf)
    epoch = 0
    top_valid_score = None
    for epoch in range(args.max_epoch):
        if args.msmarco:
            loss = train_marco_iteration(args, model, optimizer, scheduler, train_pairs)
        else:
            loss = train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}', file=logf)

        valid_score = validate(args, model, dataset, valid_run, qrelf, epoch)
        print(f'validation epoch={epoch} score={valid_score}')
        print(f'validation epoch={epoch} score={valid_score}', file=logf)
        
        if (top_valid_score is None) or (valid_score > top_valid_score):
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            print(f'newtopsaving epoch={epoch} score={top_valid_score}', file=logf)
            
            if(args.freeze_bert >= 1):
                model.save(os.path.join(args.model_out_dir, 'weights.p'), without_bert=True)
            else:
                model.save(os.path.join(args.model_out_dir, 'weights.p'))

        logf.flush()

    print(f'topsaving score={top_valid_score}', file=logf)

def train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels):
    total = 0
    model.train()
    total_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, args.grad_acc_size): 
            scores = model(record['query_tok'],
                        record['query_mask'],
                        record['doc_tok'],
                        record['doc_mask'])
            count = len(record['query_id']) // 2

            ## score estimator
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            # print("loss.device", loss.device)
            loss.backward()

            total_loss += loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()

                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                model.bert.zero_grad()
                
            pbar.update(count)
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss

def train_marco_iteration(args, model, optimizer, scheduler, train_pairs):
    total = 0
    model.train()
    total_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_marco_train_pairs(model, train_pairs, args.grad_acc_size): 
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            loss.backward()
            total_loss += loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                model.bert.zero_grad()
            pbar.update(count)
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss

def validate(args, model, dataset, run, qrelf, epoch):
    if args.msmarco:
        VALIDATION_METRIC = 'recip_rank'
    else:
        VALIDATION_METRIC = 'P.20'
    runf = os.path.join(args.model_out_dir, f'{epoch}.run')
    run_model(args, model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(args, model, dataset, run, runf, desc='valid'):
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, args.batch_size):
            if (args.freeze_glow) and (args.freeze_nice):
                scores = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'])
            elif not args.freeze_glow:
                scores, _ = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'])
            elif not args.freeze_nice:
                scores, _, _ = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def compute_metrics(args, model, dataset, run, lines, runf, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, desc='metrics'):
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        flow_loss = 0
        cnt = 0
        with open(runf, 'w') as runfile:
            pass

        for records in data.iter_test_shfl_record(model, dataset, lines, args.batch_size):
            if args.model == 'colbert' and args.after_clinear:
                scores, out_tuple = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'],
                            value_return=True,
                            after_clinear=True)
            elif 'nice' not in args.model and 'glow' not in args.model:
                scores, out_tuple = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'],
                            value_return=True)
            elif 'nice' in args.model:
                scores, out_tuple, _ = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'],
                            value_return=True)
                ## Flow loss
                flow_loss += measure_nice_loss(args, model, out_tuple[:2], LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm)

            elif 'glow' in args.model:
                scores, _out_tuple = model(records['query_tok'],
                            records['query_mask'],
                            records['doc_tok'],
                            records['doc_mask'],
                            value_return=True)
                out_tuple = [_out_tuple[2].squeeze(), _out_tuple[5].squeeze(), _out_tuple[-2], _out_tuple[-1]]
            
            if len(out_tuple) > 1: ## bi-encoder
                if args.metrics_sw: ## only in repbert
                    assert('colbert' not in args.model)
                    q_z = out_tuple[2].squeeze()
                    d_z = out_tuple[3].squeeze()
                else:
                    q_z = out_tuple[0].squeeze()
                    d_z = out_tuple[1].squeeze()

                q_z = q_z.reshape(-1, q_z.shape[-1])
                d_z = d_z.reshape(-1, d_z.shape[-1])

                q_z = q_z[q_z.sum(1) != 0] ## q_z.sum(1).shape = [bs * seq_len]
                d_z = d_z[d_z.sum(1) != 0]

                if args.normalize:
                    q_z = F.normalize(q_z, p=2, dim=-1)
                    d_z = F.normalize(d_z, p=2, dim=-1)

                ## Isotropy scores
                Iso_q = measure_isotropy_all(q_z)
                Iso_d = measure_isotropy_all(d_z)
                Iso_qd = measure_isotropy_all(torch.cat((q_z, d_z), 0))
                
                with open(runf, 'a') as runfile:
                    qid = records['query_id']
                    did = records['doc_id']
                    writer = csv.writer(runfile)
                    writer.writerow(list(qid) + list(did) + list(Iso_q) + list(Iso_d) + list(Iso_qd) + [float(flow_loss)])

            elif len(out_tuple) == 1:
                z = out_tuple[0]
                if args.normalize:
                    z = F.normalize(z, p=2, dim=-1)

                Iso = measure_isotropy_all(z)

                with open(runf, 'a') as runfile:
                    qid = records['query_id']
                    did = records['doc_id']
                    writer = csv.writer(runfile)
                    writer.writerow(list(qid) + list(did) + list(Iso))

            pbar.update(len(records['query_id']))
        
def trec_eval(qrelf, runf, metric):
    print("qrelf",qrelf)
    print("runf", runf)
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

## make x a leaf variable (requires_grad=True)
def make_leaf_variable(x):
    x = x.data
    x.requires_grad=True
    return x

def to_np(x):
    return x.detach().cpu().numpy()

## I(W)
def measure_I_W(W):
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    u, s, v = torch.svd(W) ## v.shape=[dim, n_data]
    z = torch.exp(torch.matmul(W, v)).sum(0) ## (W*v).shape=[n_data, n_data]
    i_w = z.min() / z.max()
    return float(i_w)

def measure_avg_cos(W):
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    n_data = W.shape[0]
    W_norm = torch.norm(W, dim=1)
    W /= W_norm.unsqueeze(1)
    WW = torch.matmul(W, W.T) ## WW.shape[n_data, n_data]
    triu_numel = n_data * (n_data-1) / 2
    cos = torch.sum(torch.triu(WW, diagonal=1)) / triu_numel
    return float(cos)

def measure_largest_eigval(W):
    assert(len(W.shape) == 2)
    ## W: embedding matrix, shape=[n_data, dim]
    eig_vals, _ = torch.linalg.eigh(torch.matmul(W.T, W))
    largest_eigval_ratio = torch.max(eig_vals) / torch.sum(eig_vals)
    return float(largest_eigval_ratio)

def measure_Isoscore(W):
    return IsoScore.IsoScore(to_np(W))

def measure_isotropy_all(W, qd=False):
    I_W = measure_I_W(W)
    avg_cos = measure_avg_cos(W)
    largest_eigval = measure_largest_eigval(W)
    if qd:
        Isoscore = measure_Isoscore(W)
        return (I_W, avg_cos, largest_eigval, Isoscore)
    else:
        return (I_W, avg_cos, largest_eigval)

def measure_distances_btw_points(A, B, same=False):
    ## A.shape=[bs, m, d]
    ## B.shape=[bs, n, d]
    ## output.shape=[bs]
    assert(len(A.shape)==3)
    AB = torch.cdist(A, B) ## AB.shape=[bs, m, n]
    if same: ## A=B
        assert(A.shape[1] == B.shape[1])
        n = A.shape[1]
        triu_sum = torch.sum(torch.triu(AB), dim=(1,2))
        triu_numel = n * (n-1) / 2
        return triu_sum / triu_numel
    else: ## A!=B
        return torch.mean(AB, dim=(1,2))

def measure_within_between_ratio(A, B):
    ## A.shape=[bs, m, d]
    ## B.shape=[bs, n, d]
    ## output type=scalar
    A_within = measure_distances_btw_points(A, A, same=True)
    B_within = measure_distances_btw_points(B, B, same=True)
    AB_between = measure_distances_btw_points(A, B)
    ratio_A = torch.mean(AB_between / A_within)
    ratio_B = torch.mean(AB_between / B_within)
    return ratio_A, ratio_B

def measure_nice_loss(args, nice_model, z_tuple, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm):
    if args.nice_prior == 'logistic':
        nice_loss_fn = LogisticPriorNICELoss(size_average=True)
    else:
        nice_loss_fn = GaussianPriorNICELoss(size_average=True)

    def loss_fn(fx):
        """Compute NICE loss w/r/t a prior and optional L1 regularization."""
        if args.nice_lmbda == 0.0:
            return nice_loss_fn(fx, nice_model.nice.scaling_diag)
        else:
            return nice_loss_fn(fx, nice_model.nice.scaling_diag) + args.nice_lmbda*l1_norm(nice_model, include_bias=True)
    
    if len(z_tuple) == 2:
        q_z, d_z = z_tuple
        if args.flow_qnd:
            flow_loss = loss_fn(torch.cat((q_z, d_z), 0))
        else:
            flow_loss = loss_fn(q_z) + loss_fn(d_z)
    elif len(z_tuple) == 1:
        z = z_tuple[0]
        flow_loss = loss_fn(z)
    return flow_loss

def compute_token_rep(args, model, out_dir, layer=11):
    batch_size=args.batch_size

    if 'roberta' in args.model:
        cls_id = model.tokenizer.vocab['<s>']
        sep_id = model.tokenizer.vocab['</s>']
        pad_id = model.tokenizer.vocab['<pad>']
    else:
        cls_id = model.tokenizer.vocab['[CLS]']
        sep_id = model.tokenizer.vocab['[SEP]']
        pad_id = model.tokenizer.vocab['[PAD]']
    
    vocab_size = model.tokenizer.vocab_size
    rep = torch.zeros([vocab_size, model.BERT_SIZE])

    for iter in range(math.ceil(vocab_size // batch_size)):
        input_ids = torch.cat((torch.full(size=(batch_size, 1), fill_value=cls_id, dtype=torch.long).cuda(), torch.arange(iter*batch_size, min((iter+1)*batch_size, vocab_size)).unsqueeze(1).cuda(), torch.full(size=(batch_size, 1), fill_value=sep_id, dtype=torch.long).cuda()), 1)
        # print(input_ids.shape) ## shape=[batch_size, 3]
        rep[iter*batch_size:(iter+1)*batch_size] = model.bert(input_ids, output_hidden_states=True)[2][layer][:, 1, :].data ## shape=[batch_size, BERT_SIZE]
    
    rep_mean = torch.mean(rep, 0) ## shape=
    print("rep_mean.shape", rep_mean.shape)
    rep_std = torch.std(rep, 0)
    print("rep_std.shape", rep_std.shape)
    torch.save(rep, out_dir + "token_rep_l" + str(layer) + ".pt")
    torch.save(rep_mean, out_dir + "token_rep_mean_l" + str(layer) + ".pt")
    torch.save(rep_std, out_dir + "token_rep_std_l" + str(layer) + ".pt")

def get_mean_from_out_ls(args, out_ls):
    """Calculate mean for whitening."""
    qry_out, doc_out = out_ls ## qry_out.shape=[bs, seq_len, BERT_SIZE] for ColBERT, RepBERT, and GlowColBERT / qry_out.shape=[-1, BERT_SIZE] for NICERepBERT and GlowRepBERT
    
    if len(qry_out.shape) == 3:
        if args.whitening_tokenwise:
            qry_mask = (torch.sum(qry_out, -1) != 0).long() ## qry_mask.shape=[bs, seq_len]
            doc_mask = (torch.sum(doc_out, -1) != 0).long()
            qry_sum = torch.sum(qry_out, [0,1])  ## qry_sum.shape=[BERT_SIZE]
            doc_sum = torch.sum(doc_out, [0,1])
            _sum = qry_sum + doc_sum          ## _sum.shape=[BERT_SIZE]
            m = qry_mask.sum() + doc_mask.sum()    ## Calculate m considering maskings. m: Total # of meaning tokens.
        else:
            if 'colbert' in args.model:
                raise ValueError
            qry_mask = (torch.sum(qry_out, -1) != 0).long() ## qry_mask.shape=[bs, seq_len]
            doc_mask = (torch.sum(doc_out, -1) != 0).long()
            _qry_mean = torch.sum(qry_out, 1).data / qry_mask.sum(-1).unsqueeze(-1) ## qry_mean.shape=[bs, BERT_SIZE]
            _doc_mean = torch.sum(doc_out, 1).data / doc_mask.sum(-1).unsqueeze(-1)
            _sum = torch.sum(_qry_mean, 0) + torch.sum(_doc_mean, 0)   ## sum.shape=[BERT_SIZE]
            m = qry_out.shape[0] * 2
    elif len(qry_out.shape) == 2:
        qry_sum = torch.sum(qry_out, axis=0) ## qry_sum.shape = [dim]
        doc_sum = torch.sum(doc_out, axis=0)
        _sum = qry_sum + doc_sum
        m = qry_out.shape[0] + doc_out.shape[0]
    else:
        raise ValueError
    return m, _sum

def get_cov_from_out_ls(args, out_ls, mean):
    """Calculate covariance for whitening.
       mean.shape = [BERT_SIZE]"""
    qry_out, doc_out = out_ls ## qry_out.shape=[bs, seq_len, BERT_SIZE] for ColBERT, RepBERT, and GlowColBERT / qry_out.shape=[-1, BERT_SIZE] for NICERepBERT and GlowRepBERT

    if len(qry_out.shape) == 3:
        if args.whitening_tokenwise:
            qry_mask = (torch.sum(qry_out, -1) != 0).long() ## shape=[bs, seq_len]
            doc_mask = (torch.sum(doc_out, -1) != 0).long()
            m = qry_mask.sum() + doc_mask.sum()   ## Calculate m considering maskings. m: Total # of meaning tokens.
            _qry_cov_sum = torch.matmul(((qry_out - mean.reshape(1,1, mean.shape[0])) * qry_mask.unsqueeze(-1)).permute(0,2,1), ((qry_out - mean.reshape(1,1, mean.shape[0])) * qry_mask.unsqueeze(-1))).data ## shape=[bs, BERT_SIZE, BERT_SIZE] (Sum over seq_len)
            _doc_cov_sum = torch.matmul(((doc_out - mean.reshape(1,1, mean.shape[0])) * doc_mask.unsqueeze(-1)).permute(0,2,1), ((doc_out - mean.reshape(1,1, mean.shape[0])) * doc_mask.unsqueeze(-1))).data ## shape=[bs, BERT_SIZE, BERT_SIZE] (Sum over seq_len)
            _cov_sum = torch.sum(_qry_cov_sum, 0) + torch.sum(_doc_cov_sum, 0) ## shape=[BERT_SIZE, BERT_SIZE]
        else:
            if 'colbert' in args.model:
                raise ValueError
            qry_mask = (torch.sum(qry_out, -1) != 0).long() ## shape=[bs, seq_len]
            doc_mask = (torch.sum(doc_out, -1) != 0).long()
            _qry_mean = torch.sum(qry_out, 1).data / qry_mask.sum(-1).unsqueeze(-1) ## qry_mean.shape=[bs, BERT_SIZE]
            _doc_mean = torch.sum(doc_out, 1).data / doc_mask.sum(-1).unsqueeze(-1)
            m = qry_out.shape[0] * 2
            _qry_cov_sum = torch.matmul((_qry_mean - mean.reshape(1, mean.shape[0])).permute(1,0), (_qry_mean - mean.reshape(1, mean.shape[0]))).data ## shape=[BERT_SIZE, BERT_SIZE] (Sum over bs)
            _doc_cov_sum = torch.matmul((_doc_mean - mean.reshape(1, mean.shape[0])).permute(1,0), (_doc_mean - mean.reshape(1, mean.shape[0]))).data ## shape=[BERT_SIZE, BERT_SIZE] (Sum over bs)
            _cov_sum = _qry_cov_sum + _doc_cov_sum ## shape=[BERT_SIZE, BERT_SIZE]
    elif len(qry_out.shape) == 2:
        m = qry_out.shape[0] + doc_out.shape[0]
        _qry_cov_sum = torch.matmul((qry_out - mean.reshape(1, mean.shape[0])).permute(1,0), (qry_out - mean.reshape(1, mean.shape[0]))).data ## _qry_cov_sum.shape = [dim, dim]
        _doc_cov_sum = torch.matmul((doc_out - mean.reshape(1, mean.shape[0])).permute(1,0), (doc_out - mean.reshape(1, mean.shape[0]))).data
        _cov_sum = _qry_cov_sum + _doc_cov_sum ## shape=[BERT_SIZE, BERT_SIZE]
    else:
        raise ValueError
    return m, _cov_sum

def compute_train_mean_std(args, model, train_pairs, dataset, qrels, out_dir, layer=11):
    ## Only works for layer11 now
    n=0
    mean = torch.zeros([model.BERT_SIZE]).to(model.bert.device)
    cov = torch.zeros([model.BERT_SIZE, model.BERT_SIZE]).to(model.bert.device)

    if args.msmarco:
        # Calculate sample mean first.
        for iter in range(args.whitening_n_epochs):
            print(iter)
            _n = 0
            for record in data.iter_marco_train_pairs(model, train_pairs, args.grad_acc_size):
                # print(record['query_tok'])
                _, out_ls = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'],
                                whitening_value_return=True)
                count = len(record['query_id']) // 2
                m, _sum = get_mean_from_out_ls(args, out_ls)

                mean = (n / (n + m)) * mean + (1 / (n + m)) * _sum.data  # mean.shape = [BERT_SIZE]
                _n += count
                n += m
                if _n >= args.batch_size * args.batches_per_epoch:
                    break
        # Calculate sample covariance.
        n = 0
        for iter in range(args.whitening_n_epochs):
            print(iter)
            _n = 0
            for record in data.iter_marco_train_pairs(model, train_pairs, args.grad_acc_size): 
                _, out_ls = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'],
                                whitening_value_return=True)
                count = len(record['query_id']) // 2
                m, _cov_sum = get_cov_from_out_ls(args, out_ls, mean)
                
                if n == 0: 
                    cov = (1 / (m - 1)) * _cov_sum
                else:
                    cov = ((n - 1) / (n + m - 1)) * cov + (1 / (n + m - 1)) * _cov_sum.data
                _n += count
                n += m
                if _n >= args.batch_size * args.batches_per_epoch:
                    break
    else:
        # Calculate sample mean first.
        for iter in range(args.whitening_n_epochs):
            print(iter)
            _n = 0
            for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, args.grad_acc_size):
                _, out_ls = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'],
                                whitening_value_return=True)
                count = len(record['query_id']) // 2
                m, _sum = get_mean_from_out_ls(args, out_ls)

                mean = (n / (n + m)) * mean + (1 / (n + m)) * _sum.data  # mean.shape = [BERT_SIZE]
                _n += count
                n += m
                # print("mean _n", _n)
                if _n >= args.batch_size * args.batches_per_epoch:
                    break
        # Calculate sample covariance.
        n = 0
        for iter in range(args.whitening_n_epochs):
            print(iter)
            _n = 0
            for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, args.grad_acc_size):
                _, out_ls = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'],
                                whitening_value_return=True)
                count = len(record['query_id']) // 2
                m, _cov_sum = get_cov_from_out_ls(args, out_ls, mean)
                
                if n == 0: 
                    cov = (1 / (m - 1)) * _cov_sum
                else:
                    cov = ((n - 1) / (n + m - 1)) * cov + (1 / (n + m - 1)) * _cov_sum.data
                _n += count
                n += m
                # print("cov _n", _n)
                if _n >= args.batch_size * args.batches_per_epoch:
                    break

    if args.whitening_n_epochs==1:
        wne = ""
    else:
        wne = "_e" + str(args.whitening_n_epochs)

    if args.whitening_tokenwise:
        torch.save(mean, out_dir + "train_mean_l" + str(layer) + wne + ".pt")
        print("saved mean (tokenwise)")
        torch.save(cov, out_dir + "train_cov_l" + str(layer) + wne + ".pt")
        print("saved cov (tokenwise)")
    else:
        torch.save(mean, out_dir + "sw_train_mean_l" + str(layer) + wne + ".pt")
        print("saved mean (sequencewise)")
        torch.save(cov, out_dir + "sw_train_cov_l" + str(layer) + wne + ".pt")
        print("saved cov (sequencewise)")

def main_cli():
    MODEL_MAP = modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model training and validation')
    ## model
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vbert')
    parser.add_argument('--model_card', default='bert-base-uncased', help='pretrained model card')
    parser.add_argument('--initial_bert_weights', type=str, default=None)
    parser.add_argument('--model_out_dir')
    parser.add_argument('--rep_type', default='cls', help='type of representation', choices=['cls', 'last_avg', 'last2_avg'])
    parser.add_argument('--whitening', default=False, type=bool, help='whether to whiten or not')
    parser.add_argument('--whitening_tokenwise', default=True, type=bool, help='whether to whiten tokenwise or not')
    parser.add_argument('--wo_clinear', default=False, type=bool, help='(Only for ColBERT) whether to use clinear or not')
    parser.add_argument('--wo_mask', default=False, type=bool, help='(Only for ColBERT) whether to use mask for input or not')
    parser.add_argument('--centering', default=False, type=bool, help='whether to center or not')
    parser.add_argument('--centering_tokenwise', default=False, type=bool, help='whether to center tokenwise or not')

    ## data
    parser.add_argument('--msmarco', default=False, type=bool, help='whether to use ms marco or not')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))

    ## training
    parser.add_argument('--gpunum', type=str, default="0", help='gpu number')
    parser.add_argument('--random_seed', type=int, default=42, help='ranodm seed number')    
    parser.add_argument('--freeze_bert', type=int, default=0, help='freezing bert')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--grad_acc_size', type=int, default=2, help='gradient accumulation size')
    parser.add_argument('--batches_per_epoch', type=int, default=64, help='# batches per epoch')
    parser.add_argument('--scheduler', type=str, default=None, help='learning rate scheduler (None/linear are  only possible)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate(bert)')
    parser.add_argument('--non_bert_lr', type=float, default=1e-4, help='learning rate(non-bert)')
    parser.add_argument('--dim_red', type=int, default=0, help='dimension to reduce to')

    ## Flow
    parser.add_argument('--freeze_glow', type=bool, default=False, help='freezing glow')
    parser.add_argument('--freeze_nice', type=bool, default=False, help='freezing NICE')

    args = parser.parse_args()

    setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())

    print("Load Model")
    model = MODEL_MAP[args.model](args).cuda()

    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    if args.msmarco:
        train_pairs = args.train_pairs
    else:
        train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    ## initial
    if(args.initial_bert_weights is not None):
        wts = args.initial_bert_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load_duet(wts[0], wts[1])

    os.makedirs(args.model_out_dir, exist_ok=True)
    main(args, model, dataset, train_pairs, qrels, valid_run, args.qrels.name)

if __name__ == '__main__':
    main_cli()
