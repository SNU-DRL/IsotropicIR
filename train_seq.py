import os
import sys
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from math import log
import numpy as np

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

def main(args, model, dataset, train_pairs, qrels, valid_run, qrelf, LogisticPriorNICELoss, GaussianPriorNICELoss, rescale, l1_norm):
    _verbose = False
    _flow_logf = os.path.join(args.model_out_dir, 'flow_train.log')
    _aggregator_logf = os.path.join(args.model_out_dir, 'aggregator_train.log')
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
    optim_glow_params = {'params': glow_params, 'lr': args.glow_lr}
    optim_nice_params = {'params': nice_params, 'lr': args.nice_lr}

    ############################## 1. Train FLOW ##############################
    optim_params=[]
    if args.freeze_bert == 0:
        optim_params.append(optim_bert_params)
        print("adding bert params to optim_params")

    if not args.freeze_glow:
        optim_params.append(optim_glow_params)
        print("adding glow params to optim_params")

    if not args.freeze_nice:
        optim_params.append(optim_nice_params)
        print("adding nice params to optim_params")

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
    logf = open(_flow_logf, "w")
    print(f'max_epoch={args.max_epoch}', file=logf)
    epoch = 0
    top_valid_score = None
    if args.save_all_checkpoints:
        model.save(os.path.join(args.model_out_dir, 'flow_weights_e0_' + str(0) + '.p'))
    
    for epoch in range(args.max_epoch):
        if args.msmarco:
            loss, flow_loss = train_marco_iteration(args, model, optimizer, scheduler, train_pairs, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=False, flow_loss_backward=True, save_checkpoints=(epoch==0 and args.save_all_checkpoints))
        else:
            loss, flow_loss = train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=False, flow_loss_backward=True, save_checkpoints=(epoch==0 and args.save_all_checkpoints))
        print(f'train epoch={epoch} loss={loss} flow_loss={flow_loss}', file=logf)
        print(f'train epoch={epoch} loss={loss} flow_loss={flow_loss}')

        if args.save_all_checkpoints:
            model.save(os.path.join(args.model_out_dir, 'flow_weights_e' + str(epoch) + '.p'))

        if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
            if(args.freeze_bert >= 1):
                model.save(os.path.join(args.model_out_dir, 'flow_weights1.p'), os.path.join(args.model_out_dir, 'flow_weights2.p'), without_bert=True)
            else:
                model.save(os.path.join(args.model_out_dir, 'flow_weights1.p'), os.path.join(args.model_out_dir, 'flow_weights2.p'))
        else:
            model.save(os.path.join(args.model_out_dir, 'flow_weights.p'))

        logf.flush()

    print(f'topsaving score={top_valid_score}', file=logf)

    ############################## 2. Train Aggregator ##############################
    if args.train_aggregator:
        optim_params=[optim_non_bert_params]
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
        logf = open(_aggregator_logf, "w")
        print(f'max_epoch={args.max_epoch}', file=logf)
        epoch = 0
        top_valid_score = None
        for epoch in range(args.max_epoch):
            if args.msmarco:
                loss, flow_loss = train_marco_iteration(args, model, optimizer, scheduler, train_pairs, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=True, flow_loss_backward=False)
            else:
                loss, flow_loss = train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=True, flow_loss_backward=False)
            print(f'train epoch={epoch} loss={loss} flow_loss={flow_loss}', file=logf)
            print(f'train epoch={epoch} loss={loss} flow_loss={flow_loss}')

            valid_score = validate(args, model, dataset, valid_run, qrelf, epoch)
            print(f'validation epoch={epoch} score={valid_score}')
            print(f'validation epoch={epoch} score={valid_score}', file=logf)
            
            if (top_valid_score is None) or (valid_score > top_valid_score):
                top_valid_score = valid_score
                print('new top validation score, saving weights')
                print(f'newtopsaving epoch={epoch} score={top_valid_score}', file=logf)
                if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                    model.save(os.path.join(args.model_out_dir, 'agg_weights1.p'), os.path.join(args.model_out_dir, 'agg_weights2.p'))
                else:
                    model.save(os.path.join(args.model_out_dir, 'agg_weights.p'))

            logf.flush()

        print(f'topsaving score={top_valid_score}', file=logf)

def train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=True, flow_loss_backward=True, save_checkpoints=False):
    total = 0
    model.train()
    total_loss = 0.
    total_flow_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, args.grad_acc_size): 
            if not args.freeze_glow:
                scores, glow_tuple = model(record['query_tok'],
                            record['query_mask'],
                            record['doc_tok'],
                            record['doc_mask'])
                
                if len(glow_tuple) == 6:
                    q_log_p_sum, q_logdet, q_z, d_log_p_sum, d_logdet, d_z = glow_tuple ## q_z.shape=[bs, 3, 16, 16], q_sldj.shape=[4]
                elif len(glow_tuple) == 3:
                    log_p_sum, logdet, z = glow_tuple
            
            elif not args.freeze_nice:
                if args.per_seq:
                    scores, nice_tuple, mask_tuple = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'])
                    if len(nice_tuple) == 2:
                        q_z, d_z = nice_tuple
                        q_msk, d_msk = mask_tuple
                    elif len(nice_tuple) == 1:
                        z = nice_tuple[0]
                        msk = mask_tuple[0]
                else:
                    scores, nice_tuple, _ = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'])
                    if len(nice_tuple) == 2:
                        q_z, d_z = nice_tuple
                    elif len(nice_tuple) == 1:
                        z = nice_tuple[0]
            
            else:
                scores = model(record['query_tok'],
                            record['query_mask'],
                            record['doc_tok'],
                            record['doc_mask'])

            count = len(record['query_id']) // 2

            ## score estimator
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            if loss_backward:
                loss.backward()

            ## Glow LOSS (TODO: per_seq, qnd)
            if not args.freeze_glow:
                if len(glow_tuple) == 6:
                    flow_loss = ((-1) * (q_log_p_sum + q_logdet) / (log(2) * model.BERT_SIZE)).mean() + ((-1) * (d_log_p_sum + d_logdet) / (log(2) * model.BERT_SIZE)).mean()
                    flow_loss += args.flow_act_reg * ((torch.norm(q_z) + torch.norm(d_z)) / 2)
                elif len(glow_tuple) == 3:
                    flow_loss = ((-1) * (log_p_sum + logdet) / (log(2) * model.BERT_SIZE)).mean()
                    flow_loss += args.flow_act_reg * torch.norm(z)
            
            ### NICE LOSS
            elif not args.freeze_nice:
                if args.nice_prior == 'logistic':
                    nice_loss_fn = LogisticPriorNICELoss(size_average=True)
                else:
                    nice_loss_fn = GaussianPriorNICELoss(size_average=True)

                def loss_fn(fx):
                    """Compute NICE loss w/r/t a prior and optional L1 regularization."""
                    if args.nice_lmbda == 0.0:
                        return nice_loss_fn(fx, model.nice.scaling_diag)
                    else:
                        return nice_loss_fn(fx, model.nice.scaling_diag) + args.nice_lmbda*l1_norm(model, include_bias=True)
                
                if args.per_seq: ## NICE only
                    if len(nice_tuple) == 2:
                        if type(q_msk) == int:
                            q_z_split = list(torch.split(q_z, q_z.shape[0] // (args.grad_acc_size * 2), dim=0))
                            d_z_split = list(torch.split(d_z, d_z.shape[0] // (args.grad_acc_size * 2), dim=0))
                        else:
                            q_z_split = list(torch.split(q_z, list(q_msk.sum(1).long()), dim=0))
                            d_z_split = list(torch.split(d_z, list(d_msk.sum(1).long()), dim=0))
                        # print("len(q_z_split)", len(q_z_split), "len(d_z_split)", len(d_z_split))
                        flow_loss = 0
                        for e in range(len(q_z_split)):
                            _q_z = q_z_split[e]
                            _d_z = d_z_split[e]
                                
                            if not flow_loss_backward:
                                _q_z = make_leaf_variable(_q_z)
                                _d_z = make_leaf_variable(_d_z)

                            ## args.flow_qnd: True -> query and document together to Flow model
                            if args.flow_qnd:
                                _flow_loss = loss_fn(torch.cat((_q_z, _d_z), 0))
                            else:
                                _flow_loss = loss_fn(_q_z) + loss_fn(_d_z)
                            _flow_loss += args.flow_act_reg * ((torch.norm(_q_z) + torch.norm(_d_z)) / 2)
                            flow_loss += _flow_loss
                
                    elif len(nice_tuple) == 1:
                        z_split = list(torch.split(z, list(msk.sum(1).long()), dim=0))
                        flow_loss = 0
                        for e in range(len(z_split)):
                            _z = z_split[e]
                            if not flow_loss_backward:
                                _z = make_leaf_variable(_z)
                            _flow_loss = loss_fn(_z)
                            _flow_loss += args.flow_act_reg * torch.norm(_z)
                            flow_loss += _flow_loss

                else:
                    if len(nice_tuple) == 2:
                        if not flow_loss_backward:
                            q_z = make_leaf_variable(q_z)
                            d_z = make_leaf_variable(d_z)

                        ## args.flow_qnd: True -> query and document together to Flow model
                        if args.flow_qnd:
                            flow_loss = loss_fn(torch.cat((q_z, d_z), 0))
                        else:
                            flow_loss = loss_fn(q_z) + loss_fn(d_z)
                        flow_loss += args.flow_act_reg * ((torch.norm(q_z) + torch.norm(d_z)) / 2)
                    elif len(nice_tuple) == 1:
                        if not flow_loss_backward:
                            z = make_leaf_variable(z)
                        flow_loss = loss_fn(z)
                        flow_loss += args.flow_act_reg * torch.norm(z)
            
            ## ELSE LOSS
            else:
                flow_loss  = torch.zeros(1)

            ## FLOW backward
            if flow_loss_backward:
                flow_loss.backward()

            total_loss += loss.item()
            total_flow_loss += flow_loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                    model.bert1.zero_grad()
                    model.bert2.zero_grad()
                else:
                    model.bert.zero_grad()
            pbar.update(count)

            if (save_checkpoints) and (total % args.batch_size == 0) and (total < 100):
                model.save(os.path.join(args.model_out_dir, 'flow_weights_e0_' + str(total) + '.p'))
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss, total_flow_loss
            

def train_marco_iteration(args, model, optimizer, scheduler, train_pairs, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, loss_backward=True, flow_loss_backward=True, save_checkpoints=False):
    total = 0
    model.train()
    total_loss = 0.
    total_flow_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_marco_train_pairs(model, train_pairs, args.grad_acc_size): 
            if not args.freeze_glow:
                scores, glow_tuple = model(record['query_tok'],
                            record['query_mask'],
                            record['doc_tok'],
                            record['doc_mask'])
                
                if len(glow_tuple) == 6:
                    q_log_p_sum, q_logdet, q_z, d_log_p_sum, d_logdet, d_z = glow_tuple ## q_z.shape=[bs, 3, 16, 16], q_sldj.shape=[4]
                elif len(glow_tuple) == 3:
                    log_p_sum, logdet, z = glow_tuple
            
            elif not args.freeze_nice:
                if args.per_seq:
                    scores, nice_tuple, mask_tuple = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'])
                    if len(nice_tuple) == 2:
                        q_z, d_z = nice_tuple
                        q_msk, d_msk = mask_tuple
                    elif len(nice_tuple) == 1:
                        z = nice_tuple[0]
                        msk = mask_tuple[0]
                else:
                    scores, nice_tuple, _ = model(record['query_tok'],
                                record['query_mask'],
                                record['doc_tok'],
                                record['doc_mask'])
                    if len(nice_tuple) == 2:
                        q_z, d_z = nice_tuple
                    elif len(nice_tuple) == 1:
                        z = nice_tuple[0]
            
            else:
                scores = model(record['query_tok'],
                            record['query_mask'],
                            record['doc_tok'],
                            record['doc_mask'])
            
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            if loss_backward:
                loss.backward()

            ## Glow LOSS (TODO: per_seq, qnd)
            if not args.freeze_glow:
                if len(glow_tuple) == 6:
                    flow_loss = ((-1) * (q_log_p_sum + q_logdet) / (log(2) * model.BERT_SIZE)).mean() + ((-1) * (d_log_p_sum + d_logdet) / (log(2) * model.BERT_SIZE)).mean()
                    flow_loss += args.flow_act_reg * ((torch.norm(q_z) + torch.norm(d_z)) / 2)
                elif len(glow_tuple) == 3:
                    flow_loss = ((-1) * (log_p_sum + logdet) / (log(2) * model.BERT_SIZE)).mean()
                    flow_loss += args.flow_act_reg * torch.norm(z)
            
            ### NICE LOSS
            elif not args.freeze_nice:
                if args.nice_prior == 'logistic':
                    nice_loss_fn = LogisticPriorNICELoss(size_average=True)
                else:
                    nice_loss_fn = GaussianPriorNICELoss(size_average=True)

                def loss_fn(fx):
                    """Compute NICE loss w/r/t a prior and optional L1 regularization."""
                    if args.nice_lmbda == 0.0:
                        return nice_loss_fn(fx, model.nice.scaling_diag)
                    else:
                        return nice_loss_fn(fx, model.nice.scaling_diag) + args.nice_lmbda*l1_norm(model, include_bias=True)
                
                if args.per_seq:
                    if len(nice_tuple) == 2:
                        if type(q_msk) == int:
                            q_z_split = list(torch.split(q_z, q_z.shape[0] // (args.grad_acc_size * 2), dim=0))
                            d_z_split = list(torch.split(d_z, d_z.shape[0] // (args.grad_acc_size * 2), dim=0))
                        else:
                            q_z_split = list(torch.split(q_z, list(q_msk.sum(1).long()), dim=0))
                            d_z_split = list(torch.split(d_z, list(d_msk.sum(1).long()), dim=0))
                        flow_loss = 0
                        for e in range(len(q_z_split)):
                            _q_z = q_z_split[e]
                            _d_z = d_z_split[e]
                                
                            if not flow_loss_backward:
                                _q_z = make_leaf_variable(_q_z)
                                _d_z = make_leaf_variable(_d_z)

                            ## args.flow_qnd: True -> query and document together to Flow model
                            if args.flow_qnd:
                                _flow_loss = loss_fn(torch.cat((_q_z, _d_z), 0))
                            else:
                                _flow_loss = loss_fn(_q_z) + loss_fn(_d_z)
                            _flow_loss += args.flow_act_reg * ((torch.norm(_q_z) + torch.norm(_d_z)) / 2)
                            # print(args.flow_act_reg * ((torch.norm(q_z) + torch.norm(d_z)) / 2))
                            flow_loss += _flow_loss
                
                    elif len(nice_tuple) == 1:
                        z_split = list(torch.split(z, list(msk.sum(1).long()), dim=0))
                        flow_loss = 0
                        for e in range(len(z_split)):
                            _z = z_split[e]
                            if not flow_loss_backward:
                                _z = make_leaf_variable(_z)
                            _flow_loss = loss_fn(_z)
                            _flow_loss += args.flow_act_reg * torch.norm(_z)
                            flow_loss += _flow_loss

                else:
                    if len(nice_tuple) == 2:
                        if not flow_loss_backward:
                            q_z = make_leaf_variable(q_z)
                            d_z = make_leaf_variable(d_z)
                        
                        ## args.flow_qnd: True -> query and document together to Flow model
                        if args.flow_qnd:
                            flow_loss = loss_fn(torch.cat((q_z, d_z), 0))
                        else:
                            flow_loss = loss_fn(q_z) + loss_fn(d_z)
                    elif len(nice_tuple) == 1:
                        if not flow_loss_backward:
                            z = make_leaf_variable(z)
                        flow_loss = loss_fn(z)
                        flow_loss += args.flow_act_reg * torch.norm(z)
                
            ## ELSE LOSS
            else:
                flow_loss  = torch.zeros(1)

            ## FLOW backward
            if flow_loss_backward:
                flow_loss.backward()

            total_loss += loss.item()
            total_flow_loss += flow_loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                    model.bert1.zero_grad()
                    model.bert2.zero_grad()
                else:
                    model.bert.zero_grad()
            pbar.update(count)
            
            if (save_checkpoints) and (total % args.batch_size == 0) and (total < 100):
                model.save(os.path.join(args.model_out_dir, 'flow_weights_e0_' + str(total) + '.p'))
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss, total_flow_loss
            
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

def main_cli():
    MODEL_MAP = modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model training and validation')
    ## model
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vbert')
    parser.add_argument('--model_card', default='bert-base-uncased', help='pretrained model card')
    parser.add_argument('--initial_bert_weights', type=str, default=None)
    parser.add_argument('--model_out_dir')
    parser.add_argument('--rep_type', default='cls', help='type of representation', choices=['cls', 'last_avg', 'last2_avg'])
    parser.add_argument('--wo_clinear', default=False, type=bool, help='(Only for ColBERT) whether to use clinear or not')
    parser.add_argument('--wo_mask', default=False, type=bool, help='(Only for ColBERT) whether to use mask for input or not')

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
    parser.add_argument('--train_aggregator', type=bool, default=False, help='whether to train aggregator or not')
    parser.add_argument('--save_all_checkpoints', type=bool, default=False, help='whether to save checkpoints of all epochs or not')

    ## FLOW
    parser.add_argument('--hetero_flow', type=bool, default=False, help='whether to use heterogeneous FLOW for query and document or not')
    parser.add_argument('--flow_mf', type=bool, default=False, help='False (default): FLOW -> mean, else: mean -> FLOW')
    parser.add_argument('--flow_qnd', type=bool, default=False, help='put query and document together to FLOW model')
    parser.add_argument('--flow_act_reg', type=float, default=0, help='weight decay coefficient of flow activations - 0: NO weight decay')
    parser.add_argument('--per_seq', type=bool, default=False, help='whether to train nice per sequence or not')

    ## Glow
    parser.add_argument('--freeze_glow', type=bool, default=False, help='freezing glow')
    # parser.add_argument('--n_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--n_block', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--n_flow', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--glow_lr', type=float, default=1e-4, help='learning rate(glow)')
    parser.add_argument('--glow_two_conv', type=bool, default=False, help='Whether to use two 1x1 conv in Glow')

    ## NICE
    parser.add_argument('--nice_layernorm', type=bool, default=False, help='whether to put layernorm to the end of NICE or not')
    parser.add_argument('--freeze_nice', type=bool, default=False, help='freezing NICE')
    parser.add_argument('--nice_nhidden', type=int, default=2000, help='number of hidden units for NICE')
    parser.add_argument('--nice_nlayers', type=int, default=4, help='number of hidden layers for NICE')
    parser.add_argument('--nice_prior', type=str, default='logistic', help='prior type of nice', choices=['logistic', 'gaussian'])
    parser.add_argument('--nice_lr', type=float, default=1e-4, help='learning rate(nice)')
    parser.add_argument('--nice_lmbda', type=float, default=0.0, help='lambda for nice')
    
    ## Whitening
    parser.add_argument('--whitening_flow', type=bool, default=False, help='whitening -> flow')
    parser.add_argument('--flow_whitening', type=bool, default=False, help='flow -> whitening')
    parser.add_argument('--whitening_mu_file', default=None, type=str, help='whitening mu file name')
    parser.add_argument('--whitening_cov_file', default=None, type=str, help='whitening cov file name')
    parser.add_argument('--whitening_tokenwise', default=False, type=bool, help='whether to whiten tokenwise or not')
    parser.add_argument('--whitening_k', default=None, type=int, help='whitening k')

    args = parser.parse_args()

    setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())
    
    sys.path.append("nice_pytorch")
    from nice.loss import LogisticPriorNICELoss, GaussianPriorNICELoss
    from nice.utils import rescale, l1_norm

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
    main(args, model, dataset, train_pairs, qrels, valid_run, args.qrels.name, LogisticPriorNICELoss, GaussianPriorNICELoss, rescale, l1_norm)


if __name__ == '__main__':
    main_cli()
