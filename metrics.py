import argparse
import train
import data
import os
import torch
import sys

def main_cli():
    MODEL_MAP = train.modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model re-ranking')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--model_card', default='bert-base-uncased', help='pretrained model card')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--gpunum', type=str, default="0", help='gup number')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--dim_red', type=int, default=0, help='dimension to reduce to')
    parser.add_argument('--rep_type', default='cls', help='type of representation', choices=['cls', 'last_avg', 'last2_avg'])
    parser.add_argument('--whitening', default=False, type=bool, help='whether to whiten or not')
    parser.add_argument('--whitening_tokenwise', default=False, type=bool, help='whether to whiten tokenwise or not')
    parser.add_argument('--whitening_mu_file', default=None, type=str, help='whitening mu file name')
    parser.add_argument('--whitening_cov_file', default=None, type=str, help='whitening cov file name')
    parser.add_argument('--whitening_k', default=None, type=int, help='whitening k')
    parser.add_argument('--wo_clinear', default=False, type=bool, help='(Only for ColBERT) whether to use clinear or not')
    parser.add_argument('--wo_mask', default=False, type=bool, help='(Only for ColBERT) whether to use mask for input or not')
    parser.add_argument('--centering', default=False, type=bool, help='whether to center or not')
    parser.add_argument('--centering_tokenwise', default=False, type=bool, help='whether to center tokenwise or not')
    parser.add_argument('--whitening_flow', type=bool, default=False, help='whitening -> flow')
    parser.add_argument('--flow_whitening', type=bool, default=False, help='flow -> whitening')

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
    
    ## metrics
    parser.add_argument('--normalize', type=bool, default=False, help='whether to normalize the vectors or not')
    parser.add_argument('--metrics_sw', type=bool, default=False, help='whether to compute metrics sequencewise or not')

    ## ColBERT after clinear
    parser.add_argument('--after_clinear', type=bool, default=False, help='whether to use representations after clinear or not')
    
    args = parser.parse_args()
    #setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())

    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)
    with open(args.run.name, 'r') as f:
        lines=f.readlines()

    sys.path.append("/data2/xlpczv/FLOW_BERTIR/nice_pytorch")
    from nice.loss import LogisticPriorNICELoss, GaussianPriorNICELoss
    from nice.utils import rescale, l1_norm

    if args.model.startswith("hq"):
        model = MODEL_MAP[args.model.replace("hq","")](args, hetero_query=True).cuda()
    elif args.model.startswith("hv"):
        model = MODEL_MAP[args.model.replace("hv","")](args, hetero_value=True).cuda()
    elif args.model.startswith("p2"):
        model = MODEL_MAP[args.model.replace("p2","")](args, SS_prefix=True).cuda()
    else:
        model = MODEL_MAP[args.model](args).cuda()
                                    
    if(args.model_weights is not None):
        wts = args.model_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load(wts[0], wts[1])

    train.compute_metrics(args, model, dataset, run, lines, args.out_path.name, LogisticPriorNICELoss, GaussianPriorNICELoss, l1_norm, desc='metrics')

if __name__ == '__main__':
    main_cli()
