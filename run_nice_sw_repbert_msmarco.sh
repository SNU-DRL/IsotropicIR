#! /bin/bash
data=msmarco
model_card=bert-base-uncased
model=nicerepbert

gpunum=0
freeze_bert=2
batch_size=16

bert_lr=1e-5
non_bert_lr=1e-4

## NICE
nice_lr=1e-4
nice_nhidden=1000
nice_nlayers=5
nice_prior=gaussian

dim_red=0
rep_type=last_avg

for MAX_EPOCH in 3
do
    for random_seed in 40 41 42
    do
        for fold in full
        do
            if [ ! -z "$1" ]; then
                fold=$1
            fi

            if [ ! -z "$2" ]; then
                random_seed=$2
            fi
            pre_path="/data2/xlpczv/IsotropicIR/models/"
            pre_w=$model_card"_"$data"_repbert_"$fold"_e10_seed"$random_seed"_blr1e-5_nblr1e-4"
            exp="NICE_sw_"$nice_prior$nice_nhidden$nice_nlayers$nice_lr"_e"$MAX_EPOCH"_seed"$random_seed

            # # 1. make ./models/$model/weights.p (weights file) in ./models.
            echo "training"
            outdir="PRE_"$pre_w"_POST_"$data"_"$model"_"$fold"_"$exp
            echo $outdir

            echo $model
            python train_seq.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --qrels data/$data/qrels \
                --train_pairs data/$data/$fold.train.pairs \
                --valid_run data/$data/$fold.valid.run \
                --model_out_dir models/$outdir \
                --max_epoch $MAX_EPOCH \
                --gpunum $gpunum \
                --random_seed $random_seed  \
                --freeze_bert $freeze_bert \
                --non_bert_lr $non_bert_lr  \
                --bert_lr $bert_lr \
                --initial_bert_weights $pre_path$pre_w"/weights.p" \
                --grad_acc_size $batch_size \
                --nice_nhidden $nice_nhidden \
                --nice_nlayers $nice_nlayers \
                --nice_prior $nice_prior \
                --nice_lr $nice_lr \
                --freeze_glow True \
                --dim_red $dim_red \
                --rep_type $rep_type \
                --flow_mf True \
                --msmarco True \
                --batches_per_epoch 1024

            # 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
            echo "testing"
            python rerank.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --run data/$data/$fold.test.run \
                --model_weights models/$outdir/flow_weights.p \
                --out_path models/$outdir/test.run \
                --gpunum $gpunum \
                --nice_nhidden $nice_nhidden \
                --nice_nlayers $nice_nlayers \
                --freeze_glow True \
                --dim_red $dim_red \
                --rep_type $rep_type \
                --flow_mf True

            #3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
            echo "evaluating"
            bin/trec_eval -m all_trec data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result

            echo "token-wise metrics"
            python metrics.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --run data/$data/$fold.test.run \
                --model_weights models/$outdir/flow_weights.p \
                --out_path models/$outdir/metrics_tw.run \
                --gpunum $gpunum \
                --nice_nhidden $nice_nhidden \
                --nice_nlayers $nice_nlayers \
                --freeze_glow True \
                --dim_red $dim_red \
                --rep_type $rep_type \
                --flow_mf True

            echo "sequence-wise metrics"
            python metrics.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --run data/$data/$fold.test.run \
                --model_weights models/$outdir/flow_weights.p \
                --out_path models/$outdir/metrics_sw.run \
                --gpunum $gpunum \
                --nice_nhidden $nice_nhidden \
                --nice_nlayers $nice_nlayers \
                --freeze_glow True \
                --dim_red $dim_red \
                --rep_type $rep_type \
                --metrics_sw True \
                --flow_mf True
        done
    done
done