#! /bin/bash
model_card=bert-base-uncased
model=colbert

gpunum=0
MAX_EPOCH=10 ## 30
freeze_bert=0 ## 0: melt, 2: freeze

bert_lr=1e-5
non_bert_lr=1e-4

dim_red=0
wo_clinear=True

for data in msmarco
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
            exp="e"$MAX_EPOCH"_seed"$random_seed"_blr"$bert_lr"_nblr"$non_bert_lr

            # # 1. make ./models/$model/weights.p (weights file) in ./models.
            echo "training"
            outdir=$data"_"$model"_"$fold"_"$exp
            echo $outdir

            echo $model
            python train.py \
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
                --freeze_glow True \
                --freeze_nice True \
                --dim_red $dim_red \
                --wo_clinear $wo_clinear \
                --msmarco True \
                --batches_per_epoch 1024

            # 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
            echo "testing"
            python rerank.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --model_weights models/$outdir/weights.p \
                --run data/$data/$fold.test.run \
                --out_path models/$outdir/test.run \
                --freeze_glow True \
                --freeze_nice True \
                --dim_red $dim_red \
                --gpunum $gpunum \
                --wo_clinear $wo_clinear

            #3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
            echo "evaluating"
            bin/trec_eval -m all_trec data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result

            echo "metrics"
            python metrics.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --model_weights models/$outdir/weights.p \
                --run data/$data/$fold.test.run \
                --out_path models/$outdir/metrics.run \
                --freeze_glow True \
                --freeze_nice True \
                --dim_red $dim_red \
                --gpunum $gpunum \
                --wo_clinear $wo_clinear
        done
    done
done