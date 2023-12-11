#! /bin/bash
model_card=bert-base-uncased
model=glowrepbert

gpunum=0
freeze_bert=2
batch_size=16

bert_lr=1e-5
non_bert_lr=1e-4

## Glow
n_block=2
n_flow=3
glow_lr=1e-4

dim_red=0
rep_type=last_avg

for data in msmarco
do
    for MAX_EPOCH in 3
    do
        for random_seed in 40 41 42
        do
            # for fold in f1 f2 f3 f4 f5
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
                exp="Glow_"$n_block$n_flow$glow_lr"_Prefix0001e-4_LoRA0001e-4_e"$MAX_EPOCH"_seed"$random_seed

                echo "training"
                outdir="PRE_"$pre_w"_POST_"$data"_"$model"_"$fold"_"$exp
                echo $outdir
                echo $model
                python train_seq.py \
                    --model $model \
                    --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                    --qrels data/$data/qrels \
                    --train_pairs data/$data/$fold.train.pairs \
                    --valid_run data/$data/$fold.valid.run \
                    --model_out_dir models/$outdir \
                    --initial_bert_weights $pre_path$pre_w"/weights.p" \
                    --max_epoch $MAX_EPOCH \
                    --gpunum $gpunum \
                    --random_seed $random_seed  \
                    --freeze_bert $freeze_bert \
                    --non_bert_lr $non_bert_lr  \
                    --bert_lr $bert_lr \
                    --grad_acc_size $batch_size \
                    --n_block $n_block \
                    --n_flow $n_flow \
                    --glow_lr $glow_lr \
                    --freeze_nice True \
                    --dim_red $dim_red \
                    --rep_type $rep_type \
                    --msmarco True \
                    --batches_per_epoch 1024

                echo "testing"
                python rerank.py \
                    --model $model \
                    --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                    --run data/$data/$fold.test.run \
                    --model_weights models/$outdir/flow_weights.p \
                    --out_path models/$outdir/test.run \
                    --gpunum $gpunum \
                    --n_block $n_block \
                    --n_flow $n_flow \
                    --freeze_nice True \
                    --dim_red $dim_red \
                    --rep_type $rep_type

                echo "evaluating"
                bin/trec_eval -m all_trec data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result

                echo "token-wise metrics"
                python metrics.py \
                    --model $model \
                    --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                    --run data/$data/$fold.test.run \
                    --model_weights models/$outdir/flow_weights.p \
                    --out_path models/$outdir/metrics_tw.run \
                    --gpunum $gpunum \
                    --n_block $n_block \
                    --n_flow $n_flow \
                    --freeze_nice True \
                    --dim_red $dim_red \
                    --rep_type $rep_type

                echo "sequence-wise metrics"
                python metrics.py \
                    --model $model \
                    --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                    --run data/$data/$fold.test.run \
                    --model_weights models/$outdir/flow_weights.p \
                    --out_path models/$outdir/metrics_sw.run \
                    --gpunum $gpunum \
                    --n_block $n_block \
                    --n_flow $n_flow \
                    --freeze_nice True \
                    --dim_red $dim_red \
                    --rep_type $rep_type \
                    --metrics_sw True
            done
        done
    done
done
