#! /bin/bash
model_card=bert-base-uncased
model=colbert

gpunum=2
MAX_EPOCH=10
wne=_e$MAX_EPOCH

bert_lr=1e-5
non_bert_lr=1e-4

dim_red=0
wo_clinear=True

for data in robust wt
do
    for random_seed in 40 41 42
    do
        for fold in f1 f2 f3 f4 f5
        do
            if [ ! -z "$1" ]; then
                fold=$1
            fi

            if [ ! -z "$2" ]; then
                random_seed=$2
            fi
            outdir=$data"_colbert_"$fold"_e30_seed"$random_seed"_blr"$bert_lr"_nblr"$non_bert_lr

            # 2. Compute mean and std. vectors for whitening
            echo "token_rep"
            python token_rep.py \
                --model $model \
                --model_card $model_card \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --qrels data/$data/qrels \
                --train_pairs data/$data/$fold.train.pairs \
                --model_weights models/$outdir/weights.p \
                --out_path models/$outdir/ \
                --gpunum $gpunum \
                --dim_red $dim_red \
                --whitening_n_epochs $MAX_EPOCH \
                --wo_clinear $wo_clinear \
                --whitening_tokenwise True

            echo "whitening testing"
            python rerank.py \
                --model $model \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --run data/$data/$fold.test.run \
                --model_weights models/$outdir/weights.p \
                --out_path models/$outdir/"whitening.run" \
                --dim_red $dim_red \
                --gpunum $gpunum \
                --freeze_glow True \
                --freeze_nice True \
                --whitening True \
                --whitening_mu_file models/$outdir/train_mean_l11$wne.pt \
                --whitening_cov_file models/$outdir/train_cov_l11$wne.pt \
                --whitening_tokenwise True \
                --wo_clinear $wo_clinear

            echo "evaluating"
            bin/trec_eval -m all_trec data/$data/qrels models/$outdir/"whitening.run" > models/$outdir/"whitening_eval.result"

            echo "whitening metrics"
            python metrics.py \
                --model $model \
                --datafiles data/$data/queries.tsv data/$data/documents.tsv \
                --run data/$data/$fold.test.run \
                --model_weights models/$outdir/weights.p \
                --out_path models/$outdir/"whitening_metrics.run" \
                --dim_red $dim_red \
                --gpunum $gpunum \
                --freeze_glow True \
                --freeze_nice True \
                --whitening True \
                --whitening_mu_file models/$outdir/train_mean_l11$wne.pt \
                --whitening_cov_file models/$outdir/train_cov_l11$wne.pt \
                --whitening_tokenwise True \
                --wo_clinear $wo_clinear
        done
    done
done