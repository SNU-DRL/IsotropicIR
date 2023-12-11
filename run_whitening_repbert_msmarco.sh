#! /bin/bash
data=msmarco
model_card=bert-base-uncased
model=repbert

gpunum=0
MAX_EPOCH=3
wne=_e$MAX_EPOCH

bert_lr=1e-5
non_bert_lr=1e-4

dim_red=0
rep_type=last_avg

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
        outdir=$model_card"_"$data"_repbert_"$fold"_e10_seed"$random_seed"_blr1e-5_nblr1e-4"

        # 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
        echo "token_rep"
        ## Sequence-wise
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
            --rep_type $rep_type \
            --whitening_n_epochs $MAX_EPOCH \
            --msmarco True \
            --batches_per_epoch 1024

        # ## Token-wise
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
            --rep_type $rep_type \
            --whitening_n_epochs $MAX_EPOCH \
            --whitening_tokenwise True \
            --msmarco True \
            --batches_per_epoch 1024

        echo "whitening testing"
        ## Sequence-wise
        python rerank.py \
            --model $model \
            --datafiles data/$data/queries.tsv data/$data/documents.tsv \
            --run data/$data/$fold.test.run \
            --model_weights models/$outdir/weights.p \
            --out_path models/$outdir/"sw_whitening.run" \
            --dim_red $dim_red \
            --gpunum $gpunum \
            --freeze_glow True \
            --freeze_nice True \
            --rep_type $rep_type \
            --whitening True \
            --whitening_mu_file models/$outdir/sw_train_mean_l11$wne.pt \
            --whitening_cov_file models/$outdir/sw_train_cov_l11$wne.pt

        ## Token-wise
        python rerank.py \
            --model $model \
            --datafiles data/$data/queries.tsv data/$data/documents.tsv \
            --run data/$data/$fold.test.run \
            --model_weights models/$outdir/weights.p \
            --out_path models/$outdir/"tw_whitening.run" \
            --dim_red $dim_red \
            --gpunum $gpunum \
            --freeze_glow True \
            --freeze_nice True \
            --rep_type $rep_type \
            --whitening True \
            --whitening_mu_file models/$outdir/train_mean_l11$wne.pt \
            --whitening_cov_file models/$outdir/train_cov_l11$wne.pt \
            --whitening_tokenwise True

        echo "evaluating"
        bin/trec_eval -m all_trec data/$data/qrels models/$outdir/"sw_whitening.run" > models/$outdir/"sw_whitening_eval.result"
        bin/trec_eval -m all_trec data/$data/qrels models/$outdir/"tw_whitening.run" > models/$outdir/"tw_whitening_eval.result"

        ## Sequence-wise
        echo "Sequence-wise whitening metrics"
        python metrics.py \
            --model $model \
            --datafiles data/$data/queries.tsv data/$data/documents.tsv \
            --run data/$data/$fold.test.run \
            --model_weights models/$outdir/weights.p \
            --out_path models/$outdir/"sw_whitening_metrics.run" \
            --dim_red $dim_red \
            --gpunum $gpunum \
            --freeze_glow True \
            --freeze_nice True \
            --rep_type $rep_type \
            --whitening True \
            --whitening_mu_file models/$outdir/sw_train_mean_l11$wne.pt \
            --whitening_cov_file models/$outdir/sw_train_cov_l11$wne.pt

        ## Sequence-wise
        echo "Token-wise whitening metrics"
        python metrics.py \
            --model $model \
            --datafiles data/$data/queries.tsv data/$data/documents.tsv \
            --run data/$data/$fold.test.run \
            --model_weights models/$outdir/weights.p \
            --out_path models/$outdir/"tw_whitening_metrics.run" \
            --dim_red $dim_red \
            --gpunum $gpunum \
            --freeze_glow True \
            --freeze_nice True \
            --rep_type $rep_type \
            --whitening True \
            --whitening_mu_file models/$outdir/train_mean_l11$wne.pt \
            --whitening_cov_file models/$outdir/train_cov_l11$wne.pt \
            --whitening_tokenwise True
    done
done