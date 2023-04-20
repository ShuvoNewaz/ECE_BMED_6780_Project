# !/bin/bash

set -euo pipefail

N=10
batch_size=8
dropout=0.2
stage=0

. parse_options.sh

exp=exp/ensemble_dropout_$dropout

if [ $stage -le 0 ]; then
    echo "Training $N models"
    for idx in `seq 1 $N`; do
        model_exp=$exp/esfpnet_$idx
        mkdir -p model_exp
        python src/trainer.py --config config/esfpnet.py \
            --options exp=$model_exp batch_size=$batch_size model.args.dropout=$dropout

        model_exp=$exp/unet_$idx
        mkdir -p model_exp
        python src/trainer.py --config config/unet.py \
            --options exp=$model_exp batch_size=$batch_size model.args.dropout=$dropout
    done
fi

if [ $stage -le 1 ]; then
    echo "Generate Prediction"
    for idx in `seq 1 $N`; do
        model_exp=$exp/esfpnet_$idx
        python src/trainer.py \
            --config config/esfpnet.py \
            --predict \
            --options exp=$model_exp \
            data.test.im=data/tr_im.nii.gz

        model_exp=$exp/unet_$idx
        python src/trainer.py \
            --config config/unet.py \
            --predict \
            --options exp=$model_exp \
            data.test.im=data/tr_im.nii.gz
    done
fi

if [ $stage -le 2 ]; then
    echo "Train ensemble net"
    python src/trainer.py --config config/ensemble.py \
        --options exp=$exp \
        model.args.in_channels=`ls -d $exp/*/logits.npy | wc -l` \
        data.train.im=$exp \
        data.val.im=$exp \
        data.test.im=$exp
fi

if [ $stage -le 3 ]; then
    echo "Generate Prediction"
    for idx in `seq 1 $N`; do
        model_exp=$exp/esfpnet_$idx
        python src/trainer.py \
            --config config/esfpnet.py \
            --predict \
            --options exp=$model_exp \
            data.test.im=data/val_im.nii.gz \
            data.test.msk=data/val_mask.nii.gz 2>&1 | tee $model_exp/evaluate.log

        model_exp=$exp/unet_$idx
        python src/trainer.py \
            --config config/unet.py \
            --predict \
            --options exp=$model_exp \
            data.test.im=data/val_im.nii.gz \
            data.test.msk=data/val_mask.nii.gz 2>&1 | tee $model_exp/evaluate.log
    done
fi