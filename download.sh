# !/bin/bash

datadir=data
modeldir=Pretrained
stage=0

set -euxo pipefail

. parse_options.sh

if [ $stage -le 0 ]; then
    mkdir -p $datadir/.cache

    [ -f $datadir/test_im.nii.gz ] || \
        wget -O $datadir/test_im.nii.gz https://figshare.com/ndownloader/files/25953983
    [ -f $datadir/Test-Images-Clinical-Details.csv ] || \
        wget -O $datadir/Test-Images-Clinical-Details.csv https://figshare.com/ndownloader/files/25953974
    [ -f $datadir/.cache/rp_im.zip ] || \
        wget -O $datadir/.cache/rp_im.zip https://figshare.com/ndownloader/files/25954007
    [ -f $datadir/.cache/rp_lung_msk.zip ] || \
        wget -O $datadir/.cache/rp_lung_msk.zip https://figshare.com/ndownloader/files/25954010
    [ -f $datadir/.cache/rp_msk.zip ] || \
        wget -O $datadir/.cache/rp_msk.zip https://figshare.com/ndownloader/files/25954013

    for i in `ls $datadir/.cache/*.zip`; do
        unzip -n $i -d $datadir
    done

    [ -f $datadir/rp_im/tr_im.nii.gz ] || \
        wget -O $datadir/rp_im/tr_im.nii.gz https://figshare.com/ndownloader/files/25953977
    [ -f $datadir/rp_msk/tr_mask.nii.gz ] || \
        wget -Omit $datadir/rp_msk/tr_mask.nii.gz https://figshare.com/ndownloader/files/25953980

    mkdir $datadir/tmp && mv $datadir/rp_im/9.nii.gz $datadir/tmp && \
    python src/data/preprocess.py \
        --read_directory $datadir/tmp \
        --save_directory $datadir/val_im \
        --data_type image && rm -rf $datadir/tmp
    gzip -f $datadir/val_im.nii

    mkdir $datadir/tmp && mv $datadir/rp_msk/9.nii.gz $datadir/tmp && \
    python src/data/preprocess.py \
        --read_directory $datadir/tmp \
        --save_directory $datadir/val_mask \
        --data_type mask && rm -rf $datadir/tmp
    gzip -f $datadir/val_mask.nii


    python src/data/preprocess.py \
        --read_directory $datadir/rp_im \
        --save_directory $datadir/tr_im \
        --data_type image
    python src/data/preprocess.py \
        --read_directory $datadir/rp_msk \
        --save_directory $datadir/tr_mask \
        --data_type mask

    gzip -f $datadir/tr_im.nii
    gzip -f $datadir/tr_mask.nii
fi

if [ $stage -le 1 ]; then
    mkdir -p $modeldir
    gdown -O $modeldir --folder https://drive.google.com/drive/folders/1FLtIfDHDaowqyF_HhmORFMlRzCpB94hV
fi