_base_ = ['base.py']

exp='exp/ensemble'
seg_type='ensemble'
num_epochs=5

model=dict(model_name='single_Conv', args=
    dict(
        in_channels=1,
        out_channels=1,
        kernel_size=(64,64),
        padding='same',
        bias=False
    ))

data=dict(
    train=dict(im=exp, msk='data/tr_mask.nii.gz', logit_file='train.npy'),
    val=dict(im=exp, msk='data/val_mask.nii.gz', logit_file='val.npy'),
    test=dict(im=exp, msk='data/val_mask.nii.gz', logit_file='val.npy'),
)