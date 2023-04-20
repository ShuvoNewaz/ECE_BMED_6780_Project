_base_ = ['base.py']

exp='exp/ensemble'
seg_type='ensemble'
num_epochs=10

model=dict(model_name='single_Conv', args=
    dict(
        in_channels=1,
        out_channels=1,
        kernel_size=(3,3),
        dilation=0
    ))

data=dict(
    train=dict(im=exp, msk='data/tr_mask.nii.gz'),
    val=dict(im=exp, msk='data/tr_mask.nii.gz'),
    test=dict(im=exp, msk=''),
)