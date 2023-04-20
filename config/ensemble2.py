_base_ = ['base.py']

exp='exp/ensemble'

# Uncomment this line to evaluate results on the validation set
# data=dict(test=dict(im='data/val_im.nii.gz', msk='data/val_mask.nii.gz'))

unet = dict(
        model_name='unet',
        args = dict(
            n_channels = 1,
            n_classes = 1
        ),
        pretrained = "exp/covid-ct-unet/best.pt"
)

esfpnet = dict(
    model_name='esfpnet',
    args = dict(
        model_type='B5',
        embedding_dim = 160
    ),
    pretrained = "exp/covid-ct-esfpnet/best.pt"
)

model = dict(
    model_name = 'ensemble',
    args = dict(
        models = [unet, esfpnet],
        method = "average"
    )
)