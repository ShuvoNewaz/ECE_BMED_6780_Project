_base_ = ['base.py']

exp='exp/covid-ct-unet'

# Uncomment this line to evaluate results on the validation set
# data=dict(test=dict(im='data/val_im.nii.gz', msk='data/val_mask.nii.gz'))

model = dict(
        model_name='unet',
        args = dict(
            n_channels = 1,
            n_classes = 1,
            dropout=0
        )
    )