_base_ = ['base.py']

exp='exp/covid-ct-esfpnet'

# Uncomment this line to evaluate results on the validation set
# data=dict(test=dict(im='data/val_im.nii.gz', msk='data/val_mask.nii.gz'))

model = dict(
        model_name='esfpnet',
        args = dict(
            model_type='B5',
            embedding_dim = 160
        )
    )