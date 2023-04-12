_base_ = ['base.py']

exp='exp/covid-ct-unet'
batch_size=4

model = dict(
        model_name='unet',
        args = dict(
            n_channels = 1,
            n_classes = 1
        )
    )