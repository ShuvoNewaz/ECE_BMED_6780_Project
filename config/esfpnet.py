_base_ = ['base.py']

exp='exp/covid-ct-esfpnet'
batch_size=4

model = dict(
        model_name='esfpnet',
        args = dict(
            model_type='B5',
            embedding_dim = 160
        )
    )