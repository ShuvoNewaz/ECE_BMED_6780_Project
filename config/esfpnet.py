_base_ = ['base.py']

exp='exp/covid-ct-esfpnet'

model = dict(
        model_name='esfpnet',
        args = dict(
            model_type='B5',
            embedding_dim = 160
        )
    )