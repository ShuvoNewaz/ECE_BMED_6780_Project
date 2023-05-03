exp='covid-ct'
batch_size=8
seg_type='infection'
load_from_disk=False
num_epochs=25
lr=1e-4

model=dict(model_name='pspnet', args=dict())

data=dict(
    train=dict(im='data/tr_im.nii.gz', msk='data/tr_mask.nii.gz'),
    val=dict(im='data/val_im.nii.gz', msk='data/val_mask.nii.gz'),
    test=dict(im='data/test_im.nii.gz', msk=''),
)

