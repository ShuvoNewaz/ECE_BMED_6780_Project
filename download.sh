# #!/bin/bash
# # http://medicalsegmentation.com/covid19/
# # Downloads and organizes the dataset and the required pretrained weights.
datadir=dataset
pretrained_dir=pretrained
rm -r "${datadir}"
rm -r "${pretrained_dir}"

train_data_dir="${datadir}/train"
validation_data_dir="${datadir}/validation"
test_data_dir="${datadir}/test"
saved_esfpnet_dir="${pretrained_dir}/esfpnet"
saved_resnet_dir="${pretrained_dir}/resnet"
for dir in "${datadir}" "${train_data_dir}"\
            "${validation_data_dir}" "${test_data_dir}"\
            "${pretrained_dir}" "${saved_esfpnet_dir}"\
            "${saved_resnet_dir}";
do
    mkdir -p "${dir}"
done

# Download data

## Validation Data
wget https://figshare.com/ndownloader/articles/13521488/versions/2
unzip 2 -d ${datadir}
rm "${datadir}/Test-Images-Clinical-Details.csv"
rm 2
gdown --fuzzy "https://drive.google.com/file/d/1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-/view?usp=sharing"
mv "tr_lungmasks_updated.nii.gz" "${validation_data_dir}/lung_mask.nii.gz"

### Rename and move validation files
declare -A validation_rename_dict
validation_rename_dict["tr_im.nii.gz"]="im.nii.gz"
validation_rename_dict["tr_mask.nii.gz"]="mask.nii.gz"
for key in "${!validation_rename_dict[@]}";
do
    mv "${datadir}/${key}" "${validation_data_dir}/${validation_rename_dict[$key]}"
done

### Rename and move test files
mv "${datadir}/val_im.nii.gz" "${test_data_dir}/im.nii.gz"

## Training Data
wget https://figshare.com/ndownloader/articles/13521509/versions/2
unzip 2 -d ${datadir}
rm 2

# Process RadioPedia
declare -a items_to_unzip=("rp_msk.zip"
                            "rp_im.zip"
                            "rp_lung_msk.zip"
                            )
for item in "${items_to_unzip[@]}";
do
    unzip "${datadir}/${item}" -d "${datadir}"
    rm "${datadir}/${item}"
done

## Merge the disjoint RadioPedia iamges and masks
declare -A item_types
item_types["rp_im"]="image"
item_types["rp_msk"]="mask"
item_types["rp_lung_msk"]="mask"

declare -A destination_dirs
destination_dirs["rp_im"]="${train_data_dir}/im"
destination_dirs["rp_msk"]="${train_data_dir}/mask"
destination_dirs["rp_lung_msk"]="${train_data_dir}/lung_mask"

for item in "${!item_types[@]}";
do
    src_dir="${datadir}/${item}"
    dest_dir="${destination_dirs[$item]}"
    python src/data/merge_rp.py -rd "${src_dir}" -sd "${dest_dir}.nii.gz" -dt "${item_types[$item]}"
    rm -r "${src_dir}"
done

## Remove images that have no mask and extract lung regions
python src/data/ignore_no_seg.py -dir "${train_data_dir}"
python src/data/ignore_no_seg.py -dir "${validation_data_dir}"

# Download pretrained weights

gdown -O "${saved_esfpnet_dir}" --folder https://drive.google.com/drive/folders/1FLtIfDHDaowqyF_HhmORFMlRzCpB94hV

declare -A resnet_links
resnet_links["resnet18"]="https://download.pytorch.org/models/resnet18-5c106cde.pth"
resnet_links["resnet50"]="https://download.pytorch.org/models/resnet50-19c8e357.pth"
resnet_links["resnet101"]="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
resnet_links["resnet152"]="https://download.pytorch.org/models/resnet152-b121ed2d.pth"

declare -A resnet_local_names
resnet_local_names["resnet18"]="resnet18.pth"
resnet_local_names["resnet50"]="resnet50_v2.pth"
resnet_local_names["resnet101"]="resnet101_v2.pth"
resnet_local_names["resnet152"]="resnet152_v2.pth"

for resnet in "${!resnet_links[@]}";
do
    wget -O "${saved_resnet_dir}/${resnet_local_names[$resnet]}"\
    "${resnet_links[${resnet}]}"
done