### Before Started

\> We suppose you have downloaded datasets

- Configure the JOSN file first.

```shell
# in ./uacd/Siam_NestedUNet/metadata.json
{
    "patch_size": 256,
    "augmentation": true,
    "num_gpus": 1,
    ...
}
```

- Configure the file paths in the following files.

```shell
# in ./uacd/Siam_NestedUNet/train.py, line 20
save_path = 'save/path'  

# in ./uacd/Siam_NestedUNet/eval.py, line 18
path = 'path/of/model'   # the path of the model

# in ./uacd/Siam_NestedUNet/visulization.py, line 18
path = 'path/of/model'   # the path of the model
# line 56, the output path
file_path = './output_img/WHU-CD/base/' + base_name + '.png'
```

### Get Started

```shell
cd ./uacd/Siam-NestedUNet/

# training
python train.py

# evaluation
python eval.py

# visulization
python visualization.py
```
