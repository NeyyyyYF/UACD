### Before Started

\> We suppose you have downloaded datasets.

- Configure the dataset path in the following file.

```shell
# in ./uacd/ChangeFormer/data_config.py, We can configure multiple dataset paths.
if data_name == 'LEVIR':
	self.label_transform = "norm"
	self.root_dir = '/path/to/dataset/root'
# for example
elif data_name == 'DATASET NAME'
	self.label_transform = "norm"
	self.root_dir = '/path/to/dataset'
```

- Configure the script files.

\> You can find the running script of ChangeFormer at the following path：`./uacd/ChangeFormer/scripts/`

We have provided running scripts for training and inference on various datasets. You can adjust the script codes according to your own needs.

### Get started

```shell
cd ./uacd/ChangeFormer

sh scripts/run_changeformer_***.sh
```
