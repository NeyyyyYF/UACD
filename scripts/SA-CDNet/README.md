### Before Started

\> We suppose you have downloaded datasets.

- Configure the dataset path in the following file.

```shell
# for dataset, in ./network/SA-CDNet/datasets/data.py, line 16
root = "data/root"
```

- For training, configure parameters in the following file.

```shell
# in ./uacd/SA-CDNet/scripts/run_train.sh
    --working_path "$WORKING_PATH" \
    --DATA_NAME "$DATA_NAME" \
    --NET_NAME "$NET_NAME" \
    ...
```

- For evaluation, configure parameters in the following file.

```shell
# in ./uacd/SA-CDNet/scripts/run_eval.py
    --working_path "$WORKING_PATH" \
    --DATA_NAME "$DATA_NAME" \
    --NET_NAME "$NET_NAME" \
    ...
```

### Get Started

\> we suppose you are in `./uacd`

```
cd ./uacd/SA-CDNet/

# training
sh scripts/run_train.sh

# evaluation
sh scripts/run_eval.sh
```
