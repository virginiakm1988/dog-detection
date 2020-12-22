# dog-detection

## get dataset
```
bash get_dataset.sh
```
## get sample model
```
bash get_model.sh
```

## train
```
bash train.sh <INPUT_DIR> <OUTPUT_DIR>
```

## test
```
bash test.sh <INPUT_IMG_DIR> <OUTPUT_IMG_DIR> <THRESHOLD> <MODEL_PATH>
```


## others
run these testing code have good performance
```
 bash test.sh ROI/0009/4.PNG ROI/0009/3.PNG 0.1 model/model_3.pkl
```
