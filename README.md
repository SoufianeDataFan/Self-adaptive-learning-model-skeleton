
# Quick Start : 
 
 - In the function 'main', add the inputs directory and select the training mode (turn on/off the automatic model tuning).
 
```
if __name__ == "__main__":
    INPUT_DIR='/path/to/inputs/directory/' # directory of the inputs data 
    os.chdir(INPUT_DIR)
    with timer("Full model run"):
        main(self_tuning= True) 
 
```
 - I recommend you use a GPU for faster evaluation. Otherwise, make sure you change computing settings before you start. 
 
```
'tree_method':'gpu_hist', 
'predictor':'gpu_predictor'
```

 - The script is commented as much as possible so you can easily edit it.
 - Feel free to change the optimization space and replace the `Xgboost`  by other algs such us `lightGBM`. (look for another post of mine about lightgbm soon ;) )
 - The script is clear and easy to edit. The main modeling pipeline is already set up, and you can easily adjust it for your own data. 
 


## Dataset: 

[Go to repository ](https://www.kaggle.com/c/telstra-recruiting-network/data/).

