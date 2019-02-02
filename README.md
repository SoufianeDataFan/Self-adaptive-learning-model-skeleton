# Quick Start

 - Start from the function main to add directory and choose to turnon/off the automatic model tuning. 
 
```
if __name__ == "__main__":
    INPUT_DIR='/path/to/directory/' # directory of the inputs data 
    os.chdir(INPUT_DIR)
    with timer("Full model run"):
        main(self_tuning= True) 
 
```
 - I recommend you run the training file on the GPU for faster evaluation. Otherwise, make sure you change computing settings before you start. 
 
```
'tree_method':'gpu_hist', 
'predictor':'gpu_predictor'
```

 - The script is commented as much as possible so you can easily edit it and . Optimization space can be changed, and the Xgboost can be replaced by lightGBM and other algs. 

 - The script is clear and easy to edit. The main modeling pipeline is already set up, and you can easily adjust it for your own data. 
 


## Business perspective:

Interruption of connectivity can happen due to the failure of some nodes in a telecommunication network. Ensuring reliable network connectivity is curial for these companies since it can harm their brand if it happens repetitively. 
For this purpose, the engineers should conduct preventive maintenance actions on different nodes of the network that are spread on an expansive area. 

The challenge for the maintenance engineers is to prioritize the nodes with the highest fault severity level 1, 2, or 3 (imagine these nodes are in different locations). 

This is a multi-class problem with a tricky dataset (The meaning of the features is not easily clear from the beginning unless you are experienced in the field). 

This a great classifier with a single model with automatic tuning pipeline and some features engineering techniques. 





## Dataset: 

[Go to repository ](https://www.kaggle.com/c/telstra-recruiting-network/data/).

