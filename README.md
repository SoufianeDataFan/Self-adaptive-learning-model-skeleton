
# Quick Start : 
 
 - Make sure you add the inputs directory and select the training mode (turn on/off the automatic model tuning).
```
if __name__ == "__main__":
    INPUT_DIR='/path/to/inputs/directory/'
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
 - Feel free to change the optimization space and replace the `Xgboost`  by other algs such us `lightGBM`. 
 - The script is clear and easy to edit. The main modeling pipeline is already set up, and you can easily adjust it for your own data. 
 
## Business perspective:

 - Interruption of connectivity can happen due to the failure of some nodes in a telecommunication network. Ensuring reliable network connectivity is curial. It can harm the brand of the company if it happens repetitively. 
For this purpose, maintenance engineers must conduct preventive actions periodically on different nodes of the network that are spread out throughout an expansive area. Without prioritization, maintenance would be a very time-consuming task. 

 - Practically speaking, the challenge here for maintenance managers is to define those nodes with the highest fault severity level 1, 2, or 3. 

 - This is a multi-class problem with a tricky dataset that requires a special understanding of the features due to their technical nature. Dealing with technical datasets would not be obvious from the beginning but it's not magic. The Data dictionary is provided with the dataset below. 

 - This is a great classifier with a single model and automatic tuning pipeline and some features engineering techniques. 



## Dataset: 

[Go to repository ](https://www.kaggle.com/c/telstra-recruiting-network/data/).

