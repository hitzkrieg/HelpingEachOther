# Helping Each Other
Code for the paper, **"Helping each Other: A Framework for Customer-to-Customer Suggestion Mining using a Semi-supervised Deep Neural Network"**

## About
This system can classify sentences as *suggestions from customers to other fellow customers*. The architecture extracts convolutional features and attention features apart from task-specific linguistic features. The model is trained in semi-supervised fashion using self-training. The loss function also uses weighted cross-entropy to handle class imbalance. 


## Instructions 
Please download the preprocessed files maintaining this same directory structure: [link](https://drive.google.com/open?id=1q3IUfAlBuUiIN5pyxgOqa4EqJSVZb500)

The directory has a file named `self_training.py` which runs on Python3. 
The file trains and evaluates for one data domain "hotel" or "electronics" which can be configured in line no 37. 
It reads already preprocessed files, trains and evaluates the model. 

The datasets:
1. Labeled data: Inside the Directory Original_data there are two files : `hotel.txt` and `electronics.txt`
2. Unlabeled data: `Original_data/Unlabeled_reviews/Text` has the files for both the domains.

The latest code for extracting linguistic features, preparing the embedding matrices , and sequences has unfortunately not been maintained well. You can have a look at the files `Original_data/prepare.py` and `Preprocessing/linguistic_features.py` for basic code, which you can adapt. 
