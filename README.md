# fruit-vegetable-classification
Training a simple ConvNet to classify fruits and vegetables with PyTorch

[Dataset](https://www.kaggle.com/moltean/fruits) is available in kaggle. Download the dataset, unzip it to where "CNN.py" or "CNN.ipynb" is

In this dataset, there are 131 classes.

CNN architecture
```bash
INPUT -> [CONV -> RELU -> POOL]*3 -> FC
```


Install the required libraries
```bash
$ pip install -r requirements.txt
```

Then run the CNN.py file or run the CNN.ipynb file cell by cell.

```bash
$ python CNN.py
```

## Results
Some predictions 
| ![results.png](results.png) | 
|:--:| 
| classification results |

Train Loss History
| ![loss.png](loss.png) | 
|:--:| 
| epoch - Loss |

Testing the network on the test dataset
```bash
overall accuracy is 0.92
overall precision is 0.93
overall recall is 0.91
```
