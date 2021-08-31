### EntropyMetrics
Usage: There are four mandatory arguments.

1. `dataset_name` is the name of the dataset that will be used for saving results.
2. `path/to/training/data` is the (relative) path to the folder containing training data. They can be separated into different folders. This code uses the torchvision ImageFolder and DataLoader.
3. Set `plot_histograms` to True if you want to save the histograms of the entropy normal curves that will be fitted to your data.
4. Set `classes` to True if you wish to calculate the cross entropy of a dataset with classes.

Template:
`python crossentropymetrics.py dataset_name path/to/training/data plot_histograms classes`

Example:
`python crossentropymetrics.py MockIN MockImageNet/train True True`
