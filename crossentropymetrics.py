"""
Script to calculate various entropy metrics given a (training) dataset.
Test run on small dataset done on 30-AUG-2021.
Contact michal [dot] fishkin [at] mail [dot] utoronto [dot] ca for questions about the code.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import skimage.measure
from scipy.stats import norm
from scipy.special import erf
from skimage.color import rgb2gray
import cv2
import itertools
import os
import sys
import torchvision
import torch



# Command-line Arguments
dataset_name = sys.argv[1]
train_data_directory = sys.argv[2]
plot_histograms = sys.argv[3]
classes = sys.argv[4]

# Results folder for images and .csv files
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "{}_Results/".format(dataset_name))

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

print("Outputs will be stored at", results_dir)

# Data loading and processing
print("Loading data from", train_data_directory)
trnsfrm = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),])

imgfolder = torchvision.datasets.ImageFolder(train_data_directory, trnsfrm)
dataloader = torch.utils.data.DataLoader(imgfolder,
                                          batch_size=1,
                                          shuffle=True)


# Preparing partitions of the datasets for each label. This will allow us to calculate per-class entropy easily later on.
if classes == 'True':
    label_img_dict = dict()
    for i in range(len(imgfolder.classes)):
        label_img_dict[i] = []
    for step, (img, label) in enumerate(dataloader):
        np_img = np.uint8(np.asarray(img[0][0]*255))
        label_img_dict[label.item()].append(np_img)

# Preparing main imagelist - coverting to uint8 as is required to use cv2 filters.
imgs = []
for step, (img, label) in enumerate(dataloader):
    imgs.append(np.uint8(np.asarray(img[0][0]*255)))

## ENTROPY CALCULATIONS
def get_shannon_entropy(dataset):
    entropies = []
    for img in dataset:
        entropies.append(skimage.measure.shannon_entropy(img))
    return entropies

def get_laplace_entropy(dataset):
    entropies = []
    for img in dataset:
        x = cv2.Laplacian(img, cv2.CV_64F)
        entropies.append(skimage.measure.shannon_entropy(x))
    return entropies

## This class consists of a dataframe and functions that, given a dataset, populate the frame.
class EntropyData():

    def __init__(self):
        column_names = ["Dataset Name",
                        "Shannon Mean",
                        "Shannon Std. Dev.",
                        "Laplace Mean",
                        "Laplace Std. Dev."]
        self.EntropyDataFrame = pd.DataFrame(columns = column_names)
        return

    def run_entropy_analysis(self, dataset, name = "DATASET"):
        """
        This function will get the desired entropy type for a dataset and this apply a normal curve to its density histogram.
        """
        shannon_entropies = get_shannon_entropy(dataset)
        laplace_entropies = get_laplace_entropy(dataset)

        shannon_mu, shannon_sigma = self.plot_density_with_curve(shannon_entropies, dataset_name = name, entropy_name = "Shannon")
        laplace_mu, laplace_sigma = self.plot_density_with_curve(laplace_entropies, dataset_name = name, entropy_name = "Laplace")

        row = {"Dataset Name": name,
               "Shannon Mean": shannon_mu,
               "Shannon Std. Dev.": shannon_sigma,
               "Laplace Mean": laplace_mu,
               "Laplace Std. Dev.": laplace_sigma
               }

        self.EntropyDataFrame = self.EntropyDataFrame.append(row, ignore_index = True)

        return

    def plot_density_with_curve(self, entropies, dataset_name = "DATASET", entropy_name = "Shannon"):

        fig = plt.figure()
        mu, sigma = norm.fit(entropies)

        plt.title("Density of {} Entropy in {}".format(entropy_name, dataset_name))
        plt.hist(entropies, density = True, color = "lightblue", edgecolor="k", linewidth=1.2, bins = 25)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin * 0.85, xmax*1.15, 100)
        p = norm.pdf(x, mu, sigma)

        if plot_histograms == "True":
            plt.xlim(xmin * 0.85, xmax*1.15)
            plt.plot(x, p, 'k', linewidth=2)
            plt.legend(["$\mathcal{N}$"+"$(\mu={}, \sigma^2={})$".format(mu.round(2), sigma.round(2))])
            plt.savefig(results_dir + "{}_{}.png".format(entropy_name, dataset_name))

        return mu, sigma

    def get_cross_entropy(self):
        """
        Calculates cross-entropy and KL Divergence between classes.
        Puts results into class variables.
        Note: This function should only be called if the dataset has classes, i.e. classes == True.
        :return: None
        """
        def ce(a, b, c, d, x):
            # Avoid division by zero in log().
            eps = 0#1e-9
            # Formula for cross entropy is the CE integral evaluated at twice the distance of the dataset mean and zero.
            # This is to simplify the integration instead of having it from -inf to inf.
            # Both distributions are assumed normal with the parameters calculated by Shannon/Laplace entropy.
            return (np.sqrt(2*np.pi)*erf((x-a)/(np.sqrt(2)*b))\
            *(-a**2+2*a*c-b**2 +2*d**2*np.log(eps + np.exp(-(c-x)**2/(2*d**2))/(np.sqrt(2*np.pi)*d))-2*a*c+x**2)+\
            2*b*np.exp(-(a*-x)**2/(2*b**2))*\
            (a-2*c+x))/(4*np.sqrt(2*np.pi)*d**2)

        dataset_name =  self.EntropyDataFrame.iloc[0, 0]

        shannon_info = self.EntropyDataFrame.iloc[1:, [0, 1, 2]]
        laplace_info = self.EntropyDataFrame.iloc[1:, [0, 3, 4]]

        nrows = shannon_info.shape[0]
        combinations = list(itertools.combinations(range(nrows), 2))

        self.shannon_CE = np.zeros((nrows, nrows))
        self.shannon_KL = np.zeros((nrows, nrows))

        self.laplace_CE = np.zeros((nrows, nrows))
        self.laplace_KL = np.zeros((nrows, nrows))

        max_avg_shannon = self.EntropyDataFrame["Shannon Mean"].max()
        max_avg_laplace = self.EntropyDataFrame["Laplace Mean"].max()

        for combo in combinations:
            p, q = combo
            p_mean, p_std = shannon_info.iloc[p, [1, 2]]
            q_mean, q_std = shannon_info.iloc[q, [1, 2]]

            # KL formula from http://allisons.org/ll/MML/KL/Normal/

            self.shannon_CE[p, q] = ce(p_mean, p_std, q_mean, q_std, max_avg_shannon*2) - ce(p_mean, p_std, q_mean, q_std, 0)
            self.shannon_CE[q, p] = ce(q_mean, q_std, p_mean, p_std, max_avg_shannon*2) - ce(q_mean, q_std, p_mean, p_std, 0)
            self.shannon_KL[p, q] = ( (p_mean - q_mean)**2 + p_std - q_std ) / (2*q_std) + np.log(np.sqrt(q_std)/np.sqrt(p_std))
            self.shannon_KL[q, p] = ( (q_mean - p_mean)**2 + q_std - p_std ) / (2*p_std) + np.log(np.sqrt(p_std)/np.sqrt(q_std))


            p_mean, p_std = laplace_info.iloc[p, [1, 2]]
            q_mean, q_std = laplace_info.iloc[q, [1, 2]]

            self.laplace_CE[p, q] = ce(p_mean, p_std, q_mean, q_std, max_avg_laplace*2) - ce(p_mean, p_std, q_mean, q_std, 0)
            self.laplace_CE[q, p] = ce(q_mean, q_std, p_mean, p_std, max_avg_laplace*2) - ce(q_mean, q_std, p_mean, p_std, 0)
            self.laplace_KL[p, q] = ( (p_mean - q_mean)**2 + p_std - q_std ) / (2*q_std) + np.log(np.sqrt(q_std)/np.sqrt(p_std))
            self.laplace_KL[q, p] = ( (q_mean - p_mean)**2 + q_std - p_std ) / (2*p_std) + np.log(np.sqrt(p_std)/np.sqrt(q_std))

        fig = plt.figure()
        sns.heatmap(self.shannon_CE, annot = True)
        plt.title("{} Shannon Cross Entropy".format(dataset_name))
        plt.savefig(results_dir + "{}_Shannon_Cross_Entropy.png".format(dataset_name))

        fig = plt.figure()
        sns.heatmap(self.shannon_KL, annot = True)
        plt.title("{} Shannon KL Divergence".format(dataset_name))
        plt.savefig(results_dir + "{}_Shannon_KL_Divergence.png".format(dataset_name))



        fig = plt.figure()
        sns.heatmap(self.laplace_CE, annot = True)
        plt.title("{} Laplace Cross Entropy".format(dataset_name))
        plt.savefig(results_dir + "{}_Laplace_Cross_Entropy.png".format(dataset_name))

        fig = plt.figure()
        sns.heatmap(self.laplace_KL, annot = True)
        plt.title("{} Laplace KL Divergence".format(dataset_name))
        plt.savefig(results_dir + "{}_Laplace_KL_Divergence.png".format(dataset_name))

        return

#### Results
print("Making main dataframe...")
dat_df = EntropyData()

print("Calculating entropy metrics for the entire dataset.")
dat_df.run_entropy_analysis(imgs, name = dataset_name)

if classes == 'True':
    for k in label_img_dict.keys():
        print("Calculating entropy for class {}/{}.".format(k+1, len(label_img_dict.keys())))
        dat_df.run_entropy_analysis(label_img_dict[k], name = "{} Class {}".format(dataset_name, k))
    print("Calculating cross entropy...")
    dat_df.get_cross_entropy()

dat_df.EntropyDataFrame.to_csv(results_dir + "{}EntropyInfoTable.csv".format(dataset_name))

if classes == 'True':
    np.savetxt(results_dir + "{}_Shannon_CE_Table.csv".format(dataset_name), dat_df.shannon_CE, delimiter=",")
    np.savetxt(results_dir + "{}_Shannon_KL_Table.csv".format(dataset_name), dat_df.shannon_KL, delimiter=",")
    np.savetxt(results_dir + "{}_Laplace_CE_Table.csv".format(dataset_name), dat_df.laplace_CE, delimiter=",")
    np.savetxt(results_dir + "{}_Laplace_KL_Table.csv".format(dataset_name), dat_df.laplace_KL, delimiter=",")

print("Done!")