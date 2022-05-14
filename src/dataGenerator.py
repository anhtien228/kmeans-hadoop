# pylint: disable=invalid-name, anomalous-backslash-in-string, abstract-method
### ⚙️ Generate synthesis data for MRKmeans Clustering ⚙️ ###

import argparse
import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
sns.set_theme(style="darkgrid")
class DataGenerator(object):

    @staticmethod
    def generateData(file, n_points=100, dim=2, centers=3, std=1.5):
        """
        Function used to generate blobs data points with
        specified parameters n_samples, n_features and centers 

        ⚙️ inputs:
            n_points (int): Number of data points
            dim (int): Number of features (dimension)
            center (int or array): Number of initial centroids (or coordinates array)
            file (string): Name of files storing the data points
        
        """
        points, labels = make_blobs(n_samples=n_points,
                                    n_features=dim,
                                    cluster_std=std,
                                    centers=centers)

        scatter = sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=labels, palette="muted")
        fig = scatter.get_figure()

        df = pd.DataFrame(points)
        #df.to_csv(file, header=False, index=False, sep=" ")
        df.to_csv("input/backup/input_data_{}x{}_{}k_{}std.txt".format(n_points, dim, centers, std), header=False, index=False, sep=" ")

        directory = "../images"
        if not os.path.isdir(directory):
            os.makedirs(directory)

        fig.savefig("../images/data_{}x{}_{}k.png".format(n_points, dim, centers))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str,
                        help="File to save the generated data points.")

    parser.add_argument("n_points", type=int,
                        help="Number of data points to generate.")

    parser.add_argument("dim", type=int,
                        help="Number of features (dimensions) of data.")

    parser.add_argument("centers", type=int,
                        help="Number of centroids to initialize.")

    parser.add_argument("std", type=float,
                        help="Cluster standard deviation value.")

    args = parser.parse_args()

    instanceDataGenerator = DataGenerator()
    instanceDataGenerator.generateData(args.file, args.n_points, args.dim, args.centers, args.std)