#!/home/hduser/anaconda3/bin/python
# pylint: disable=invalid-name, anomalous-backslash-in-string
### ⚙️ KMeans Clustering ⚙️ ###

import argparse
import os
import random
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import subprocess
import time
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/hduser/mrjob-key.json"
#sys.tracebacklimit = 0

class MapReduceKmeansRunner(object):
    """
    Implement helper methods to run the K-Means algorithm

    """
    @staticmethod
    def getData(input_file):
        """
        Collect the data points from generated file.
            
        ⚙️ inputs: 
            input_file (string): File storing the data points
        -----
        ⚙️ outputs:
            points (list): Array of input data points

        """
        input_points = pd.read_csv(input_file, header=None, names=["x", "y"], sep=" ")

        if len(input_points.index) < 1:
            raise Exception("The input data is empty!")

        tuple_points = [tuple(row) for row in input_points.values]
        points = np.array([point for point in tuple_points])
        print(points)
        return points


    def initCentroids(self, inputFile, nclusters):

        """
        Initialzie the centroids among the data points
            
        ⚙️ inputs: 
            inputFile (string): File storing the data points
            n_cluster (int): Number of clusters
        -----
        ⚙️ outputs:
            init_centroids (list): List of initial centroids

        """
        points = self.getData(inputFile)
        initial_centroids = [list(random.choice(points))]
        dist = []

        if nclusters < 2:
            raise Exception ("[Error] Number of cluster should be >= 2")
        for i in range(2, nclusters + 1):
            dist.append(
                [np.linalg.norm(np.array(point) - initial_centroids[i - 2])**2
                 for point in points])
            min_dist = dist[0]
            if len(dist) > 1:
                for dist_t in dist:
                    min_dist = np.minimum(min_dist, dist_t)

            sumValues = sum(min_dist)
            probabilities = [float(value) / sumValues for value in min_dist]
            cumulative = np.cumsum(probabilities)

            random_index = random.random()
            index = np.where(cumulative >= random_index)[0][0]
            initial_centroids.append(list(points[index]))
        print("Init centroids: ", initial_centroids)
        return initial_centroids


    @staticmethod
    def getCentroids(centroidsfile):
        """
        Collect the data of centroids from the generated file
            
        ⚙️ inputs: 
            centroidsfile (string): File storing the centroids
        -----
        ⚙️ outputs:
            centroids (list): List of centroids coordinates

        """
        with open(centroidsfile, "r") as inputFile:
            output_data = inputFile.readlines()

        centroids = []

        for point in output_data:
            p = re.search("\[(.*?)\]", point).group() # Retrieve all subgroups from list of centroids
            p = p.replace("[", "").replace("]", "") # Remove brackets wrapping the centroids data
            p.strip() # Strip all whitespacse
            ax_x, ax_y = p.split(",")
            ax_x = float(ax_x)
            ax_y = float(ax_y)
            point_list = [ax_x , ax_y]
            centroids.append(point_list)
        
        return centroids


    def getLabels(self, input_file, centroids_file):
        """
        Get the labels of the input data points

        ⚙️ inputs: 
            input_file (string): File storing the data points
            centroids_file (string): File storing the centroids
        -----
        ⚙️ outputs:
            labels (list): List of labels

        """
        data_points = self.getData(input_file)
        centroids = self.getCentroids(centroids_file)
        labels = []

        for point in data_points:
            distances = [np.linalg.norm(point - np.array(centroid))
                         for centroid in centroids]
            # Get the minimum distance to determine the cluster
            cluster = np.argmin(distances)
            labels.append(int(cluster))

        return labels

    @staticmethod
    def exportCentroids(centroids):
        """
        Write data of centroids into a file
        ⚙️ inputs: 
            centroids : List of centroids coordinates

        """
        f = open(CENTROIDS_FILE, "w+")
        for item in centroids:
            f.write("%s\n" % str(item))
        f.close()

    @staticmethod
    def plotClusters(points, centroids, labels):
        """
        Plot the scatter graph of clusters and save it as an image

        ⚙️ inputs:
            centroids: A list centroids of each clusters
            points: A list of data poitns belonging to the cluster
            labels (list): List of labels

        """
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(1, 1)
        temp_fig = sns.scatterplot(x=points[:, 0], y=points[:, 1], color='gray', palette="deep", ax = ax)
        palette = itertools.cycle(sns.color_palette("muted"))
        
        for i, _ in enumerate(centroids):
            label = "Centroid " + str(i)    
            plt.scatter(x=centroids[i][0], y=centroids[i][1], s=50,
                        color=next(palette), label=label)
        
        plt.legend(loc="best", fancybox=True) 

        directory = "../images"
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # std = 0
        # for file in os.listdir("../images/"):
        #     if file.startswith("data_{}x{}_{}k".format(points.shape[0], points.shape[1], k)):
        #         std = float(re.search('k_(.*)std', file).group(1))

        fig.savefig("../images/clusters_{}x{}_{}k.png".format(points.shape[0], points.shape[1], k))                                                                                                                                                                                                                                                                                                                                                                                                      


CENTROIDS_FILE = "centroids.txt"
OUTPUT_FILE = "output.txt"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hadoop MapReduce with K-Means Clustering")
    parser.add_argument("inputFile", type=str,
                        help="File of input data points for clustering process.")

    parser.add_argument("centroids", type=int,
                        help="Number of centroids.")

    args = parser.parse_args()

    data = args.inputFile
    k = args.centroids

    # Create K-Means instance. initialize the centroids and export to file
    instanceKMeans = MapReduceKmeansRunner()
    # Start time for MRKmeans
    start_time = time.time()

    centroids = instanceKMeans.initCentroids(data, int(k))
    instanceKMeans.exportCentroids(centroids)
    print("Initialize centroids successfully")

    output_file = open(OUTPUT_FILE, "w+")
    output_file.close()
    
    iteration = 1
    while True:
        print("K-Means iteration ", iteration)

        command = "python mrKMeans.py" \
                  + " --k=" \
                  + str(k) + " --centroids=" \
                  + CENTROIDS_FILE\
                  + " < " + data\
                  + " > " + OUTPUT_FILE \
                  + " -r dataproc"\
                  + " --instance-type n1-standard-1"\
                  + " --num-core-instances 2"\
                  #+ " --python-bin /home/hduser/anaconda3/bin/python"

        # Start the mrKMeans.py to:
        #   > Assign data points to each cluster (map)
        #   > Calculate partial sum of data points (combine)
        #   > Calculate the new centroids (reduce)
        #os.popen(command)
        theproc = subprocess.Popen(command, shell=True)
        theproc.communicate()
        print("Calculate new centroids succesfully")
        # Calculate new centroids
        new_centroids = instanceKMeans.getCentroids(OUTPUT_FILE)
        #print(new_centroids)
        # If new centroids is different from the former ones
        # replace it with the new coordinates
        centroid_compare = [np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(centroids, new_centroids)]
        #print(centroid_compare)
        max_d = max(centroid_compare)

        #if sorted(centroids) != sorted(new_centroids):
        if max_d > 10.0:
            centroids = new_centroids
            # Export into file of new centroids for next iteration
            instanceKMeans.exportCentroids(centroids)
        else:
            # No more changes can be made for the centroids position
            break
        
        iteration += 1
    
    exec_time = time.time() - start_time
    os.remove(OUTPUT_FILE)
    # Get the labels after the algorithm ends
    labels = instanceKMeans.getLabels(data, CENTROIDS_FILE)
    labels_file = open("labels.txt", "w+")
    for label in labels:
        labels_file.write("%s\n" % str(label))
    labels_file.close()

    points = instanceKMeans.getData(data)
    instanceKMeans.plotClusters(points, centroids, labels)
    
    time_log = "--- {} seconds ---".format(exec_time)
    print("--- %s seconds ---" % (exec_time))
    run_log = "| Test Params | Cluster = {} | N = {} | Dim = {} | Iteration {} |"\
                .format(k, points.shape[0], points.shape[1], iteration)
    print(run_log)

    with open("run_logs.txt", "a") as log_file:
        log_file.write("%s\n%s\n" % (run_log, time_log))
