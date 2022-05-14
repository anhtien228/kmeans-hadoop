# K-Means Clustering using Hadoop and MapReduce

A mini project for the K-Means Clustering implementation using Hadoop and MapReduce wiwth the help of MRJob library.
Part of HCMUT Data Mining course.

1. Programming Tool: Visual Studio Code
2. Language: Python
3. Frameworks/Libraries: Hadoop, MRJob (0.7.4)
4. Goal:
  * Implement the K-Means Clustering and deploy in local/Hadoop Cluster/Dataproc
  * Compare the performance of MapReduce (local, cluster) with scikit-learn's K-Means

Note:
* The details for implementation are all inside the path: `/src`
* The scatter plots will be stored in `/images` after running `dataGenerator.py`
* Navigate to KMeans.py and change runner to `--runner=local` to run locally or `-r dataproc` to deploy on Dataproc (dataproc version <= 1.1.0)
* Deploy MRJob on Amazon EMR is possible as well by simply using `r emr` and EMR configuration file.
