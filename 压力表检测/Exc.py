from pyspark import SparkContext,SparkConf

conf = SparkConf().setAppName("pyspark_demo")
sc.stop()
sc = SparkContext(conf = conf)

# 构造一个RDD
rdd = sc.parallelize([1, 2, 3, 3])