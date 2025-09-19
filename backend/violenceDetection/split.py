import splitfolders
""" import kagglehub
import kagglehub

path = kagglehub.dataset_download("mohamedmustafa/real-life-violence-situations-dataset")

print("Path to dataset files:", path) """
splitfolders.ratio("/home/umut/server/backend/violenceDetection/rawData", output="/home/umut/server/backend/violenceDetection/data", seed=1337, ratio=(0.7, 0.2, 0.1))
