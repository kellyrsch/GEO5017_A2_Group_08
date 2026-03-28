# GEO5017_A2_Group_08

Assignment 2 for the course GEO5017: Machine Learning for the Built Environment at TU Delft

## To run this code:

1. make sure to add the data under "/pointclouds-500/" (data is expected to be 500 .xyz files named by index in buckets of 100 by label - in the order "building", "car", "fence", "pole", "tree")
2. install the dependencies using `pip install -r requirements.txt`
3. run "/code/main.py"

### Notes:

The code currently takes quite a while to run, even on capable hardware. This is due to the `create_learning_curves` function, which estimates a total of 200 SVM and 200 RF models. You can reduce this by setting num_random_samples to 1, which will lead to 20 + 20 models being estimated, but a potentially less accurate learning curve.