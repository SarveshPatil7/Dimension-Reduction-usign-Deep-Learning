# Dimension Reduction using Deep Learning for High-Dimensional Reliability Analysis

This project implements a deep learning–based framework for dimensionality reduction in high-dimensional reliability analysis (in MATLAB) as proposed in [https://doi.org/10.1016/j.ymssp.2019.106399]. The goal is to approximate expensive high-dimensional limit-state functions with a lower-dimensional surrogate model while retaining predictive accuracy for reliability estimation. This is a proof-of-concept / research prototype for applying deep-learning–based dimension reduction to reliability / surrogate modeling problems. Users can swap in their own high-dimensional function or simulation instead of the benchmark (Griewank function) used here, train the autoencoder + DFN + GP pipeline on their data and use the trained surrogate for fast predictions, reliability estimation, or design exploration. Potential use cases include, but are not limited to: estimation of CFD/FEA results when simulating a large number of models like in Topology optimization, reliability based design parameter selection for manufacturing to include the probablistic effects of tolerances, fatigue analysis when using statistical material models and many other high dimensional systems that can benefit from reduced order calculation.

Dependencies:
MATLAB with Deep Learning Toolbox

Method summary:
1. Data generation - This project uses the file 'data generator' to calculate a training dataset for our example, Griewank function. Skip if you have a dataset from actual tests. The code also creates some separate data points for model validation.
2. Auto encoder - An autoencoder learns the patterns in the training data and is used to obtain the dataset with reduced dimension in a latent space. It is coupled with an auto decoder and the goal is to have minimum difference between input and output layers.
3. Training the deep feed forward network - Creates a deep feedforward network (also known as a standard multilayer perceptron) that helps map the input data points to variables in latent space.
4. Gaussian process surrogate model - Creates a GP surrogate model for latent variables and related them to the output variables from the training dataset.
5. Model validation - Use the trained DFN and surrogate model to predict system response variables for the input points in validation dataset, and compare them with the actual system response. We see the results as Mean Square Error.
