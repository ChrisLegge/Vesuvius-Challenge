# Vesuvius-Challenge
This repository has all of our solutions and models for the Vesuvius Challenge. The Vesuvius Challenge is a machine learning competition focused on recovering hidden text from ancient Herculaneum scrolls that cannot be physically unrolled. Using 3D X-ray tomography data, we are detecting the presence of ink beneath the scroll's surface and reconstruct them as readable text.

The Core of this project is that we are solving a binary image segmentation problem, where the model predicts the probability of ink at each location on the scroll's surface. This is done through density variation between the scrolls surface and ink which are in the 3D volumetric Scan data. 

#Problem 
Each scroll is represented as a 3D volumetric dataset composed of hundreds of aligned X-ray slices taken at different depths below the surface. Ink is not directly visible; instead, it introduced faint structural patterns that must be inferred from the volumetric data. 

Input: A 3D x-ray volume (height*width*depth) and Corresponding ground-truth ink masks for training regions 
Output goal: Generate a 2D ink probability map, where each pixel value indicates the likelihood that ink exists at that surface location. 

#Approach to the Problem 
-Volumetric Data Processing 
  Load and normalise 3D X-ray volumes 
  Identify informative depth ranges where ink signals are strongest 
  Convert the 3D data into 2D representations using slice stacking or projections

-Creating the Model Architecture 
  Convolutional Neural Networks (CNNs), primarily U-Net-style architectures
  Designed for dense pixel-wise prediction 
  Outputs a 2D probability map with values in the range [0, 1] 

-Training the Model
  Supervised Learning using binary ink masks 
  Patch-based training to manage memory constraints 
  Loss Functions including Binary Cross-Entropy (BCE) and Dice Loss

-Inferencing and Evaluation
  Generate ink probability maps for the unseen scroll regions 
  Post-process predictions for improved segmentation quality 

#Output
The final model produces a probabilistic heatmap over the scroll surface. Regions with high predicted probability correspond to likely ink strokes, enabling downstream reconstruction of ancient text. 

