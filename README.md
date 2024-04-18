# SPLITZ
This is the official codebase for the paper: SPLITZ: CERTIFIABLE ROBUSTNESS VIA SPLIT LIPSCHITZ RANDOMIZED SMOOTHING.
We introduce SPLITZ, a novel and practical approach that combines the synergistic advantages of both Lipschitz constrained training and randomized smoothing into a unified framework. The core concept involves splitting a classifier into two segments: constraining the Lipschitz constant of the first segment, and applying randomization to smooth the second segment.

# Scripts
 * Run "train.py" to train a SPLITZ classifier with Gaussian data augmentation inside the model.
 * Run "certify.py" to certify the robustness of the classifier
 * Run "analyze.py" to generate the plots and tables.
