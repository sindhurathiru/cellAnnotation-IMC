# Automated Cell Type Annotation
<i>Automated cell phenotyping for imaging mass cytometry data using a deep convolutional autoencoder with joint classifier.</i>

Identifying cell types in tissue sample - referred to as cell phenotyping - is a crucial step in most analyses of cell-level data. It's typically a manual process which can result in user bias and difficulty scaling. The objective of this project is to automate the cell annotation process using deep learning.

## Files
<b>cellAnnotation.py</b> contains the code to load the raw data, pre-process, and run through the network. Classifier identifies and phenotypes cells into four basic types: Immune, Stromal, Tumour, and Other. Optimum values of structural parameters were determined based on classification accuracy. The convolutional autoencoder was pre-trained alone to initialize network weights, then trained jointly for reconstruction loss and classification accuracy. The model was evaluated using independent testing set and the process was repeated for 20 randomly generated sets, using the average of performance metrics as measure of performance. A conventional CNN was also implemented as a baseline to compare to.
