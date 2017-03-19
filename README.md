#**Vehicle Detection and Tracking** 

This project is part of Udacity's [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). The goal of the project is to use computer vision and machine learning to detect vehicles in a video from a front-facing camera on a car. 

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detection frame by frame to reject outliers and follow detected vehicles.
* Draw a bounding box for vehicles detected.

The steps mentioned above are covered in more details in the sections below.

###**Data preparation**

The datasets provided by Udacity are comprised of images taken from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video. 

Random samples from the datasets are shown below:



The datasets are composed of image data that was extracted from video. As subsequent images from a video are highly correlated, using a randomized train-test split will not be a good decision because images in the training set may be nearly identical to images in the test set.

By exploring the image folders in the dataset, I noticed that the image files are numbered and if sorted naturally we can have multiple sequences on top of each other where each sequence represents images for the same car.

The function `split_dataset()` in cell `Helper Functions` is used to perform a valid split into training and validation datasets. The function reads a list of files and naturally sorts the filenames first before splitting the files into two groups specified by the `split_pct` parameter which indicates the training-validation split percentage. In such case, we try to reduce the probability of having identical images in the training and test data set.

The dataset was splitted into 75% for training and 25% for testing

To load the dataset, I first use glob on the datasets folder:

    vehicle_loc='./labeled_data/vehicles/'
    nonvehicle_loc='./labeled_data/non-vehicles/' 
    vehicle_samples = glob.glob(vehicle_loc + '**/*.png', recursive=True)
    nonvehicle_samples = glob.glob(nonvehicle_loc + '**/*.png', recursive=True)

Then I call `split_dataset()` on both `vehicle_samples` and `nonvehicle_samples` lists

    for index, dataset in enumerate([nonvehicle_samples, vehicle_samples]):
	    X_train_dataset, y_train_dataset, X_val_dataset, y_val_dataset = split_dataset(dataset, index, split_pct)
	    X_train_samples.extend(X_train_dataset)
	    y_train_samples.extend(y_train_dataset)
	    X_val_samples.extend(X_val_dataset)
	    y_val_samples.extend(y_val_dataset)

The code for data exploration and loading the dataset is contained in the third and fifth code cells of the IPython notebook

###**Histogram of Oriented Gradients (HOG)**  

#### Extracting HOG features from the training images

I used the function `skimage.hog()` to extract the HOG features from images.  I experimented with different values for the parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) on some random selected images from each of the vehicle and non-vehicle classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The code for this step is contained in the fourth code cell of the IPython notebook using the helper function `single_img_features()`

Below is an example of how the HOG features are different between a vehicle and a non-vehicle image when using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` ,  `cells_per_block=(2, 2)` and `hog_channel=0`:


#### Training the classifier

 I settled on using HOG features, spatial features and histogram features for training the classifier. I trained a Linear SVM using the `sklearn.LinearSVC()` function. The features were normalized using `sklearn.StandardScaler()`. Before training, the data was shuffled using `sklearn.shuffle()`.

I trained and evaluated the classifier using a wide range of parameters. At first, I experimented with different color spaces while fixing all the other parameters. I found that `RGB` was the one to yield the least accuracy and `YCrCb` to yield the best accuracy. I experimented with different values for `orientations`and found the accuracy to increase by increasing the number of orientation bins. Selecting all the channels for extracting the HOG features resulted in a better accuracy than choosing single channels.

For the spatial and histogram features, choosing 32 as the size of the reduced image and histogram bins respectively was better in accuracy than choosing 16.

After experimenting with different parameters, the winning combination in terms of accuracy was:

`Color_space = 'YCrCb' 
Number of orientation bins (HOG): 10 
HOG pixels per cell: 8
HOG cells per block: 2 
Channels to use: ALL
Spatial binning dimensions: 32 x 32 
Number of histogram bins: 32`

The above parameters yielded an accuracy of `0.9883` on the test dataset.

The classifier was trained in code cells 6 and 7 of the IPython notebook.

###**Sliding Window Search**

I used a sliding window approach to search each frame. To optimize the feature extraction, hog features are extracted only once and then sub-sampled to get all of the overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells. Instead of resizing the window for different window sizes, the whole frame image is scaled according the window scaling factor and and the search window is always of size 64 x 64.

I used 4 scaling factors of `1.25`, `1.5`, `2`and `3`  which corresponds to window sizes of `80 x 80` ,  `96 x 96`, `128 x 128` and `194 x 194` respectively which provided good results.

For each window, spatial, histogram features and hog features are computed and fed to the classifier for prediction. I also computed the confidence score for each prediction using the function `decision_function()`for the classifier. This allows to force a threshold on the predicted values as a mean to filter outliers.

The code for the implementation of the sliding window approach is in code cell 8

Below is an example output of using the sliding window approach on a series of test images:


The positions of positive detections in each frame are recorded. From the positive detections, a heatmap is created and then thresholded over several frames  to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Each blob corresponded to a vehicle. Finally bounding boxes are drawn to cover the area of each blob detected. Heatmaps are shown in the example above on the right-hand side of the images.

Below is an example of an image after thresholding the heatmap:



###**Video Implementation**

The function `processframe( )` in code cell 10 is used to process each frame in the video.

A running sum of the last several heatmaps was created and then thresholded to remove false positives.

The bounding boxes are validated to remove outliers by checking the width and height of each box and comparing to a threshold. This is implemented in function `draw_labeled_bboxes_thresholded( )`in Helper functions (code cell 2)

Here's a link to my video result


###**Reflection**