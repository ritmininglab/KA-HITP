-------------------------------------------------------

Source Code
-------------------------------------------------------

This package contains source code for submission: Uncertainty-Aware Knowledge Acquisition for Interactive Image Captioning


Specification of dependencies：

The code is written in Python 3.8. Based on the specified software version, OpenCV 3.0, scipy 1.5, Keras 2.0, Tensorflow 2.5, scikit-image 0.17, scikit-learn 0.23 libraries are required for the environment setting. All files need to be on the same directory in order for the algorithms to work.


A demo video is provided to illustrate the usage of the proposed model.
For user interaction, a pop-up window will be generated to allow the user to type in the corresponding words (separated by commas) to guide the caption generation process. The keywords are reordered and the prediction of candidate captions based on the current image are printed on the command window.
Then the model is updated based on the new data, and proceeds to make automated prediction of keywords and candidate captions on another image. Results are printed on the command window.


To run the algorithm, change the path to the current directory in the command window, and run the [main.py] file:

main.py
The main method that implements the proposed algorithm to perform caption prediction and refining captions based on user interactions. Results will be generated on console.

The main methods call the following functions:

1. mCap.m
Includes methods that define the architecture of the network, customized block and layers of the network.

2. config.m
Includes configurations.

3. utilIO2.m
Includes utility methods that reading, writing and processing images and caption data.

4. utilIOKey.m
Includes utility methods for reading, writing and processing key concept data.

5. utilIOKey.m
Includes utility methods for reading, writing and processing ranking score data.

6. utilMisc.m
Includes utility methods for importing and exporting model parameters.

7. utilVisualize.m
Includes utility methods for visualization results.

8. insertionTrans.m
Includes methods for training insertion module.

9. keywordTrans.m
Includes methods for training keyword prediction module.