## Plant Disease Recognition using Deep Learning and PyTorch

Using deep learning and machine learning has helped to solve many agricultural issues. Deep learning based computer vision systems have been especially helpful. Such deep learning systems/models can easily recognize different diseases in plants when we train them on the right dataset. For example, we can train a deep learning model to recognize different types of diseases in rice leaves.

Most of the time, obtaining large and well-defined datasets to solve such problems becomes an issue. Because for deep learning models to recognize diseases in plants, they will need to train on huge amounts of data. But that does not mean we cannot train a simple deep learning model to check whether such models can work or not.

In fact, in this repo, we will use around 1300 images for training a deep learning model. For plant disease recognition, this may not seem much. But as we will see later on, this is a good starting point, which gives us a deep learning model that works fairly well.

### The Plant Disease Recognition Dataset

We will use the plant disease recognition dataset from Kaggle to train [dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset) a deep learning model in this post.

The dataset contains images of leaves from different plants which may or may not be affected by a disease.

This dataset contains a total of 1530 images with three classes:

* Healthy: Leaves with no diseases.
* Powdery: These are the leaves that are affected by powdery mildew disease. It is a type of fungal disease that can affect plants based on the time of year. You can read more about the disease here.
* Rust: The rust disease can affect different plants. It is a type of fungal disease as well.

### ResNet18 from Scratch Training

In this subsection, we will train the ResNet18 that we built from scratch in the last tutorial.

All the code is ready, we just need to execute the train.py script with the --model argument from the project directory.

**python src/train.py --epochs 20**

As you may see, we are getting almost 100% training accuracy. The validation accuracy is already 100%. The training accuracy may be a bit lower because of the augmentations. Looks like training a few more epochs would give slightly better training results as well.

Let’s take a look at the accuracy and loss graphs.

<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_scratch_accuracy.png" width="520"/> 

<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_scratch_loss.png" width="520"/> 

As we can see, the validation accuracy is already 100% from the very first epoch. This happens when using the momentum factor along with the SGD optimizer. Without the momentum factor, the convergence is slightly slower.


### Testing the Trained Model for Plant Disease Recognition

We already have the trained model with us. This section will accomplish two things.

Run the trained model on the test images using the test.py script.
Visualize the class activation maps on the test image using the cam.py script.
To run the test script, execute the following command in the terminal.

**python src/test.py**

With the currently trained model, we get more than 97% accuracy.

Now, to visualize the class activation maps, run the following command.

**python src/cam.py**

The above will carry out the testing of the model as well. But we are interested only in the class activation maps which are saved in the outputs/cam_results directory.

### Summary and Conclusion

In this blog post, we used deep learning to solve a real-world problem on a small scale. We trained a ResNet34 model for plant disease recognition. After training the model, we tested it on the test set and also visualized the class activation maps. This gave us better insights into what the model is looking at while making predictions. 