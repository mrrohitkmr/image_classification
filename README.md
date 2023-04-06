# image_classification

Training a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

# Architecture
The CNN architecture used in this project is a sequential model with several convolutional layers followed by max pooling layers and dropout regularization. The last layer is a fully connected layer with a softmax activation function, which outputs the predicted class probabilities. The model is trained using the Adam optimizer and categorical cross-entropy loss function.

The project includes data preprocessing steps, including scaling the pixel values of the images to the range of 0 to 1 and one-hot encoding the class labels. Additionally, the data augmentation technique is used to artificially increase the size of the training set by applying random transformations to the images, such as rotations and horizontal flips.

# Training
During the training process, the model is evaluated on both the training and validation sets using accuracy as the performance metric. The training process is also visualized using TensorBoard, which allows for the monitoring of the loss and accuracy values during training.

After training for 100 epochs, the model achieves a test accuracy of approximately 72%. The project demonstrates the effectiveness of CNNs for image classification tasks and the importance of data preprocessing and data augmentation techniques in improving model performance.
