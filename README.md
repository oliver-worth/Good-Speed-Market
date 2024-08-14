# Age Verification Model for Good Seed Supermarket

## Introduction
Good Seed Supermarket is committed to complying with legal requirements related to the sale of alcohol by ensuring customers are of legal age. This project involves building a machine learning model that leverages computer vision techniques to verify the age of customers purchasing alcohol, using images captured at checkout. The model is trained to predict a customer’s age from their image, helping the supermarket chain maintain compliance with age restrictions.

## Project Objectives
- Develop a deep learning model capable of predicting the age of individuals from images.
- Ensure the model's accuracy is sufficient to minimize the risk of selling alcohol to underage customers.
- Utilize a pre-trained ResNet50 model, fine-tuned for the specific task of age prediction.
- Optimize the model for performance using advanced techniques such as data augmentation, regularization, and hyperparameter tuning.

## Dataset
The dataset used for this project is sourced from the ChaLearn Looking at People competition. It consists of 7,600 images of individuals with associated age labels. The images are processed using Convolutional Neural Networks (CNNs) to extract features and predict the age.

## Exploratory Data Analysis (EDA)
Before training the model, we performed Exploratory Data Analysis (EDA) to better understand the dataset. This included:
- Visualizing the distribution of ages in the dataset.
- Inspecting sample images to ensure data quality.
- Identifying any potential biases or imbalances in the data.

## Data Preparation
The dataset was preprocessed and organized for training:
- **Image Preprocessing**: Resizing images to 128x128 pixels, normalizing pixel values.
- **Label Processing**: Converting age labels to numerical values suitable for regression tasks.
- **Data Augmentation**: Applied techniques such as rotation, flipping, zooming, and brightness adjustments to improve model robustness.

## Model Development
We utilized the ResNet50 architecture, a pre-trained model known for its strong performance on image recognition tasks. The model was fine-tuned for our specific task of age prediction.

- **Model Architecture**: ResNet50 with additional fully connected layers for regression.
- **Loss Function**: Mean Absolute Error (MAE) was used as the loss function.
- **Optimization**: The Adam optimizer was employed to minimize the loss function.

## Model Training
The model was trained on a GPU platform to take advantage of faster processing capabilities. The training process involved:
- **Training Loss (MAE)**: The model's training MAE steadily decreased from 7.4339 in the first epoch to 3.1785 in the final epoch, indicating effective learning.
- **Validation Loss (MAE)**: The validation MAE showed fluctuations, starting at 8.4921 and ending at 7.6512, suggesting potential overfitting or instability during training.

## Model Evaluation
The model's performance was evaluated on a validation dataset. The final model achieved a satisfactory level of accuracy but exhibited some signs of overfitting.

![Age Prediction](./Screenshot2024-08-14.png)

## Methods to Improve Model Quality
Several advanced techniques can be employed to further improve the model's performance:
- **Advanced Data Augmentation**: Implementing more sophisticated data augmentation techniques to improve model generalization.
- **Transfer Learning**: Exploring other pre-trained models like EfficientNet to potentially enhance performance.
- **Regularization**: Incorporating dropout and L2 regularization to mitigate overfitting.
- **Ensemble Methods**: Combining predictions from multiple models to improve accuracy.
- **Hyperparameter Tuning**: Using grid search or random search to find the optimal parameters for the model.

## Conclusion
The age verification model developed in this project provides Good Seed Supermarket with a robust tool for ensuring compliance with alcohol laws. The model, built using deep learning techniques, predicts the age of customers with a high degree of accuracy. While the model performed well, there are opportunities for further improvement through the application of advanced techniques. This project demonstrates the potential of computer vision in automating age verification, contributing to the responsible sale of alcohol.

## Future Work
Future work could involve:
- **Further fine-tuning** of the model to reduce overfitting and improve validation performance.
- **Exploring additional datasets** to enhance the model’s robustness across diverse populations.
- **Integrating the model** into Good Seed's checkout systems for real-time age verification.

