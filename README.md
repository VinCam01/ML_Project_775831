# Machine Learning Project - BIP Group

**Team Members**: Raffaele Torelli 775831 - Daniele De Robertis 787291 - Vincenzo Camerlengo 773731

## 1 - Introduction
On 1 January 2019, the Italian government introduced legislation requiring that all invoices issued between parties in Italy must be transmitted exclusively in electronic format through the "*Sistema di Interscambio*" (SDI) of the Italian "*Agenzia delle Entrate*". To simplify this process, several dedicated portals and software can be used, including BIP xTech and TeamSystem. The latter are consistently engaged in the pursuit of novel technologies and methodologies with the objective of optimising the user experience. They espouse a customer-centric approach to innovation, whereby their designers integrate the needs of individuals, technological possibilities and requirements for business success.

It is well established that certain invoice types are not subject to value-added tax and enjoy tax exemptions. The nature of these exemptions is coded using a nature code, which comprises 21 different values. Additionaly, users has to map the nature code to exemption code that represent the reason why the invoice is not subject to VAT. This task is complex and sophisticated due to the presence of 64 exemption codes, which necessitates that a nature code may correspond to more than one IvaM code. 

The objective of this project is to develop a machine learning model that is capable of predicting and suggesting the VAT exemption code (IvaM) to users based on the data on the invoice. This will enable the process to be streamlined. The following sections will engage in a comprehensive examination of the argument presented in this work.

## 2 - Methods
The company provides us with a substantial dataset, comprising approximately 130,000 invoice lines and 45 characteristics for each of them, including very useful informations like ATECO code, document type, invoice type, VAT rate, article description, amount and many more. 

### 2.1 - Data Preprocessing
The first step of our work has been the visualization of missing value:

<div align="center">
  <img src="images/nan.png" alt="">
</div>

<p align="right">
  <em><small>Figure 1</small></em>
</p>

As we can see in *Figure 1*, in our dataset there were some columns with almost only null values. So we decided to drop the variable that have more than 100,000 null values and also the ones that were not important to reach our goal. Then, in order to manage the remaining NaN values, we fill them with the most frequent class within the variable. This approach allows us to preserve the integrity of our data and avoid to lose potentially useful informations. 

Sequently, we focus our attention on the variables, making them suitable for the prediction. The first problem we encountered was the presence of unbalanced classes in many columns: in other words, there were many classes with a few observations. To overcome this issue, we chose a threshold, below which all classes were grouped into a new class called 'OTHER'. 

<div align="center">
  <img src="images/iva_tdoc.png" alt="">
</div>

<p align="right">
  <em><small>Figure 2</small></em>
</p>

In *Figure 2* is showed an example on how change the distibution of the variables '*Iva*' and '*Tdoc*' after that operation. We chosen a thresold of 1,200 occurrences for the first and 1,000 for the second.
Very interesting was also the case of the '*Ateco*' column: there were 386 classes extremely unbalanced and the visualization was horrible.

<div align="center">
  <img src="images/ateco.png" alt="">
</div>

<p align="right">
  <em><small>Figure 3</small></em>
</p>

As we can see in *Figure 3*, here it has been done a more sophisticated job: before the union of the less frequent classes, we grouped the '*Ateco*' code according to the first two digits. In fact, conducting a search, we discovered that they represent the macrocategory of the commercial activities.
Lastly, we decided to maintain the same approach also for our response variable, '*IvaM*', in order to maintain high sensitivity in the model.

<div align="center">
  <img src="images/ivam.png" alt="">
</div>

<p align="right">
  <em><small>Figure 4</small></em>
</p>

Without this procedure it would have been very difficult for the model to predict exemption codes that are present very few times in the train data. However, in this case we adopted a lower threshold of 250 to avoid oversimplification of the model.

Once the classes were balanced, we proceeded implementing binary encoding and one-hot-encoding for the categorical variables. Then, since after encoding the number of columns increased, we implemented the Cramer's method in order to eliminate or put in the class 'OTHER', all those variables that have a Cramer's coefficient less than 0.25; indicating a low correlation with the response variable. The Cramer's method is a statistical technique employed to assess the correlation between two categorical variables represented in a contingency table. It entails calculating the chi-square statistic for the table and normalising it to account for table size. The resulting value, known as Cramer's coefficient, ranges from 0 to 1, with higher values indicating a stronger association. It is used to quantify the degree of relationship between categorical variables.

### 2.2 - Models 
In this section we will explore all the statistical and machine learning models that we implemented. For every model has been done a k-fold cross-validation with k equal to five. Additionally, where possible has been used the Optuna hyperparameter tuning to discover the best parameters. We choose this optimizator because is computationally less expensive then Grid Search, and performs better compared to the Random Search; in fact Optuna employs Bayesian optimisation to efficiently explore the hyper-parameter space, focusing on hyper-parameter sets that demonstrate optimal performance during the search process and incorporating historical performance data.

#### 2.2.1 - Logistic Regression
The first model used is a Logistic Regression.

**Key Points**:
- Used to model the relationship between a binary dependent variable and one or more independent variables.
- Estimates the probability of occurrence of the event of interest.
- Produces coefficients indicating the effect of independent variables on the probability of success.

As we will see in next sections, despite being the 'simplest' of the models used, the metrics are still very high

#### 2.2.2 - Random Forest
Secondly we implemented a Random Forest.

**Key Points**:
- Ensemble learning method used for classification tasks.
- Builds multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.
- Each tree is trained on a random subset of the training data and features, promoting diversity among the trees.
- Handles non-linear relationships well and is robust to overfitting.

**Hyperparameters**:
- n_estimators: 117 -> number of trees in the forest.
- 'max_depth': 27 -> maximum depth of each tree. 
- 'min_samples_split': 3 -> minimum number of samples required to split an internal node.
- 'min_samples_leaf': 1 -> minimum number of samples required in a leaf.

#### 2.2.3 - Decision Tree
Then we used a Decision Tree. 

**Key Points**:
- It represents a tree-like structure where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.
- The model makes predictions by traversing the tree from the root node to a leaf node based on the features of the input data.
- Unlike Random Forest, the decision tree model is a single tree structure.

**Hyperparameters**:
- 'max_depth': 42 -> maximum depth of the decision tree.
- 'min_samples_split': 2 -> minimum number of samples required to split an internal node.
- 'min_samples_leaf': 1 -> minimum number of samples required to be at a leaf node
- 'criterion': entropy -> specifies the function used to measure the quality of a split. 'Entropy' measures the impurity of a split, aiming to maximize information gain.

#### 2.2.4 - Support Vector Machine
Proceed our exploration implementing a Support Vector Machine.

**Key Points**:
- Finds the hyperplane that best separates classes in the feature space.
- Maximizes margin between classes' nearest data points (support vectors).
- Effective in high-dimensional spaces and robust to overfitting.

The implementation of this model took a long time, so carrying out hyperparameters tuning would have been computationally extremely time-consuming and inefficient since, as we shall see in the results section, the metrics are very good anyway.

#### 2.2.5 - K-Nearest Neighbors
Then we perforder a KNN model.

**Key Points**:
- It classifies data points based on the majority class of their nearest neighbors.
- The "K" in KNN represents the number of neighbors to consider when making predictions.
- It is non-parametric and instance-based, meaning it does not make assumptions about the underlying data distribution.

**Hyperparameters**:
- 'n_neighbors': 14 -> number of neighbors considered during classification
- 'weights': 'distance' -> specifies how neighbors are weighted during prediction. Option 'distance', means that closer neighbors have more influence on the prediction.
- 'algorithm': 'auto' -> specifies the algorithm used to compute nearest neighbors. The option 'Auto' automatically selects the most suitable algorithm based on the data size and nature.
- 'leaf_size': 21 -> determines the maximum leaf size during tree construction. Is important to notice that in this case, this hyperparameter is superfluous; infact it is specific to tree-based algorithms like ball_tree or kd_tree.

#### 2.2.6 - Artificial Neural Network
Lastly, we implemented an ANN.

**Key Points**:
- Artificial Neural Networks consist of interconnected nodes (neurons) organized in layers: input layer, hidden layers, and output layer.
- Information is processed through the network by propagating forward from input to output layers.
- Neurons in each layer apply an activation function to the weighted sum of inputs to produce output.
- ANNs are powerful for complex pattern recognition and non-linear relationships in data.
- Training involves adjusting weights and biases through backpropagation using optimization algorithms like gradient descent.

**Parameters**:
- EPOCHS = 50 -> number of times the entire dataset is passed forward and backward through the neural network during training. 
- BATCH_SIZE = 128 -> number of samples processed before the model's parameters are updated. 
- NB_CLASSES = 26 -> number of classes in the classification task.
- N_SPLITS = 5 -> number of folds for cross-validation.


## 3 - Experimental Design
In this section, we present the experiments conducted to validate the effectiveness of our model for classifying exception codes.

### 3.1 - Experiment 1
The main purpose is to evaluate the performance of the proposed model’s performance based on two metrics.
- Accuracy: measures the model’s ability to correctly predict the class of an observation, it represents the ratio of correctly classified instances to the total number of instances in the dataset.
- F1 score: provides a unique metric that include both the “precision” and “recall”. Precision measures the accuracy of positive predictions made by the model. It is calculated as the ratio of true positive predictions to the total number of positive predictions made by the model. Recall measures the model's ability to identify all relevant instances in the dataset. It is calculated as the ratio of true positive predictions to the total number of actual positive instances in the dataset.

In evaluating classification models, both accuracy and the F1 score are indispensable. Accuracy provides a general overview of correctness, but in cases of class imbalance, it may be skewed by dominant classes. Meanwhile, the F1 score, by combining precision and recall, offers a balanced perspective, crucial for discerning the model's ability to correctly identify minority classes and minimize false positives/negatives. Together, these metrics ensure a comprehensive assessment, considering both overall correctness and the model's precision-recall balance, vital for real-world applicability.
### 3.2 - Experiment 2
The main purpose is to compare the models based on the chosen metrics to find the most performant one and implement it as the final solution.

## 4 - Results
This section discusses and compares the performance of different models to determine which one is the most effective for classifying exception codes.

### **4.1 - Logistic Regression Results**


- **Mean Accuracy**: 93,73%
- **F1 Score**: 0.9344


### **4.2 - Random Forest Results** 
We have introduced the Random Forest model, that is a powerful ensemble learning method in handling complex classification tasks and providing accurate predictions. <br>
Considering the evaluation metrics and all the previous findings, we can estimate that the average accuracy is 96.6 % and the F1 score is 97% for IvaM exemptions codes prediction in the Random Forest model: <br>

- **Mean Accuracy**: 96.63%
- **F1 Score**: 0.9659

### **4.3 - Decision Tree Results** 
The Decision Tree model has showcased remarkable performance, registering notable accuracy and F1 scores values:

- **Mean accuracy**: 96.21%
- **F1 Score**: 0.9620

This indicates its proficiency in accurately classifying instances and effectively balancing precision and recall.

### **4.4 - Support Vector Machine Results** 
SVMs operate by finding the optimal hyperplane that best separates different classes in the feature space, maximizing the margin between them. The Support Vector Machine model demonstrates us a slight degrowth over previously tested models, achieving consistent accuracy levels.

- **Mean accuracy**: 95.12%
- **F1 Score**: 0.9503

This suggests a marginal but notable deterioration in the model's learning capability. <br>
For this model, we had to stop hyperparameters tuning process because we have noticed it was extremely computationally expensive, so we decided to don't perform it since the metres were already good.

### **4.5 - K-Nearest Neighbours Results** 
We have implemented a KNN model with the goal of exploiting the similarities between the exemption codes to make more accurate predictions. Considering the KNN results, there was an ascent regarding our parameters, going to confirm the significance and validity of the model.

- **Mean Accuracy**: 96.36%
- **F1 Score**: 0.9633

### **4.6 Artificial Neural Network Results** 
Lastly, we have introduced an ANN model to learn from data and make predictions or decisions accurately and efficiently. For this particular model, we have added a third parameter, the mean loss, to evaluate its performance during training. The mean loss provides insight regarding how the model is minimizing errors between its predictions and the actual values in the training data.

- **Mean Accuracy for 5 folds**: 95.33%
- **F1 Score**: 0.9515
- **Mean loss is**: 0.1479

The mean loss provides insight regarding how the model is minimizing errors between its predictions and the actual values in the training data. By monitoring and optimizing this, we aim to improve the model's ability.

**4.7 Global Results** <br> 
We have compared the 3 models that had the highest accuracy and F1 score values, to evaluate which model was the best according to these two parameters.

<div align="center">
  <img src="images/model_comparison.png" alt="">
</div>

<p align="right">
  <em><small>Figure 6</small></em>
</p>

This bar chart displays that Random Forest model is the most efficient one, outperforming other models in terms of both accuracy and computational efficiency. <br>

In addition, we compute the confusion matrix for IvaM exemption codes. This measure, in addition to telling us how accurate the model is, gives us an idea of "how" the model gets the prediction wrong by providing the false negative e and false positive. 

<div align="center">
  <img src="images/conf_mtx.png" alt="">
</div>

<p align="right">
  <em><small>Figure </small></em>
</p>

The confusion matrix shows that the model has a good ability to distinguish between the most common classes. The cells on the main diagonal represent the correct predictions. We note that the cells with the darkest shades of blue are on the diagonal, indicating that the model has good accuracy for these classes, especially for class 374.0 and class 324.0.

## 5 - Conclusions 
- Overall, we are pleased with the performance of our model. Our satisfaction stems from its consistent adherence to our established quality benchmarks and its ongoing improvement towards achieving optimal predictive capabilities. This reliability instills confidence in the model's ability to make accurate predictions, contributing to our overall confidence in its effectiveness for the task at hand.

- While we are generally pleased with the performance of our model, we acknowledge its limitation in predicting exemption codes categorized under the class "Others". We recognize that this issue could potentially be addressed by utilizing a more balanced dataset. By ensuring that the dataset includes a sufficient representation of instances from each exemption code category, including those within the "Others" class, we can provide the model with a more comprehensive understanding of the underlying patterns and relationships.

- This approach may enable the model to better capture the nuances associated with predicting exemption codes across all classes, thereby enhancing its overall predictive accuracy and effectiveness.



