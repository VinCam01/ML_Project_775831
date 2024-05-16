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
In this section we will explore the different 




