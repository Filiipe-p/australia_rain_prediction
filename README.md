# Australia Rain Prediction
 <h3>Ironhack Final Project</h3>
 Project status: In Progress
 
 # Project objective
 The main objective of the project is to create the best possible model to predict the rain in some places in Australia
 
 # Methods

  - Filtering
  - Data cleaning
  - Classification
  - Predicting weather from a model
  - Model improvement
  - Plotting
  - Getting data from APIs
  
  # Technologies 

  - Python
  - Pandas
  - SkLearn
  - Numpy
  - Seaborn
  - Pycaret
  - Geopy
  - World Weather on-line API [link](https://www.worldweatheronline.com/)
  
  # Steps
<h3>1.Understanding the Datasets</h3>
 
 ![image](https://user-images.githubusercontent.com/104516688/208012856-6d5a334d-27b3-4417-aac8-9001011ae340.png)<br>
   As we can see, 'sunshine' has a strong correlation, but many nulls. We will solve this later.

<h3>2.Merging the winds data set and cleaning and dropping columns that will affect the models</h3>

 ![image](https://user-images.githubusercontent.com/104516688/208013217-04eaf7fd-4711-4ab8-9651-0789457b9037.png)<br>
 Let's remove 'modelo_vigente' and 'amountOfRain' to avoid leakage

<h3>3.Getting more useful information from the table: </h3>

![image](https://user-images.githubusercontent.com/104516688/208014135-78bed88d-1b32-4aa4-ae8d-50d51ee25748.png)<br>
And from the locations, we can check useful distance features using its Coordinates :

![image](https://user-images.githubusercontent.com/104516688/208014825-971cf891-355d-4775-9486-46d29c50dc78.png)<br>
In australia, the proximity with the shore is a key information when we talk about rain. So, using the API from kbgeo and the coordinates we can get those distances.<br><br>
The dataframe will look like this:<br>
![image](https://user-images.githubusercontent.com/104516688/208015075-1cbbe616-6f5a-4c8f-aa28-8481371c60ba.png)<br>

Now, with a more robust dataset, let's check the data:
![image](https://user-images.githubusercontent.com/104516688/208015385-c858c3e3-98a1-4ca3-a2b2-af0792dfc98e.png)<br>

![image](https://user-images.githubusercontent.com/104516688/208015556-a6fc85dd-1721-42c1-b949-dbf610b0cd36.png)<br><br>

Looking deeper in the target column:<br><br>
![image](https://user-images.githubusercontent.com/104516688/208015817-88641301-5a16-48ad-b991-c54c2892e7ab.png)<br><br>

The amount of days without rain is far greater than the amount of rainy days. This information can be useful to build the model
About the missing values, we can use pycaret imputation methods and see what happens...

<h3> Creating Model: </h3>

![image](https://user-images.githubusercontent.com/104516688/208016128-36fa9f83-d819-4faa-a34d-dfbb4e345d8d.png)<br><br>

![image](https://user-images.githubusercontent.com/104516688/208016441-712dfe95-e1c5-4926-aef3-1b501f0d92c6.png)

LGBM is the best model for predicting. but still, even though we had good Accuracy and AUC, the Recall score was not what we expected. Let's take a better look...
<br>
Area Under Curve:<br>
![image](https://user-images.githubusercontent.com/104516688/208016635-5e8509de-cdf7-4c49-bbf3-0df8cf07f1ab.png)<br><br>

Feature Importance:<br>
![image](https://user-images.githubusercontent.com/104516688/208016785-b19781f6-6bb3-4a7c-b8c1-563899bb9ca1.png)<br><br>

The Pressure at 9am was the most important feature in the model. And the Coast distance was not as important as expected.<br><br>

Confusion Matrix:<br>
![image](https://user-images.githubusercontent.com/104516688/208017039-801418e6-3184-4d4d-b30a-a86ef91fca21.png)<br><br>

With the confusion matrix we can visualize why the recall score was so low: Once the 'raintomorrow' feature is unbalanced, the amount of predicted negatives will be far greater than the amount of predicted of positives. So, the model will get it wrong more times by saying it will not rain and it actually rains, than the oposite.

To improve the recall score. We can create a new model, using the 'fix_imbalance' feature from pycaret...

<h3> Creating Second Model:</h3>

 ![image](https://user-images.githubusercontent.com/104516688/208017458-ec44c15b-3eba-474c-9b36-7d699409b2a7.png)<br><br>
 
 Very small improvement with the lgbm model, but depending on the finality of the analysis, we can choose a different model from this setup.
 For now, let's use this model to predict the testing data frame<br><br>
 
 ![image](https://user-images.githubusercontent.com/104516688/208017715-1de3cb59-bcdd-4106-aa3e-339c39a63188.png)<br><br>
 
 ![image](https://user-images.githubusercontent.com/104516688/208017822-a02e4b99-57ff-48ba-8e51-102f23e9d71f.png)

The model is complete!

Now, using this model and the data from December 14th 2022 (Yesterday). Let's predict if it will rain today!<br>
'Data accessed using world weather on-line API [link](https://www.worldweatheronline.com/)'


<h3>Predicting the Rain today!</h3>

Using the coordinates from the distance dataframe. it was possible to get almost all features from the original dataset.<br>

A bit of code...<br>
![image](https://user-images.githubusercontent.com/104516688/208018931-774e8e1d-f1c0-4dca-9729-8fa73bb74476.png)<br>

Voila!<br>
Data from yesterday is ready:<br>

![image](https://user-images.githubusercontent.com/104516688/208019306-210c4fba-f8ef-4b26-a733-0a3ef71a9a8e.png)<br>

Prediction using the presaved model:<br>
![image](https://user-images.githubusercontent.com/104516688/208019633-fc59df21-a0ac-4fe2-b269-1a654e42f15b.png)

<h3> Matching the Results:</h3>

Confusion tree of the model X reality:

![image](https://user-images.githubusercontent.com/104516688/208019859-725bcf85-d9a4-4879-8a44-7ef0d0923c36.png)

Lots of False negatives... That's the issue when we get low recall score, but even with the model's poor performance, this is only one day of prediction and we need more real data to evaluate deeply the model.
