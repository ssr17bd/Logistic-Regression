# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 02:02:33 2020

@author: Shaila Sarker
"""
import numpy as np
import pandas as pd
claim = pd.read_csv("D:/DS/3. HireAttorney using Logistic Regression/claimants.csv")

#dropping unnecessary column CASENUM
claim = claim.drop('CASENUM', axis = 1) #drop column, so axis = 1
claim.head()
claim.describe()

# Multiple Linear Regression or MLR [as output, Y = MPG is continuous data and we've multiple inputs X1 = Waist]
import statsmodels.formula.api as smf
model = smf.logit('ATTORNEY ~ CLMSEX + CLMINSUR + SEATBELT + CLMAGE + LOSS', data = claim).fit() #logit = logistic Regression
model.summary()
#acquired model: -0.2 + 0.4330*CLMSEX + 0.6022*CLMINSUR - 0.7811*SEATBELT + 0.0065*CLMAGE - 0.3850*LOSS

#Accuracy of the Model
#cross-check Predicted values with Actual values
pred = model.predict(claim.iloc[ :, 1: ]) #[all rows, column1 to onwards (as column0 is Y=outputValue)]
pred # Pred >> [P = e^Y/(1+e^Y)]

#create a new column named "pred" and assign 0 to all 1340 Rows
claim["Prediction"] = np.zeros(1340) 
claim.loc[pred>0.5, "Prediction"] = 1 #if P > 0.5, then Prediction = 1, otherwise Prediction = 0 remains

confusion_matrix = pd.crosstab(claim.Prediction, claim.ATTORNEY) #crossTable
confusion_matrix #diagonal [(0,0), (1,1)] shows the right predictions

accuracy = (487+393)/(487+262+198+393)
print(accuracy)
 