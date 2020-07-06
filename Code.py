# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 07:12:32 2020

@author: Windows 10
"""


import pandas as pd
import numpy as np

hotel_bookings = pd.read_csv('hotel_bookings.csv',sep=',')

hotel_bookings = hotel_bookings.drop(['reservation_status','reservation_status_date'], axis=1)

hotel_bookings.country = hotel_bookings.country.astype(str)
corr_matrix = hotel_bookings.corr()

x=hotel_bookings.copy()


#missing value
# hotel_bookings.country = hotel_bookings.country.replace(np.nan,'AAA')
hotel_bookings.market_segment = hotel_bookings.market_segment.replace('Undefined','AAA')
hotel_bookings.distribution_channel = hotel_bookings.distribution_channel.replace('Undefined','AAA')
hotel_bookings.agent.fillna(1000, inplace = True) 
hotel_bookings.company.fillna(1000, inplace = True) 
hotel_bookings.adr=hotel_bookings.adr.replace(-6.38,)
hotel_bookings.children.fillna(0, inplace = True)
hotel_bookings.meal=hotel_bookings.meal.replace('Undefined','SC')

columns_hotel = hotel_bookings.columns
print(hotel_bookings.isna().sum())
#-----------------------------------------------label encoder-----------------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(hotel_bookings['hotel'])
hotel_bookings['hotel'] = le.transform(hotel_bookings['hotel'])

le.fit(hotel_bookings['arrival_date_month'])
hotel_bookings['arrival_date_month'] = le.transform(hotel_bookings['arrival_date_month'])

le.fit(hotel_bookings['meal'])
hotel_bookings['meal'] = le.transform(hotel_bookings['meal'])

le.fit(hotel_bookings['country'])
hotel_bookings['country'] = le.transform(hotel_bookings['country'])

le.fit(hotel_bookings['market_segment'])
hotel_bookings['market_segment'] = le.transform(hotel_bookings['market_segment'])

le.fit(hotel_bookings['distribution_channel'])
hotel_bookings['distribution_channel'] = le.transform(hotel_bookings['distribution_channel'])

le.fit(hotel_bookings['reserved_room_type'])
hotel_bookings['reserved_room_type'] = le.transform(hotel_bookings['reserved_room_type'])

le.fit(hotel_bookings['assigned_room_type'])
hotel_bookings['assigned_room_type'] = le.transform(hotel_bookings['assigned_room_type'])

le.fit(hotel_bookings['deposit_type'])
hotel_bookings['deposit_type'] = le.transform(hotel_bookings['deposit_type'])

le.fit(hotel_bookings['customer_type'])
hotel_bookings['customer_type'] = le.transform(hotel_bookings['customer_type'])

#--------------------------------------------------------------------------------------------

hotel_bookings.country = hotel_bookings.country.replace(0,np.nan) #merubah yg nilainya 0 jadi NaN (NULL)
hotel_bookings.market_segment = hotel_bookings.market_segment.replace(0,np.nan) #merubah yg nilainya 0 jadi NaN (NULL)
hotel_bookings.distribution_channel=hotel_bookings.distribution_channel.replace(0,np.nan) #merubah yg nilainya 0 jadi NaN (NULL)
hotel_bookings.agent=hotel_bookings.agent.replace(max(hotel_bookings.agent),np.nan) #merubah yg nilainya max(sebelumnya nilainya 1000) jadi Nan
hotel_bookings.company=hotel_bookings.company.replace(max(hotel_bookings.company),np.nan)#merubah yg nilainya max(sebelumnya nilainya 1000) jadi Nan

hotel_bookings.country = le.fit_transform(hotel_bookings['country']) #Label Endoder yg NaN tadi jadi diurutan tengah2 (krn huruf N dari NaN)
hotel_bookings.market_segment = le.fit_transform(hotel_bookings.market_segment) #Label Endoder yg NaN tadi jadi diurutan tengah2 (krn huruf N dari NaN)
hotel_bookings.distribution_channel = le.fit_transform(hotel_bookings.distribution_channel) #Label Endoder yg NaN tadi jadi diurutan tengah2 (krn huruf N dari NaN)
hotel_bookings.agent = le.fit_transform(hotel_bookings.agent) #Label Endoder yg NaN tadi jadi diurutan tengah2 (krn huruf N dari NaN)
hotel_bookings.company = le.fit_transform(hotel_bookings.company) #Label Endoder yg NaN tadi jadi diurutan tengah2 (krn huruf N dari NaN)
#------------------------------------------------------------------------------------------

# #----------------------------Iterative Imputer------------------------------------------------
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# imputer = IterativeImputer()
# imputer.fit(hotel_bookings)
# hotel = imputer.transform(hotel_bookings)

# hotel_bookings = pd.DataFrame(hotel,columns=columns_hotel)

# #-------------------------------Pembagian Kelas dan H1, H2------------------------------------------

hotel_bookings_H1 = hotel_bookings[hotel_bookings['hotel'] == 0]
hotel_bookings_H2 = hotel_bookings[hotel_bookings['hotel'] == 1]

# hotel_bookings.to_csv('hotel_bookings_after.csv')
# hotel_bookings_H1.to_csv('hotel_bookings_H1_after.csv')
# hotel_bookings_H2.to_csv('hotel_bookings_H2.after.csv')

hotel_bookings_class = hotel_bookings['is_canceled']
hotel_bookings_H1_class = hotel_bookings_H1['is_canceled']
hotel_bookings_H2_class = hotel_bookings_H2['is_canceled']

hotel_bookings_H1 = hotel_bookings_H1.drop(['hotel'], axis=1)
hotel_bookings_H2 = hotel_bookings_H2.drop(['hotel'], axis=1)

corr_matrixH1 = hotel_bookings_H1.corr()
corr_matrixH2 = hotel_bookings_H2.corr()

hotel_bookings = hotel_bookings.drop('is_canceled', axis=1)
hotel_bookings_H1 = hotel_bookings_H1.drop(['is_canceled'], axis=1)
hotel_bookings_H2 = hotel_bookings_H2.drop(['is_canceled'], axis=1)

columns_hotel = hotel_bookings.columns
columns_hotel_H1 = hotel_bookings_H1.columns
columns_hotel_H2 = hotel_bookings_H2.columns


#------------------------------Decision Tree biasa tanpa seleksi fitur dan smoothing------------------------------------------

print("--------------------Decision Tree Accuracy-----------------------")
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print('Decision Tree')
#K-Fold 7 : 0.977074171797127

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train, x_test, y_train, y_test = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]
        
dctree = DecisionTreeClassifier()
dctree.fit(x_train,y_train)
y_predtree = dctree.predict(x_test)
cross = accuracy_score(y_test,y_predtree)
    
print("K-Fold: %.3f" %cross)
print('\n')

#H1
#K-Fold 4 : 0.9766538926986689
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train1, x_test1, y_train1, y_test1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
        
dctree = DecisionTreeClassifier()
dctree.fit(x_train1,y_train1)
y_predtree1 = dctree.predict(x_test1)
cross1 = accuracy_score(y_test1,y_predtree1)
    
print("K-Fold H1 : %.3f" % cross1)
    
print('\n')

#H2
#K-Fold H2 8 : 0.9788296385060915

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train2, x_test2, y_train2, y_test2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H2.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
        
dctree = DecisionTreeClassifier()
dctree.fit(x_train2,y_train2)
y_predtree2 = dctree.predict(x_test2)
cross2 = accuracy_score(y_test2,y_predtree2)
    
print("K-Fold H2:  %.3f" %cross2)
print('\n')

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train, x_test, y_train, y_test = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

print('Random Forest')
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train, x_test, y_train, y_test = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)
# Use the forest's predict method on the test data
predictions = rf.predict(x_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
crossRF = accuracy_score(y_test,predictions)  
print("Akurasi random forest:  %.3f" %crossRF)

# from sklearn.feature_selection import SelectFromModel
# # for feature in zip(columns_hotel_H1, rf.feature_importances_):
# #     print(feature)

# feature_imp = pd.Series(rf.feature_importances_,index=columns_hotel).sort_values(ascending=False)
# feature_imp
# sfm = SelectFromModel(rf, threshold=0.15)
# sfm.fit(x_train, y_train)
# for feature_list_index in sfm.get_support(indices=True):
#     print(columns_hotel[feature_list_index])
# cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train1, x_test1, y_train1, y_test1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
        
rf1 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf1.fit(x_train1, y_train1)
# Use the forest's predict method on the test data
predictions1 = rf1.predict(x_test1)
# Calculate the absolute errors
errors1 = abs(predictions1 - y_test1)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape1 = 100 * (errors1 / y_test1)
# Calculate and display accuracy
accuracy1 = 100 - np.mean(mape1)
print('Accuracy:', round(accuracy1, 2), '%.')
crossRF1 = accuracy_score(y_test1,predictions1)  
print("Akurasi random forest H1:  %.3f" %crossRF1)

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train2, x_test2, y_train2, y_test2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H2.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
    
rf2 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf2.fit(x_train2, y_train2)
# Use the forest's predict method on the test data
predictions2 = rf2.predict(x_test2)
# Calculate the absolute errors
errors2 = abs(predictions2 - y_test2)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape2 = 100 * (errors2 / y_test2)
# Calculate and display accuracy
accuracy2 = 100 - np.mean(mape2)
print('Accuracy:', round(accuracy2, 2), '%.')
crossRF2 = accuracy_score(y_test2,predictions2)  
print("Akurasi random forest H2:  %.3f" %crossRF2)

print("-------------- Seleksi Fitur ------------------")
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

pca=PCA()
pca = pca.fit(x_train_pca)

# X_train_pca = pca.transform(x_train_pca)
# X_test_pca = pca.transform(x_test_pca)

# cek fitur paling berpengaruh
fitur_varian = pd.DataFrame(pca.explained_variance_ratio_)
plt.plot(fitur_varian)

for col in fitur_varian:
    fitur_varian[col] = fitur_varian[col]*100

for col in fitur_varian:
    print(fitur_varian[col])

pca1 = PCA()
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train_pca1, x_test_pca1, y_train_pca1, y_test_pca1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
     
pca1 = pca1.fit(x_train_pca1)

# X_train_pca1 = pca.transform(x_train_pca1)
# X_test_pca1= pca.transform(x_test_pca1)

# cek fitur paling berpengaruh
fitur_varian1 = pd.DataFrame(pca1.explained_variance_ratio_)
plt.plot(fitur_varian1)
for col in fitur_varian1:
    fitur_varian[col] = fitur_varian[col]*100

for col in fitur_varian1:
    print(fitur_varian[col])

pca2 = PCA()
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train_pca2, x_test_pca2, y_train_pca2, y_test_pca2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
     
pca2 = pca2.fit(x_train_pca2)

# X_train_pca2 = pca.transform(x_train_pca2)
# X_test_pca2= pca.transform(x_test_pca2)

# cek fitur paling berpengaruh
fitur_varian2 = pd.DataFrame(pca2.explained_variance_ratio_)
plt.plot(fitur_varian2)
for col in fitur_varian1:
    fitur_varian[col] = fitur_varian[col]*100

for col in fitur_varian1:
    print(fitur_varian[col])
# feature_imp = pd.Series(rf.feature_importances_,index=hotel_bookings.columns_hotel).sort_values(ascending=False)
# feature_imp

#PCA dan nilai akurasi untuk hotel_bookings
pca = PCA(n_components=3)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

rf_pca = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca = DecisionTreeClassifier()
pca = pca.fit(x_train_pca)
X_train_pca = pca.transform(x_train_pca)
X_test_pca = pca.transform(x_test_pca)
rf_pca.fit(X_train_pca, y_train_pca)
rf_pca_pred = rf_pca.predict(X_test_pca)
acc_rf_pca = accuracy_score(y_test_pca,rf_pca_pred)
print("Akurasi random forest dengan 3 PCA %.3f" %acc_rf_pca)
# dctree = DecisionTreeClassifier()
dctree_pca.fit(X_train_pca, y_train_pca)
y_predtreepca = dctree_pca.predict(X_test_pca)
acc_dt_pca = accuracy_score(y_test_pca,y_predtreepca)
print("Akurasi decision tree dengan 3 PCA %.3f" %acc_dt_pca)

pca1 = PCA(n_components=3)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train_pca1, x_test_pca1, y_train_pca1, y_test_pca1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
     
rf_pca1 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca1 = DecisionTreeClassifier()
pca1 = pca1.fit(x_train_pca1)
X_train_pca1 = pca1.transform(x_train_pca1)
X_test_pca1 = pca1.transform(x_test_pca1)
rf_pca1.fit(X_train_pca1, y_train1)
rf_pca_pred1 = rf_pca1.predict(X_test_pca1)
acc_rf_pca1 = accuracy_score(y_test_pca1,rf_pca_pred1)
print("Akurasi random forest H1 dengan 3 PCA: %.3f" %acc_rf_pca1)
# dctree = DecisionTreeClassifier()
dctree_pca1.fit(X_train_pca1, y_train_pca1)
y_predtreepca1 = dctree_pca1.predict(X_test_pca1)
acc_dt_pca1 = accuracy_score(y_test_pca1,y_predtreepca1)
print("Akurasi decision tree H1 dengan 3 PCA: %.3f" %acc_dt_pca1)

pca2 = PCA(n_components=3)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train_pca2, x_test_pca2, y_train_pca2, y_test_pca2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H2.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
     
rf_pca2 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca2 = DecisionTreeClassifier()
pca2 = pca2.fit(x_train_pca2)
X_train_pca2 = pca2.transform(x_train_pca2)
X_test_pca2 = pca2.transform(x_test_pca2)
rf_pca2.fit(X_train_pca2, y_train2)
rf_pca_pred2 = rf_pca2.predict(X_test_pca2)
acc_rf_pca2 = accuracy_score(y_test_pca2,rf_pca_pred2)
print("Akurasi random forest H2 dengan 3 PCA: %.3f" %acc_rf_pca2)
# dctree = DecisionTreeClassifier()
dctree_pca2.fit(X_train_pca2, y_train_pca2)
y_predtreepca2 = dctree_pca2.predict(X_test_pca2)
acc_dt_pca2 = accuracy_score(y_test_pca2,y_predtreepca2)
print("Akurasi decision tree H2 dengan 3 PCA: %.3f" %acc_dt_pca2)
# %% [5 pca]
pca = PCA(n_components=5)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

rf_pca_5 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca_5 = DecisionTreeClassifier()
pca = pca.fit(x_train_pca)
X_train_pca = pca.transform(x_train_pca)
X_test_pca = pca.transform(x_test_pca)
rf_pca_5.fit(X_train_pca, y_train_pca)
rf_pca_pred_5 = rf_pca_5.predict(X_test_pca)
acc_rf_pca_5 = accuracy_score(y_test_pca,rf_pca_pred_5)
print("akurasi model random forest dengan 5 PCA : %.3f" %acc_rf_pca_5)
# dctree = DecisionTreeClassifier()
dctree_pca_5.fit(X_train_pca, y_train_pca)
y_predtreepca_5 = dctree_pca_5.predict(X_test_pca)
acc_dt_pca_5 = accuracy_score(y_test_pca,y_predtreepca_5)
print("Akurasi model decision tree dengan 5 PCA: %.3f" %acc_dt_pca_5)

pca1 = PCA(n_components=5)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train_pca1, x_test_pca1, y_train_pca1, y_test_pca1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
     
rf_pca1_5 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca1_5 = DecisionTreeClassifier()
pca1 = pca1.fit(x_train_pca1)
X_train_pca1 = pca1.transform(x_train_pca1)
X_test_pca1 = pca1.transform(x_test_pca1)
rf_pca1_5.fit(X_train_pca1, y_train1)
rf_pca_pred1_5 = rf_pca1_5.predict(X_test_pca1)
acc_rf_pca1_5 = accuracy_score(y_test_pca1,rf_pca_pred1_5)
print("akurasi model random forest H1 dengan 5 PCA : %.3f" %acc_rf_pca1_5)
# dctree = DecisionTreeClassifier()
dctree_pca1_5.fit(X_train_pca1, y_train_pca1)
y_predtreepca1_5 = dctree_pca1_5.predict(X_test_pca1)
acc_dt_pca1_5 = accuracy_score(y_test_pca1,y_predtreepca1_5)
print("akurasi model decision tree H1 dengan 5 PCA : %.3f" %acc_dt_pca1_5)

pca2 = PCA(n_components=5)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train_pca2, x_test_pca2, y_train_pca2, y_test_pca2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H2.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
     
rf_pca2_5 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca2_5 = DecisionTreeClassifier()
pca2 = pca2.fit(x_train_pca2)
X_train_pca2 = pca2.transform(x_train_pca2)
X_test_pca2 = pca2.transform(x_test_pca2)
rf_pca2_5.fit(X_train_pca2, y_train2)
rf_pca_pred2_5 = rf_pca2_5.predict(X_test_pca2)
acc_rf_pca2_5 = accuracy_score(y_test_pca2,rf_pca_pred2_5)
print("akurasi model random forest H2 dengan 5 PCA : %.3f" %acc_rf_pca2_5)
# dctree = DecisionTreeClassifier()
dctree_pca2_5.fit(X_train_pca2, y_train_pca2)
y_predtreepca2_5 = dctree_pca2_5.predict(X_test_pca2)
acc_dt_pca2_5 = accuracy_score(y_test_pca2,y_predtreepca2_5)
print("akurasi model decison tree H2 dengan 5 PCA : %.3f" %acc_dt_pca2_5)

# %% [14 pca]
pca = PCA(n_components=14)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_class):
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = hotel_bookings.iloc[train_index], hotel_bookings.iloc[test_index], hotel_bookings_class.iloc[train_index], hotel_bookings_class.iloc[test_index]

rf_pca_14 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca_14 = DecisionTreeClassifier()
pca = pca.fit(x_train_pca)
X_train_pca = pca.transform(x_train_pca)
X_test_pca = pca.transform(x_test_pca)
rf_pca_14.fit(X_train_pca, y_train_pca)
rf_pca_pred_14 = rf_pca_14.predict(X_test_pca)
acc_rf_pca_14 = accuracy_score(y_test_pca,rf_pca_pred_14)
print("akurasi model random forest dengan 14 PCA : %.3f" %acc_rf_pca_14)
# dctree = DecisionTreeClassifier()
dctree_pca_14.fit(X_train_pca, y_train_pca)
y_predtreepca_14 = dctree_pca_14.predict(X_test_pca)
acc_dt_pca_14 = accuracy_score(y_test_pca,y_predtreepca_14)
print("Akurasi model decision tree dengan 14 PCA: %.3f" %acc_dt_pca_14)

pca1 = PCA(n_components=14)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H1_class):
    x_train_pca1, x_test_pca1, y_train_pca1, y_test_pca1 = hotel_bookings_H1.iloc[train_index], hotel_bookings_H1.iloc[test_index], hotel_bookings_H1_class.iloc[train_index], hotel_bookings_H1_class.iloc[test_index]
     
rf_pca1_14 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca1_14 = DecisionTreeClassifier()
pca1 = pca1.fit(x_train_pca1)
X_train_pca1 = pca1.transform(x_train_pca1)
X_test_pca1 = pca1.transform(x_test_pca1)
rf_pca1_14.fit(X_train_pca1, y_train1)
rf_pca_pred1_14 = rf_pca1_14.predict(X_test_pca1)
acc_rf_pca1_14 = accuracy_score(y_test_pca1,rf_pca_pred1_14)
print("akurasi model random forest H1 dengan 14 PCA : %.3f" %acc_rf_pca1_14)
# dctree = DecisionTreeClassifier()
dctree_pca1_14.fit(X_train_pca1, y_train_pca1)
y_predtreepca1_14 = dctree_pca1_14.predict(X_test_pca1)
acc_dt_pca1_14 = accuracy_score(y_test_pca1,y_predtreepca1_14)
print("akurasi model decision tree H1 dengan 14 PCA : %.3f" %acc_dt_pca1_14)

pca2 = PCA(n_components=14)
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(hotel_bookings_H2_class):
    x_train_pca2, x_test_pca2, y_train_pca2, y_test_pca2 = hotel_bookings_H2.iloc[train_index], hotel_bookings_H2.iloc[test_index], hotel_bookings_H2_class.iloc[train_index], hotel_bookings_H2_class.iloc[test_index]
     
rf_pca2_14 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
dctree_pca2_14 = DecisionTreeClassifier()
pca2 = pca2.fit(x_train_pca2)
X_train_pca2 = pca2.transform(x_train_pca2)
X_test_pca2 = pca2.transform(x_test_pca2)
rf_pca2_14.fit(X_train_pca2, y_train2)
rf_pca_pred2_14 = rf_pca2_14.predict(X_test_pca2)
acc_rf_pca2_14 = accuracy_score(y_test_pca2,rf_pca_pred2_14)
print("akurasi model random forest H2 dengan 5 PCA : %.3f" %acc_rf_pca2_14)
# dctree = DecisionTreeClassifier()
dctree_pca2_14.fit(X_train_pca2, y_train_pca2)
y_predtreepca2_14 = dctree_pca2_14.predict(X_test_pca2)
acc_dt_pca2_14 = accuracy_score(y_test_pca2,y_predtreepca2_14)
print("akurasi model decison tree H2 dengan 5 PCA : %.3f" %acc_dt_pca2_14)


