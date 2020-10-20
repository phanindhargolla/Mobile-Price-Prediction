#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns
import sklearn as sk


# In[2]:


mobile_data = pd.read_csv('F:\\mobile\\dataset\\train.csv')


# In[3]:


mobile_data.head()


# In[4]:


#checking for null values
mobile_data.isnull().sum()
#There are no null values


# In[5]:


mobile_data.info()


# In[6]:


mobile_data['blue'] = mobile_data['blue'].astype('category')
mobile_data['dual_sim'] = mobile_data['dual_sim'].astype('category')
mobile_data['four_g'] = mobile_data['four_g'].astype('category')
mobile_data['three_g'] = mobile_data['three_g'].astype('category')
mobile_data['touch_screen'] = mobile_data['touch_screen'].astype('category')
mobile_data['wifi'] = mobile_data['wifi'].astype('category')


# In[7]:


#sns.countplot(x='price_range',data=mobile_data)
#dataset is balanced


# In[8]:


#sns.countplot(x='blue',data=mobile_data)


# In[9]:


#sns.set(style='darkgrid')
#sns.catplot(x='price_range',y='battery_power',data=mobile_data,kind='box')
#The mobile with class of higher price has higher battery power when compared to other mobiles


# In[10]:


#sns.catplot(x='price_range',y='ram',data=mobile_data,kind='violin')
# we can see as we go to higher price range the size of ram also increases


# In[11]:


#sns.catplot(x='price_range',y='int_memory',data=mobile_data,kind='swarm')


# In[12]:


mobile_data.corr()


# In[13]:


selected_columns = ['battery_power','fc','int_memory','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range']


# In[14]:


mobile_data = mobile_data[selected_columns]


# In[15]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[16]:


mobile_data = pd.get_dummies(mobile_data,drop_first=True)


# In[17]:


minmax = MinMaxScaler()
minmax_columns = ['battery_power','fc','int_memory','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time'] 
mobile_data[minmax_columns] = pd.DataFrame(minmax.fit_transform(mobile_data[minmax_columns]))


# In[18]:


X = mobile_data[['battery_power','fc','int_memory','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time','blue_1','dual_sim_1','four_g_1','three_g_1','touch_screen_1','wifi_1']].values
y = mobile_data['price_range'].values


# In[19]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


reg_model = LogisticRegression()


# In[22]:


reg_model.fit(X_train,y_train)


# In[23]:


y_pred_reg = reg_model.predict(X_test)


# In[24]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_log = accuracy_score(y_test,y_pred_reg)


# In[25]:


confusion_log = confusion_matrix(y_test,y_pred_reg)


# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


random = RandomForestClassifier(max_depth=201,n_estimators=501)


# In[35]:


random.fit(X_train,y_train)


# In[36]:


y_pred_random = random.predict(X_test)


# In[37]:


accuracy_random = accuracy_score(y_test,y_pred_random)


# In[38]:


confusion_random = confusion_matrix(y_test,y_pred_random)


# In[32]:


from sklearn.model_selection import GridSearchCV


# In[33]:


param = {'n_estimators': [i for i in range(1,800,100)],'max_depth': [i for i in range(1,800,100)]}
gs = GridSearchCV(random,param,cv=5,n_jobs=-1)
gs_fit = gs.fit(X_train,y_train)
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score',ascending = False)[:5]


# In[41]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[42]:


model = Sequential()


# In[43]:


model.add(Dense(units=5,kernel_initializer='he_uniform',activation='relu',input_dim=17))
model.add(Dense(units=8,kernel_initializer='he_uniform',activation='relu'))
model.add(Dense(units=5,kernel_initializer='he_uniform',activation='relu'))
model.add(Dense(units=4,kernel_initializer='glorot_uniform',activation='softmax'))


# In[44]:


model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[45]:


model_hist = model.fit(X_train,y_train,validation_split=0.3,batch_size=10,epochs=100)


# In[46]:


model_hist.history.keys()


# In[47]:


y_pred_nn = model.predict(X_test)


# In[48]:


y_pred1 = []
for i in range(len(y_pred_nn)):
    y_pred1.append(y_pred_nn[i].argmax())


# In[49]:


count = 0
for i in range(len(y_pred1)):
    if y_pred1[i] == y_test[i]:
        count += 1 


# In[50]:


testing_accuracy = count/len(y_pred1)


# In[51]:


#plt.figure(figsize=(6,4))
#plt.plot(model_hist.history['accuracy'])
#plt.plot(model_hist.history['val_accuracy'])
#plt.title('model accuracy')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['train','test'],loc='upper left')


# In[58]:


from tensorflow.keras.models import model_from_json


# In[59]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# In[85]:


x = np.array([0.3,0.1,0.1,0.19,0.17,0.907,0.948,0.9482,0.18,0.0,0.2,0,0,0,1,0,0]).reshape(1, -1)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model.predict(x).argmax()


# In[ ]:





# In[ ]:




