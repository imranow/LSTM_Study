import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

rmse2 = 0

tr_filename="F:\\myprogram4\\NREL_Forecast\\Data\\train_NREL_solar_data.csv"
train_data = np.loadtxt(tr_filename,delimiter=',')

va_filename="F:\\myprogram4\\NREL_Forecast\\Data\\validate_NREL_solar_data.csv"
validate_data = np.loadtxt(tr_filename,delimiter=',')

te_filename="F:\\myprogram4\\NREL_Forecast\\Data\\test_NREL_solar_data.csv"
test_data = np.loadtxt(te_filename,delimiter=',')

x_tr  = train_data[:,0:9]
t_tr  = train_data[:,-1]

x_va  = validate_data[:,0:9]
t_va  = validate_data[:,-1]

x_te  = test_data[:,0:9]
t_te  = test_data[:,-1]

Ndays_tr = x_tr.shape[0]//11
Ndays_va = x_va.shape[0]//11
Ndays_te = x_te.shape[0]//11

train_x = x_tr.reshape(Ndays_tr,11,9) 
train_t = t_tr.reshape(Ndays_tr,11,1) 

validate_x = x_va.reshape(Ndays_va,11,9)
validate_t = t_va.reshape(Ndays_va,11,1) 
    
test_x = x_te.reshape(Ndays_te,11,9) 
test_t = t_te.reshape(Ndays_te,11,1) 

model = Sequential()
    #model.add(LSTM(30,input_shape=(11,9)))
    #model.add(Dense(9,activation='linear'))
    #model.add(LSTM(input_dim=9,output_dim=25,return_sequences=True))
model.add(LSTM(50,input_shape=(11,9),return_sequences=True))
model.add(Dense(1,activation='linear'))
model.compile(loss='mse',optimizer='adam')
    #history = model.fit(train_x,train_t,epochs=50,validation_data=(test_x,test_t))
history = model.fit(train_x,train_t,epochs=100,batch_size=50,validation_data=(validate_x,validate_t))
    
pyplot.plot(history.history['loss'],label='train')
    #pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()
pyplot.show()
    
yhat = model.predict(test_x)
y_te = yhat.reshape(Ndays_te*11,)
    
rmse2 += mean_squared_error(y_te,t_te)*Ndays_te*11

rmse = sqrt(rmse2/4026)*1087.4396/2
print('Test RMSE: %.3F' % rmse)

