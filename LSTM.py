from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, LSTM
from keras.layers import concatenate
import math
#import pandas_datareader as web
import numpy as np


# load dataset(4lu grafik cizer  1)
dataset = read_csv('XU100.csv', header=0, index_col=0)
data_ = dataset.head()
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4, 5]
i = 1
# plot each column
pyplot.figure(figsize=(16,8))
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
"""

"""
#(kapanisa gore tekli grafik cizer  2)
pyplot.figure(figsize=(16,8))
pyplot.title('Fiyat Kapanis Grafigi')
pyplot.plot(dataset['Close'])
pyplot.xlabel('Date',fontsize=18)
pyplot.ylabel('Kapanis Fiyatlari (ENDEKS)',fontsize=18)
pyplot.show()

# sadece kapanis fiyati sutununa gore veribirimi olustur
dataclose = dataset.filter(['Close'])
dataset_close = dataclose.values # numpy arraya donusturur
#egitilen data modeli icin satir sayilarini al
training_data_len = math.ceil( len(dataset_close) * .8)
print(training_data_len)

# veri olceklendir
# temel olarak olcekleme icin kullanilacak minimum ve maksimum degerleri hesaplar
# bu iki degere dayanarak verileri donusturuyorlar
# degerlerin 0 ile 1 arasinda olacagini soyleyin
# bu bir hucre
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset_close)
print(scaled_data)



# yeni bir egitim seti olusturacagiz
# bu hucrede tum egitim verileri setini olusturacagiz
# yapmak istedigimiz sey olcekli egitim veri setini olusturmak


train_data = scaled_data[0:training_data_len,:]#indeks 0 dan veri uzunluguna kadar geri dondurme
X_train = [] # sonraki tum sutunlari X trainine ve Y ye ayirir
Y_train = []

for i in range(60, len(train_data)):#60 aralikta egitim suresinin uzunlugu icin bir dongu olusturalim
    X_train.append(train_data[i-60:i,0])#son 60 degeri X egitim veri setimize ekleyelim x degeri 60 deger icerecek 0 konumundan 59 a kadar
    #ve ardindan Y kare train verisi altina endekslenir
    Y_train.append(train_data[i, 0]) # konumunda stun alacagiz bu yuzden ilk gecis icin ayarlanan 61. degeri 60. pozisyon simdi ne oldugunu gormek icin
    if i<=61 :
        print(" son 60 veri seti :\n{}".format(X_train))
        print(" 61. veri seti : \n {}" .format(Y_train))
    
    
 
# simdi hucrede yeni bir hucre olusturacagiz
# Numpy dizilerini LSTM modelini egitmek icin kullanabiliriz        
# X ve Y trainlerini numpy dizinisine donusturulmesi
X_train, Y_train = np.array(X_train), np.array(Y_train)
# bir LSTM agi. girdinin bir sayi biciminde 3 boyutlu olmasini bekler
# train veri seti 2 boyutludrur bu yuzden X tipi shape ile 2 boyutlu satir sayisini ve sutun sayisini gorebilecegiz
#print(X_train.shape)#satir ve sutun 2 boyut gosterir (1879,60
#LSTM 3 boyutlu istedigi icin 3 boyuta cevirecegiz
X_train = np.reshape(X_train , ( X_train.shape[0], X_train.shape[1], 1 ))
print(X_train.shape)


#LSTM modeli olusturulmasi
# LSTM katmani eklenmesi ve 50 noron eklenmesi input shape giris katmani 60 olan zaman adimlarinin ve 1 tekrar olan ozelliklerin sayisinin
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
#simdi ikinci katmanin ekleyelim 50 noronu olacak ve donus dizileri...
model.add(LSTM( 50, return_sequences = False ))
#birkac katman daha ekleyecegiz yogun bir katman ekleyecegiz bu katman 25 norona sahip olacak duzenli noron ag katmani
model.add(Dense(25))
# bir katman daha ekleyecegiz 1 norona sahip bir katman.
model.add(Dense(1))

# modelin derlenmesi
# modelimize optimize ediciyi vermeliyiz ve sonra kayip fonksiyonu vermeliyiz
model.compile(optimizer = 'adam' , loss='mean_squared_error')#modelin egitimde ne kadar iyi oldugunu olcmek icin kullanilir

#modelin egitilmesi#islem uzun suruyor epochlu kisim
model.fit(X_train, Y_train, batch_size=1, epochs=1)

#yeni bir hucre olusturacagiz ve bu hucrede test yapacagiz
#1879 dan 2003 dizisine olceklendirilmis degerler iceren yeni bir dizi olusturalim
#veri kumesinin sonu ve tum sutunlari geri alacagiz
test_data = scaled_data[training_data_len - 60: , :]
#Create data sets X_test and Y_test
X_test = []
Y_test = dataset_close[training_data_len:, :] #Y_testi, modelimizin tahmin etmesini istedigimiz tum degerler olacaktir

for i in range(60, len(test_data)): # X_test veri setine son 60 degeri ekleyecegiz
    X_test.append(test_data[i-60:i,0])
    
#yeni bir hucre olusturalim ve verileri numpy dizisine donusturelim
X_test = np.array(X_test)

# yeni hucre olusturalim ve bu hucrede verileri yeniden sekillendirelim
#verimiz 2 boyutlu LSTM 3 boyutlu istedigi icin 3 boyutluya cevirecegiz
X_test = np.reshape( X_test,(X_test.shape[0], X_test.shape[1], 1 ))#buda zaman adimi sayisina esit ve daha sonra sahip oldugumuz ozellikler sadece yakin fiyat iyi yani

#modellerin fiyat ve degerleri tahmin etmesini istemek icin
#X_train testi icin  modeldeki degerlerin tahmini fiyatini al 
predictions = model.predict(X_test) 
predictions = scaler.inverse_transform(predictions)# verileri tersine donusturun boylece tahminler yazilir
# Y_test verilerimizle ayni degerleri icerecek tahminler


#kok ortalama kare hatasi veya kisa surede RMSE alma
# modelin yaniti  ve RMSE dusuk degerleri ne kadar dogru tahmin ettigini ve kalintilarin standart sapmasi
# modelinizin ne kadar iyi bir performans gosterdigine dair gercekten bir fikir edinmek icin
#mse = np.sqrt(np.mean(predictions - Y_test )**2)
#mse = np.sqrt(np.mean(np.power((np.array(Y_test)-np.array(predictions)),2)))
rmse = np.sqrt(((predictions - Y_test) ** 2).mean())
print(rmse)# bu yuzde bes puan, kok ortalama kare hatasi icin sifir degeri tahminlerin dogru olmasi icin mukemmel oldugu anlamina gelir
print('Test RMSE: %.3f' % rmse)

#verileri ciz
train = dataclose[:training_data_len]
valid = dataclose[training_data_len:]
valid['Predictions'] = predictions

#verileri gorsellestirme

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Tarih',fontsize=18)
plt.ylabel('Kapanis Fiyatlari (ENDEKS)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()


# gercek ve ongorulen fiyat gosterme
print(valid)

   
       
