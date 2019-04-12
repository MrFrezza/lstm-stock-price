from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import lstm, time, csv, os

# Separando a informação desejada no CSV
def prepara_arquivo(arquivo):
    tempp = []

    with open(arquivo, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        arq = open('ex1.csv', 'w')

        next(csvFileReader)
        for row in csvFileReader:
            tempp.append(row[4] + '\n')
        
        arq.writelines(tempp)
        arq.close()

def carrega_model(nome):
    existe = False
    if os.path.isfile(nome):
        model = load_model(nome)
        existe = True
    else:
        model = Sequential()
    
    return model, existe


def treina_modelo(model, x_train, y_train, existe=False):
    if existe == False:
        model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))


        model.add(Dense(20, activation='relu'))
        model.add(Dense(output_dim=1))
        model.add(Activation('linear'))

        start = time.time()

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print("Compilation time: ", time.time() - start)


    # # Para quando não melhorar
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [es]

    # Treinando o modelo
    model.fit(x_train, y_train,
                batch_size=512,
                epochs=10,
                validation_split=0.05,
                callbacks=callbacks_list,
                verbose=1)
    
    model.save('meu_modelo.h5')

    #Step 4 - Plot the predictions!
    predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
    lstm.plot_results_multiple(predictions, y_test, 50)




# Carrega os dado
prepara_arquivo('apple_00_19.csv')

# Divide o dataset
x_train, y_train, x_test, y_test = lstm.load_data('ex1.csv', 50, True)

# Constroi/Carrega o modelo
model, existe = carrega_model('meu_modelo.h5')

print("exi: " + str(existe))
treina_modelo(model, x_train, y_train, existe)
