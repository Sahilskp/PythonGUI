import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import popup
import numpy as np
import pandas as pd
from scipy.stats.stats import NormaltestResult
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC




def gui():
    global DF 
    DF = pd.read_csv("TelcomCustomer-Churn.csv")
    layout = [[sg.Text('Gender', size=(12,1)), sg.Combo(['Male', 'Female'], size=(10,10))],
            [sg.Text('Senior Citizen', size=(12,1)), sg.Combo([1, 0], size=(10,10))],
            [sg.Text('Partner', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('Dependents', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('tenure', size=(12,1)), sg.InputText(size=(10, 1), key='TEN')],
            [sg.Text('PhoneService', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('MultipleLines', size=(12,1)), sg.Combo(['Yes', 'No', 'No phone service'], size=(10,10))],
            [sg.Text('Internet Service', size=(12,1)), sg.Combo(['DSL', 'Fibre optic', 'No'], size=(10,10), key='IS')],
            [sg.Text('OnlineSecurity', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('OnlineBackup', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('DeviceProtection', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('TechSupport', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('StreamingTV', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('StreamingMovies', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('Contract', size=(12,1)), sg.Combo(['Month-to-month', 'One year', 'Two year'], size=(12,10))],
            [sg.Text('PaperlessBilling', size=(12,1)), sg.Combo(['Yes', 'No'], size=(10,10))],
            [sg.Text('PaymentMethod', size=(12,1)), sg.Combo(['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'], size=(10,10), key='PM')],
            [sg.Text('MonthlyCharges', size=(12,1)), sg.InputText(size=(10, 1), key='MC')],
            [sg.Text('TotalCharges', size=(12,1)), sg.InputText(size=(10,1), key='TC')],
            [sg.Button('Close'), sg.Button('Add'), sg.Button('Predict')]]


    window = sg.Window('Title', layout, margins=(100, 100))
    mod = TrainingModel(DF,BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, n_estimators=20, max_features=1.0))
    mod.preProcessing()
    mod.split_data()
    mod.training()

    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        data = ['']
        if type(values) == dict:
            if '' not in values.values():
                for x in values:
                    if x == 'TC':
                        data.append(int(values[x]))
                    elif x == 'MC':
                        data.append(int(values[x]))
                    elif x == 'TEN':
                        data.append(int(values[x]))
                    elif x == 'IS':
                        data.append(values[x])
                    elif x == 'PM':
                        data.append(values[x])
                    else:
                        data.append(values[x])

        if event == "Add":
            data.append('No')
            DF.loc[len(DF.index)] = data
            mod.preProcessing()
            if int(mod.predict(mod.processedData.iloc[len(mod.processedData.index)-1, :23].values)[0]) == 0:
                sg.popup('Unsubscribed')
            else:
                sg.popup('Subscribed')
            mod.training()
            
        if event == 'Predict':
            data.append('No')
            DF.loc[len(DF.index)] = data
            mod.preProcessing()
            if int(mod.predict(mod.processedData.iloc[len(mod.processedData.index)-1, :23].values)[0]) == 0:
                sg.popup('Unsubscribed')
            else:
                sg.popup('Subscribed')
            DF = DF.drop(len(DF.index)-1)

        
        if event == "Close" or event == sg.WIN_CLOSED:
            window.close()
            break


def convert_data(val, mod):
    vals = np.array(val).reshape([1,20])
    cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    dd = pd.DataFrame(vals, index=[0], columns=cols)
    mod.preProcessing(dd, True)
    return mod.processedDataRow




class TrainingModel():
    def __init__(self, data, model):
        self.rawData = data
        self.model = model
        self.processedData = None
        self.processedDataRow = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelScore = None 

    def preProcessing(self, singleRow=0, oneRow=False):
        if oneRow == True :
            df = singleRow
        else:
            df = self.rawData

        df = df.drop('customerID', axis = 1)

        b_columns = df.columns[[0,2,3,5,6,8,9,10,11,12,13,15,19]]
        for x in b_columns:
            if x == 'gender':
                df[x].replace(('Male', 'Female'), (1, 0), inplace=True)
            elif df[x].value_counts().index.size > 2:
                df[x].replace(('Yes', 'No', df[x].value_counts().index[2]), (1, 0, 0), inplace=True)
            elif df[x].value_counts().index.size < 3:
                df[x].replace(('Yes', 'No'), (1, 0), inplace=True)
        
        is_dummy = pd.get_dummies(df['InternetService']).drop(['No'], axis=1)
        contract_dummy = pd.get_dummies(df['Contract']).drop(['Two year'], axis=1)
        paymentMethod_dummy = pd.get_dummies(df['PaymentMethod']).drop(['Mailed check'], axis=1)
        x = pd.concat([is_dummy, contract_dummy, paymentMethod_dummy], axis=1)

        temp = df.drop(['InternetService', 'Contract', 'PaymentMethod'], axis=1)
        temp1 = pd.concat([temp, x], axis=1)
        #Removing empty values in TotalCharges
        ind = temp1.loc[temp1['TotalCharges'] == ' '].index
        temp1.loc[ind, 'TotalCharges'] = 0

        #converting to float
        temp1['TotalCharges'].astype('float')
        if oneRow == True:
            self.processedDataRow = temp1    
            return self.processedDataRow
        else:
            self.processedData = temp1


        

    def split_data(self):
        temp1 = self.targetBalancing()
        X = temp1.drop(['Churn'], axis= 1)
        y = temp1['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        X_train, X_test, y_train, y_test

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def score(self):
        self.modelScore = self.model.score(self.X_test, self.y_test)

    def predict(self, d):
        return self.model.predict([d])

    def targetBalancing(self):
        temp1 = self.processedData
        dataf = temp1.loc[temp1['Churn'] == 0]
        dataf = dataf.sample(n=2000)

        comb_data = pd.concat([dataf, temp1.loc[temp1['Churn'] == 1] ])

        return comb_data



#print(mod.predict())
#print(mod.modelScore)


gui()