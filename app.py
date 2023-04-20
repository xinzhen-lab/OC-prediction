from flask import Flask, request, render_template
import numpy as np
import joblib
import threading
from scipy.special import boxcox1p
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import __main__

executor = ThreadPoolExecutor(3)
app = Flask(__name__)

class MCDF_CLF():
    def predict_and_prob(self, X):
        fusion_prob = np.zeros([X.shape[0], 2])
        for i in range(self.Fusion_num):
            temp_X_train = X[:, self.feaselect_id[self.fea_id[i], :]]
            exec('temp_prob' + '=' + 'self.basic_clf_' + str(i) + ' .predict_proba(temp_X_train)')
            exec('fusion_prob' + '+=' + 'temp_prob' + ' *' + 'self.Basic_clf_w[i]')
        fusion_pre = np.argmax(fusion_prob, 1)
        return fusion_pre, fusion_prob
    pass

class Data_processing():
    def transform(self, X, imp=True, boxcox=True, standered=True):
        if (imp == True) & (boxcox == True) & (standered == True):
            print('running')
            data_w4_pad = np.vstack([self.pad_data, X])
            imp = self.pad_method
            print(1)
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            # pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.max(self.pad_data, axis=0))] = np.max(self.pad_data, axis=0)[
                np.where(pad_X > np.max(self.pad_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.pad_data, axis=0))] = np.min(self.pad_data, axis=0)[
                np.where(pad_X < np.min(self.pad_data, axis=0))[1]]

            pad_X -= np.min(self.pad_data, axis=0)
            pad_X += 1
            for i in range(pad_X.shape[1]):
                pad_X[:, i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])

            pad_X = self.standard_para.transform(pad_X)

        if imp == True & (boxcox == False) & (standered == False):
            data_w4_pad = np.vstack([self.standard_data, X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            # pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.max(self.standard_data, axis=0))] = np.max(self.standard_data, axis=0)[
                np.where(pad_X > np.max(self.standard_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.standard_data, axis=0))] = np.min(self.standard_data, axis=0)[
                np.where(pad_X < np.min(self.standard_data, axis=0))[1]]

        return pad_X
    pass

__main__.MCDF_CLF = MCDF_CLF
__main__.Data_processing = Data_processing

def run():
    myclass = joblib.load('MCDM.model')
    data_proc = joblib.load('Processing.model')
    if os.path.exists('data.json'):
        load_f = open('data.json', 'r')
        data = json.load(load_f)
        load_f.close()
        if data['task'] == 'false':
            X = np.full([99], np.nan)
            X[myclass.input_data_head[0]] = data["CA125"]
            X[myclass.input_data_head[1]] = data["CA15_3"]
            X[myclass.input_data_head[2]] = data["CA72_4"]
            X[myclass.input_data_head[3]] = data["D_dimer"]
            X[myclass.input_data_head[4]] = data["CRP"]
            X[myclass.input_data_head[5]] = data["Age"]
            X[myclass.input_data_head[6]] = data["LYMPH"]
            X[myclass.input_data_head[7]] = data["FIB"]
            X[myclass.input_data_head[8]] = data["ALB"]
            X[myclass.input_data_head[9]] = data["HE_4"]
            X[myclass.input_data_head[10]] = data["LDH"]
            X[myclass.input_data_head[11]] = data["TTE"]
            X[myclass.input_data_head[12]] = data["FSH"]
            X[myclass.input_data_head[13]] = data["ESR"]
            X[myclass.input_data_head[14]] = data["PCT"]
            X[myclass.input_data_head[15]] = data["AG"]
            X[myclass.input_data_head[16]] = data["AFP"]
            X[myclass.input_data_head[17]] = data["HDL"]
            X[myclass.input_data_head[18]] = data["AT"]
            X[myclass.input_data_head[19]] = data["NEUT"]

            print(myclass.input_data_head)
            print(X)
            X_precessed = data_proc.transform(np.array([X]) ,imp=True, boxcox=True, standered=True)

            y_pre, y_prob = myclass.predict_and_prob(X_precessed)

            data['Negative probability'] = str(round(y_prob[0][0]*100,1))
            data['Positive probability'] = str(round(y_prob[0][1]*100,1))
            data['task'] = 'true'
            load_f = open('data.json', 'w')
            json.dump(data, load_f)
            load_f.close()


@app.route('/')
def home():
    if os.path.exists('data.json'):
        os.remove('data.json')
    return render_template('index1.html')

@app.route('/getdelay/getresult',methods=['POST','GET'])
def getresult():
    time1 = time.localtime(time.time())
    time1 = time.mktime(time1)
    str1 = ''
    if (time1%3==0):
        str1='.'
    if(time1%3==1):
        str1 = '..'
    if (time1 % 3 == 2):
        str1 = '...'
    if os.path.exists('data.json'):
        load_f = open('data.json', 'r')
        data = json.load(load_f)
        load_f.close()
        if data['task'] == 'true':
            str2= "<p>Negative probability   :   "+data["Negative probability"] + '%'+'<p></p>'+ "Positive probability   :   "+data["Positive probability"]+ '%</p>'
            return str2
        else:
            return " Running" + str1


@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result = request.form
        Age = float(result['Age'])
        D_dimer = float(result['D-dimer'])
        CA125 = float(result['CA125'])
        HE_4 = float(result['HE-4'])
        CA15_3 = float(result['CA15-3'])
        FIB = float(result['FIB'])
        LYMPH = float(result['LYMPH'])
        CRP = float(result['CRP'])
        ALB = float(result['ALB'])
        CA72_4 = float(result['CA72-4'])
        LDH = float(result['LDH'])
        TTE = float(result['TTE'])
        FSH = float(result['FSH'])
        ESR = float(result['ESR'])
        PCT = float(result['PCT'])
        AG = float(result['AG'])
        AFP = float(result['AFP'])
        HDL = float(result['HDL'])
        AT = float(result['AT'])
        NEUT = float(result['NEUT'])
        data = {
            'Age': Age,
            'D_dimer': D_dimer,
            'CA125': CA125,
            'HE_4': HE_4,
            'CA15_3': CA15_3,
            'FIB': FIB,
            'LYMPH': LYMPH,
            'CRP': CRP,
            'ALB': ALB,
            'CA72_4': CA72_4,
            'LDH': LDH,
            'TTE': TTE,
            'FSH': FSH,
            'ESR': ESR,
            'PCT': PCT,
            'AG': AG,
            'AFP': AFP,
            'HDL': HDL,
            'AT': AT,
            'NEUT': NEUT,
            'task': 'false',
            'Negative probability': '0',
            'Positive probability': '0'
        }
        load_f = open('data.json', 'w')
        json.dump(data, load_f)
        load_f.close()
        executor.submit(run)
        '''X = np.full([99], 0.0)
        X[0] = float(Age)
        X[1] = float(D_dimer)
        X[2] = float(CA125)
        X[3] = float(HE_4)
        X[4] = float(LDH)
        X[5] = float(FIB)
        X[6] = float(AT)
        X[7] = float(HDL)
        X[8] = float(ALB)
        X[9] = float(A_G)
        X_new = X[myclass.input_data_head]
        y_pre, y_prob = myclass.predict_and_prob(np.array([X_new]))'''
        return render_template('index2.html',data1 = Age,
                               data2 = D_dimer, data3 = CA125,data4 = HE_4,data5 = CA15_3,data6 = FIB,data7 = LYMPH,
                               data8 = CRP,data9 = ALB,data10 = CA72_4,data11 = LDH,
                               data12 = TTE, data13 = FSH,data14 = ESR,data15 = PCT,data16 = AG,data17 = AFP,
                               data18 = HDL,data19 = AT,data20 = NEUT)
if __name__ == '__main__':
    app.debug = True
    app.run()