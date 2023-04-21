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

def test():
    myclass = joblib.load('MCDM.model')
    data_proc = joblib.load('Processing.model')
    load_f = open('data.json', 'r')
    data = json.load(load_f)
    load_f.close()
    data['task'] = 'false'

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
        X_precessed = data_proc.transform(np.array([X]), imp=True, boxcox=True, standered=True)

        y_pre, y_prob = myclass.predict_and_prob(X_precessed)


        data['Negative probability'] = str(round(y_prob[0][0] * 100, 1))
        data['Positive probability'] = str(round(y_prob[0][1] * 100, 1))
        print('Negative probability: ' + data['Negative probability']+'%')
        print('Positive probability: ' + data['Positive probability']+'%')
        #data['task'] = 'true'
        load_f = open('data.json', 'w')
        json.dump(data, load_f)
        load_f.close()

if __name__ == '__main__':
    test()