import numpy as np
import joblib
from scipy.special import boxcox1p
class MCDF_CLF():
    def predict_and_prob(self, X):
        fusion_prob = np.zeros([X.shape[0], 2])
        for i in range(self.Fusion_num):
            temp_X_train = X[:, self.feaselect_id[self.fea_id[i], :]]
            exec('temp_prob' + '=' + 'self.basic_clf_' + str(i) + ' .predict_proba(temp_X_train)')
            exec('fusion_prob' + '+=' + 'temp_prob' + ' *' + 'self.Basic_clf_w[i]')
        fusion_pre = np.argmax(fusion_prob, 1)
        return fusion_pre, fusion_prob

class Data_processing():
    def transform(self,X,imp=True, boxcox=True, standered =True):
        if imp==True:
            data_w4_pad = np.vstack([self.pad_data,X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:,:]
            pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.max(self.pad_data, axis=0))] = np.max(self.pad_data, axis=0)[
                np.where(pad_X > np.max(self.pad_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.min(self.pad_data, axis=0))] = np.min(self.pad_data, axis=0)[
                np.where(pad_X < np.min(self.pad_data, axis=0))[1]]
        if boxcox==True:
            pad_X -= np.min(self.pad_data, axis=0)
            pad_X += 1
            for i in range(pad_X.shape[1]):
                pad_X[:,i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])

        if standered==True:
            pad_X = self.standard_para.transform(pad_X)
        return pad_X

def pridict(a):
    myclass = joblib.load('MCDM.model')
    data_proc = joblib.load('Processing.model')
    X = np.full([99], np.nan)
    for i in range(10):
        X[i] = a[i]
    X_new = X[myclass.input_data_head]
    X_precessed = data_proc.transform(np.array([X_new]))
    y_pre,y_prob = myclass.predict_and_prob(X_precessed)
    return y_pre, y_prob