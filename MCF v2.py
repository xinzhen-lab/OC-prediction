from __future__ import division, print_function
import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, hamming_loss, f1_score, recall_score, precision_score
import copy
import math
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import trace_ratio
from skfeature.function.sparse_learning_based import ll_l21
from skfeature.function.sparse_learning_based import ls_l21
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import chi_square
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import t_score
from skfeature.function.streaming import alpha_investing
from skfeature.function.wrapper import decision_tree_backward
from skfeature.function.wrapper import decision_tree_forward
from skfeature.function.wrapper import svm_backward
from skfeature.function.wrapper import svm_forward
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking, construct_label_matrix_pan
import joblib
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, GridSearchCV, \
    RandomizedSearchCV
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import matplotlib.pyplot as plt
# import kaleido


class Data_processing():
    def __init__(self,
                 padding_method = {# 'MEAN' : SimpleImputer(missing_values=np.nan, strategy='mean')
                                   # 'KNN' : KNNImputer(n_neighbors=5, weights="uniform")
                                   'MFI' : IterativeImputer(max_iter=30, random_state=0)},


                 ):
        self.pad_method = padding_method[list(padding_method.keys())[-1]]


        self.boxcox_para = None
        self.datapad_pro = False
        self.standard_para = None
        self.int_list = [15, 20, 25, 28, 29, 31, 33, 39, 57, 58, 59, 61, 62, 63, 67, 68, 69, 90, 92]
        self.scale_list = [16, 57, 68, 69, 61, 62, 63, 67, 68, 69, 89]


    def fit(self, X, int_stransfer=True):
        self.org_data = X
        self.fea_num = X.shape[1]
        self.cont_list = [i for i in range(self.fea_num) if i not in self.scale_list]
        if self.datapad_pro == False:
            if not os.path.exists('C:/Users/user/Desktop/图表 4/DP/pad_data.npy'):
                self.datapadding(X, int_stransfer=int_stransfer)
            else:
                self.pad_data = np.load('C:/Users/user/Desktop/图表 4/DP/pad_data.npy',
                                        allow_pickle=True)


        if self.boxcox_para == None:
            if not os.path.exists('C:/Users/user/Desktop/图表 4/DP/boxcox_para.npy'):
                self.get_boxcox_para(self.pad_data, int_stransfer=int_stransfer)
            else:
                self.boxcox_para = np.load('C:/Users/user/Desktop/图表 4/DP/boxcox_para.npy',
                                           allow_pickle=True)
                self.boxcox_data = np.load('C:/Users/user/Desktop/图表 4/DP/boxcox_data.npy',
                                           allow_pickle=True)



        if self.standard_para == None:
            if not os.path.exists('C:/Users/user/Desktop/图表 4/DP/standard_para.model'):
                self.get_standard_para(self.boxcox_data)
            else:
                self.standard_para = joblib.load(
                    'C:/Users/user/Desktop/图表 4/DP/standard_para.model')
                self.standard_data = np.load( 'C:/Users/user/Desktop/图表 4/DP/standarlized_data.npy',allow_pickle=True)

        self.pad_data_linespacMatrix = np.array([np.linspace(np.nanmin(self.pad_data[:,i]),np.nanmax(self.pad_data[:,i]),1000) for i in range(self.pad_data.shape[1])]).T
        self.standard_data_linespacMatrix = self.transform(self.pad_data_linespacMatrix)
        return self

    def datapadding(self, X, int_stransfer=True):
        save_path = 'C:/Users/user/Desktop/图表 4/DP/'

        imp = self.pad_method
        X_pad = imp.fit_transform(X)
        if int_stransfer:
            X_pad[:, self.int_list] = np.round(X_pad[:, self.int_list])
        for i in range(X.shape[1]):
            if len(np.where(self.org_data[i, :] == np.nan)) < len(self.org_data[i, :]):
                temp_max = np.nanmax(self.org_data[:, i])
                X_pad[np.where(X_pad[:, i] > temp_max), i] = temp_max
                temp_min = np.nanmin(self.org_data[:, i])
                X_pad[np.where(X_pad[:, i] < temp_min), i] = temp_min
        X_pad[np.where(X_pad < 0)] = 0
        self.pad_data = X_pad
        np.save(save_path + 'pad_data.npy', X_pad)

        self.datapad_pro = True
        return self

    def get_boxcox_para(self, X, int_stransfer=True):
        save_path = 'C:/Users/user/Desktop/图表 4/DP/'
        self.boxcox_para = np.empty([1, X.shape[1]])

        X_copy =copy.deepcopy(X)
        if int_stransfer:
            for i in range(X.shape[1]):
                if i not in self.scale_list:
                    temp_array = X[:, i]
                    temp_array -= min(temp_array)
                    temp_array += 1
                    temp_para = boxcox_normmax(temp_array)
                    X_copy[:, i] = boxcox1p(temp_array, boxcox_normmax(temp_array))
                    self.boxcox_para[0, i] = temp_para
                else:
                    self.boxcox_para[0, i] = np.nan
        else:

            for i in range(X.shape[1]):

                temp_array = X[:, i]
                temp_array -= min(temp_array)
                temp_array += 1
                temp_para = boxcox_normmax(temp_array)
                X_copy[:, i] = boxcox1p(temp_array, boxcox_normmax(temp_array))
                self.boxcox_para[0, i] = temp_para
        self.boxcox_data = X_copy

        np.save(save_path + 'boxcox_data.npy', self.boxcox_data)
        np.save(save_path + 'boxcox_para.npy',self.boxcox_para)
        return self

    def get_standard_para(self,X, int_stransfer=True):
        save_path = 'C:/Users/user/Desktop/图表 4/DP/'
        self.standard_data = copy.deepcopy(self.pad_data)
        # standard_para = preprocessing.StandardScaler().fit(X)
        if int_stransfer:
            self.standard_para = preprocessing.StandardScaler().fit(X[:, self.cont_list])
            self.standard_data[:, self.cont_list] = self.standard_para.transform(X[:, self.cont_list])
        else:
            self.standard_para = preprocessing.StandardScaler().fit(X)
            self.standard_data = self.standard_para.transform(X)


        joblib.dump(self.standard_para, save_path+'standard_para.model')
        np.save(save_path+'standarlized_data.npy',self.standard_data)
        return self


    def transform(self,X,imp=True, boxcox=True, standered =True,int_stransfer=True):

        if (imp == True) & (boxcox == True) & (standered == True):
            data_w4_pad = np.vstack([self.pad_data, X])
            imp = self.pad_method
            if len(X.shape) == 1:
                pad_X = imp.fit_transform(data_w4_pad)[-1:, :]
            else:
                pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            if int_stransfer:
                pad_X[:, self.int_list] = np.round(pad_X[:, self.int_list])
            # pad_X[np.where(pad_X < 0)] = 0

            #BOXCOX
            if int_stransfer:
                for i in range(pad_X.shape[1]):
                    if i not in self.scale_list:
                        temp_array = pad_X[:, i]
                        temp_array -= min(temp_array)
                        temp_array += 1
                        pad_X[:, i] = boxcox1p(temp_array, self.boxcox_para[0, i])

            else:
                for i in range(X.shape[1]):
                    temp_array = pad_X[:, i]
                    temp_array -= min(temp_array)
                    temp_array += 1
                    pad_X[:, i] = boxcox1p(temp_array, self.boxcox_para[0, i])
            #
            # if int_stransfer:
            #     for i in range(pad_X.shape[1]):
            #         if i not in self.scale_list:
            #             pad_X[:, i] -= np.min(self.pad_data[:, i], axis=0)
            #             pad_X[:, i] += 1
            #             pad_X[:, i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])
            # else:
            #     pad_X -= np.min(self.pad_data, axis=0)
            #     pad_X += 1
            #     for i in range(pad_X.shape[1]):
            #         pad_X[:, i] = boxcox1p(pad_X[:, i], self.boxcox_para[0, i])

            if int_stransfer:
                pad_X[:, self.cont_list] = self.standard_para.transform(pad_X[:, self.cont_list])
            else:
                pad_X = self.standard_para.transform(pad_X)



        if imp==True & (boxcox==False) & (standered==False):
            data_w4_pad = np.vstack([self.pad_data, X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]
            if int_stransfer:
                pad_X[:, self.int_list] = np.round(pad_X[:, self.int_list])
            # pad_X[np.where(pad_X < 0)] = 0
            pad_X[np.where(pad_X > np.nanmax(self.org_data, axis=0))] = np.nanmax(self.org_data, axis=0)[
                np.where(pad_X > np.nanmax(self.org_data, axis=0))[1]]
            pad_X[np.where(pad_X < np.nanmin(self.org_data, axis=0))] = np.nanmin(self.org_data, axis=0)[
                np.where(pad_X < np.nanmin(self.org_data, axis=0))[1]]

        if imp == True & (boxcox == False) & (standered == True):
            data_w4_pad = np.vstack([self.standard_data, X])
            imp = self.pad_method
            pad_X = imp.fit_transform(data_w4_pad)[-X.shape[0]:, :]

            if int_stransfer:
                pad_X[:, self.scale_list] = np.round(pad_X[:, self.scale_list])

            # pad_X[np.where(pad_X < 0)] = 0
            # pad_X[np.where(pad_X > np.nanmax(self.standard_data, axis=0))] = np.nanmax(self.standard_data, axis=0)[
            #     np.where(pad_X > np.nanmax(self.standard_data, axis=0))[1]]
            # pad_X[np.where(pad_X < np.nanmin(self.standard_data, axis=0))] = np.nanmin(self.standard_data, axis=0)[
            #     np.where(pad_X < np.nanmin(self.standard_data, axis=0))[1]]

        return pad_X

    def vers_transform(self, X, boxcox=True, standered =True, int_stransfer=True, strategy ='F'):
        """"F: Fitting, to construct a linespace array to match the nearest transformed value
            C: Conputed, to calculate the inverse-transformed value of transformed data """

        if strategy == 'C':
            if int_stransfer == True:
                if standered == True:
                    X[:, self.cont_list] = self.standard_para.inverse_transform(X[:, self.cont_list])
                if boxcox == True:
                    for i in self.cont_list:
                        if self.boxcox_para[0, i] != 0:
                            X[:, i] = pow(X[:, i] * self.boxcox_para[0, i], 1 / (self.boxcox_para[0, i])) + np.nanmin(
                                self.pad_data[:, i])
                        else:
                            X[:, i] = np.exp(X[:, i]) + np.nanmin(self.pad_data[:, i])
            else:
                if standered == True:
                    X = self.standard_para.inverse_transform(X)
                if boxcox == True:
                    for i in range(X.shape[1]):
                        if self.boxcox_para[0, i] != 0:
                            X[:, i] = pow(X[:, i] * self.boxcox_para[0, i], 1 / (self.boxcox_para[0, i])) + np.nanmin(
                                self.pad_data[:, i])
                        else:
                            X[:, i] = np.exp(X[:, i]) + np.nanmin(self.pad_data[:, i])
        else:
            if int_stransfer == True:
                if len(X.shape) == 1:
                    X[self.cont_list] = [self.pad_data_linespacMatrix[len(np.where(X[:,i]>self.standard_data_linespacMatrix[:,i])[0]),i] for i in self.cont_list]
                else:
                    for j in range(X.shape[0]):
                        X[j,self.cont_list] = [self.pad_data_linespacMatrix[len(np.where(X[:,i]>self.standard_data_linespacMatrix[:,i])[0]),i] for i in self.cont_list]
            else:
                if len(X.shape) == 1:
                    X = [
                        self.pad_data_linespacMatrix[len(np.where(X[:, i] > self.standard_data_linespacMatrix[:, i])[0]), i]
                        for i in range(X.shape[0])]
                else:
                    for j in range(X.shape[0]):
                        X[j, :] = [self.pad_data_linespacMatrix[len(
                            np.where(X[:, i] > self.standard_data_linespacMatrix[:, i])[0]), i] for i in range(X.shape[0])]

            d=1
        return X


class MCDF_CLF():
    def __init__(self,
                 FeatureSelector=[
                     "alpha_investing",
                     "DISR",  # 3
                     "f_score",  # 23
                     "fisher_score",  # 10
                     "gini_index",  # 24
                     "ICAP",  # 5
                     "JMI",  # 6

                     "lap_score",  # 11
                     "ll_l21",  # 15a
                     "ls_l21",  # 16
                     "MCFS",  # 17

                     "NDFS",  # 18

                     "reliefF",  # 12
                     #        "SPEC",  # 13

                     "t_score",  # 25
                     "trace_ratio",  # 14

                     #          "RFS",  # 19  #slow
                     "UDFS",  # 20
                     #     "CFS",  # 21  #slow
                     #     "chi_square",  # 22

                     #     "decision_tree_backward",  # 27
                     #     "decision_tree_forward",  # 28
                     #     "svm_backward",  # 29
                     #     "svm_forward"  # 30
                 ],
                 classifiers={
                     "Logistic Regression": LogisticRegression(penalty='l2', random_state=None),
                     "KNN": KNeighborsClassifier(),
                     "SVM": SVC(probability=True, C=1),
                     # "Naive Bayes": GaussianNB(),
                     "Decision Tree": DecisionTreeClassifier(),
                     "Extra Trees": ExtraTreesClassifier(random_state=None),
                     "Random Forest": RandomForestClassifier(n_estimators=50),
                     # "Bagging": BaggingClassifier(),
                     "AdaBoost": AdaBoostClassifier(),
                     "GradientBoosting": GradientBoostingClassifier(),
                     "XGBoost": XGBClassifier(),
                     "LightGBM": LGBMClassifier(),
                     "Catboost": CatBoostClassifier(logging_level='Silent'),

                     # "LDA": LinearDiscriminantAnalysis(),
                     # "QDA": QuadraticDiscriminantAnalysis(),
                     # "MLP": MLPClassifier()
                 },
                 Fusion_num = 20,
                 feature_num = 20,
                 class_weight=None,
                 nfold = 5
                 ):
        self.classifiers = classifiers
        self.FeatureSelector = FeatureSelector
        self.Fusion_num = Fusion_num
        self.feature_num = feature_num
        self.base = False
        self.heatmapdata = None
        self.feaselect_id = False

        self.perf_matrix = None
        self.Basic_clf_w = None

        #

        input_id_file = 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/test/feature name.xlsx'
        inp_model_id_file = 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/test/standerelized.csv'
        head_name = pd.read_excel(input_id_file, engine='openpyxl', encoding='gbk',
                                  usecols=[0, 3])
        data_head = pd.read_csv(inp_model_id_file, encoding='gbk')
        data_head = np.array(list(data_head.head(0))[2:])

        temp_head = np.full([len(data_head)],np.nan)

        for i in range(head_name.shape[0]):
            if i < 9:
                head_name[0][i] = head_name[0][i][2:]
            else:
                head_name[0][i] = head_name[0][i][3:]
        for i in range(head_name.shape[0]):
            for j in range(len(data_head)):
                if data_head[j] == head_name[0][i]:
                    temp_head[i] = j


        self.input_data_head = temp_head.astype(int)

    def fit(self, X, y, sample_weight=1):
        # fusion step: training basic clf, to get perf_matix and w
        if self.feaselect_id is False:
            self.feature_selection(X, y)

        if self.base  == False:
            self.basic_clf_train(X, y, sample_weight=None)



        self.fea_id, self.clf_id = self.chose_sorted(self.heatmapdata[0], self.Fusion_num)



        self.perf_matrix = np.zeros([7,self.Fusion_num])

        ad = ADASYN(ratio=float(sample_weight))  # kind = ['regular', 'borderline1', 'borderline2', 'svm']
        X_resampled, y_resampled = ad.fit_sample(X, y)

        for i in range(self.Fusion_num):
            # print(i)
            temp_X_train = X_resampled[:, self.feaselect_id[self.fea_id[i], :]]

            #creat basic clf
            temp_clf = GridSearchCV(self.classifiers[list(self.classifiers.keys())[self.clf_id[i]]], cv=3, param_grid={})
            temp_clf.fit(temp_X_train, y_resampled)
            exec('self.basic_clf_' + str(i) + ' = ' + 'temp_clf')

            # exec('self.basic_clf_' + str(i) + ' = '+'self.classifiers[list(self.classifiers.keys())[self.clf_id[i]]]')
            # exec('self.basic_clf_' + str(i) + ' .fit(temp_X_train,y_resampled)')

            # cal Decision Matrix
            # exec('temp_pre' + '=' + 'self.basic_clf_' + str(i) + ' .predict(temp_X_train)')
            # exec('temp_prob' + '=' + 'self.basic_clf_' + str(i) + ' .predict_proba(temp_X_train)')
            # exec('self.perf_matrix[:,i]'+'='+'np.array(get_perf_Matrix(y,temp_pre,temp_prob)).T')
            exec('self.perf_matrix[:,i]' + '=' + 'self.heatmapdata[:,self.fea_id[i], self.clf_id[i]]')


        self.Basic_clf_w = self.MCDM_func(self.perf_matrix.T,np.ones([7,1])).astype(float)

        return self

    def predict_and_prob(self,X):
        # if self.Basic_clf_w == None:
        #     raise SyntaxError('Please Trainning Model First')
        fusion_prob = np.zeros([X.shape[0],2])
        for i in range(self.Fusion_num):
            temp_X_train = X[:, self.feaselect_id[self.fea_id[i], :]]
            exec('temp_prob' + '=' + 'self.basic_clf_' + str(i) + ' .predict_proba(temp_X_train)')
            exec('fusion_prob' + '+=' + 'temp_prob'+ ' *'+ 'self.Basic_clf_w[i]')
        fusion_pre = np.argmax(fusion_prob,1)
        return fusion_pre,fusion_prob

    def save(self,saving_path = 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/'):
        joblib.dump(self,saving_path + 'MCDM.model')
        return 0


    def basic_clf_train(self, X, y, sample_weight=None):
        # to get AUC heatmap from all basic clf + feature_slct combinations
        save_path = 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/Heatmap_data/'
        data_name = 'heatmapdata_average_20.npy'
        if not os.path.exists(save_path+data_name):

            self.heatmapdata = np.zeros([len(self.FeatureSelector),len(self.classifiers),5])
            temp_fold = 0


            skf = StratifiedKFold(n_splits=self.nfold, random_state=1, shuffle=True)
            for train_id, test_id in skf.split(X, y):
                temp_fold+=1
                X_train = X[train_id, :]
                y_train = y[train_id]
                X_test = X[test_id, :]
                y_test = y[test_id]
                for i,name2 in zip(range(self.feaselect_id.shape[0]), self.FeatureSelector):
                    idx = self.feaselect_id[i, :]
                    j = 0
                    for name1, clf in self.classifiers.items():
                        AUC, ACC, SEN, SPE = self.heat_map_auc(name1, idx, X_train, y_train, X_test, y_test,
                                                               clf,
                                                               feanum=self.feature_num, FLAG_DataBalance=1)
                        self.heatmapdata[:, i, j, temp_fold] = np.array([AUC, ACC, SEN, SPE])
                        j+=1
            self.heatmapdata = np.mean(self.heatmapdata, axis=2)
            d=1  #wip
            self.base = True
            # self.heatmapdata = heatmapdata
        else:
            self.base = True
            self.heatmapdata = np.load(save_path+data_name,allow_pickle=True)
        return self

    def feature_selection(self, X, y):
        # to get AUC heatmap from all basic clf + feature_slct combinations
        # save_path = 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/feature_selected/'
        save_path = 'F:/SUBJECT/ZHONG/EOC/fea_selected index/mul_center_drop_pad/boxcox_standerlized v3_new/TJ/'

        if not os.path.exists(save_path):
            for name2 in self.FeatureSelector:
                middle_save_path = save_path + '/' + name2 + '/'
                if not os.path.exists(middle_save_path):
                    idx = self.featureid_saving(name2 + "_main", X, y, feanum=self.feature_num)
                    os.makedirs(middle_save_path)
                    np.save(middle_save_path + 'idx.npy', idx)

        else:
            file_name = os.listdir(save_path)
            self.feaselect_id = np.zeros([len(self.FeatureSelector),self.feature_num])
            if len(self.FeatureSelector) != len(file_name):
                raise SyntaxError('Feature selecting saving error')

        for i in range(len(file_name)):
            self.feaselect_id[i,:] = np.load(save_path+file_name[i]+'/idx.npy',allow_pickle=True)[:self.feature_num]
        self.feaselect_id = self.feaselect_id.astype(int)



        return self

    def MCDM_func(self, dm, weights):
        '''输入一个attribute * criteria 的矩阵'''
        M = dm ** 2
        decison_metrix = dm.copy()
        A = False
        if dm.shape[0] < 3:
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] / math.sqrt(sum(M[:, i]))
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] * weights[i]

            al_weights = np.sum(decison_metrix, axis=1) / sum(sum(decison_metrix))
        else:
            '''TOPSIS_tp'''
            '''regular the decision metrix'''
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] / float(math.sqrt(sum(M[:, i])))

            '''regular the weights'''
            '''calcute the weighted normalised decisoin metrix'''
            for i in range(decison_metrix.shape[1]):
                decison_metrix[:, i] = decison_metrix[:, i] * weights[i]
            '''Determine the worst alternative and the best alternative'''
            best = np.zeros((1, decison_metrix.shape[1]))
            worst = np.zeros((1, decison_metrix.shape[1]))
            for i in range(decison_metrix.shape[1]):
                best[0, i] = max(decison_metrix[:, i])
                worst[0, i] = min(decison_metrix[:, i])
            '''Calculate the L2-distance between the target alternative and the worst condition'''
            Sw = np.zeros((decison_metrix.shape[0], 1))
            for i in range(decison_metrix.shape[0]):
                Sw[i] = math.sqrt(sum(sum((decison_metrix[i, :] - worst) ** 2)))
                if Sw[i] == 0:
                    Sw[i] = 0.1
            '''Calculate the L2-distance between the target alternative and the best condition'''
            Sb = np.zeros((decison_metrix.shape[0], 1))
            for i in range(decison_metrix.shape[0]):
                Sb[i] = math.sqrt(sum(sum((decison_metrix[i, :] - best) ** 2)))
            S = Sw + Sb
            al_weights = Sw / S

        classifier_w = al_weights / np.sum(al_weights)
        # y_prediction = np.zeros((n_sample, 1))
        # y_proba_final = np.zeros((n_sample, len(cls)))
        #
        # threshold_weights = [1, c1_weights]
        # for i in range(n_sample):
        #     y_proba_temp = y_proba_matrix[i]
        #     y_proba_final[i, :] = np.dot(classifier_w.T, y_proba_temp) * threshold_weights
        #     # temp = np.dot(classifier_w.T, y_proba_temp)
        #     y_prediction[i] = np.where(y_proba_final[i, :] == np.max(y_proba_final[i, :]))[0][0]
        return classifier_w

    def chose_sorted(self,heatmap, top_number):
        sortlist = np.argsort(heatmap, axis=None)[::-1][:top_number]
        i_num = [i // heatmap.shape[1] for i in sortlist]
        j_num = [j % heatmap.shape[1] for j in sortlist]

        return i_num, j_num

    def featureid_saving(self, operator, X_train, y_train, **kwargs):

        # information_theoretical_based feature selection
        if 'feanum' in kwargs.keys():
            # refer_num_fea = kwargs['feanum']
            num_fea = kwargs['feanum']
        #####################################################################################
        # # 对x_train做p值检验,注意：不能同时对x_train、x_test做p值，这两者筛选得到的特征不一样
        # Pvalue = p_test(X_train, y_train, columnsname) # 加临床特征时columnames要改
        # Pfeatures, Pindex = p_test_excel_new(Pvalue, dist, phase)
        #
        # num_fea = 0
        # if len(Pindex)==0:
        #     num_fea = refer_num_fea  # 没出现p值<0.05的特征，使用全部特征
        #     X_train = X_train
        # elif 0 < len(Pindex) < refer_num_fea:  # 出现p值<0.05的特征数小于5，num_fea设为特征筛查后的实际特征数
        #     num_fea = len(Pindex)
        #     X_train = X_train[:, Pindex]
        # elif len(Pindex) >= refer_num_fea:     # 出现p值<0.05的特征数不小于5，num_fea设为想要的特征数
        #     num_fea = refer_num_fea
        #     X_train = X_train[:, Pindex]        # 选择的特征
        ###################################################################################

        # obtain the index of selected features on the training set
        if operator == "CIFE_main":  # 1
            idx, _, _ = CIFE.cife(X_train, y_train, n_selected_features=num_fea)
        elif operator == "CMIM_main":  # 2
            idx, _, _ = CMIM.cmim(X_train, y_train, n_selected_features=num_fea)
        elif operator == "DISR_main":  # 3
            idx, _, _ = DISR.disr(X_train, y_train, n_selected_features=num_fea)
        elif operator == "FCBF_main":  # 4
            idx, _ = FCBF.fcbf(X_train, y_train, delta=0)
        elif operator == "ICAP_main":  # 5
            idx, _, _ = ICAP.icap(X_train, y_train, n_selected_features=num_fea)
        elif operator == "JMI_main":  # 6
            idx, _, _ = JMI.jmi(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MIFS_main":  # 7
            idx, _, _ = MIFS.mifs(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MIM_main":  # 8
            idx, _, _ = MIM.mim(X_train, y_train, n_selected_features=num_fea)
        elif operator == "MRMR_main":  # 9
            idx, _, _ = MRMR.mrmr(X_train, y_train, n_selected_features=num_fea)
        elif operator == "fisher_score_main":  # 10
            score = fisher_score.fisher_score(X_train, y_train)  # the larger the score, the more important the
            idx = fisher_score.feature_ranking(score)
        elif operator == "lap_score_main":  # 11
            kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            score = lap_score.lap_score(X_train, W=W)
            idx = lap_score.feature_ranking(score)
        elif operator == "reliefF_main":  # 12
            score = reliefF.reliefF(X_train, y_train)
            # rank features in descending order according to score
            idx = reliefF.feature_ranking(score)
        elif operator == "SPEC_main":  # 13
            # specify the second ranking function which uses all except the 1st eigenvalue
            kwargs_style = {'style': 0}
            # obtain the scores of features
            score = SPEC.spec(X_train, **kwargs_style)
            # sort the feature scores in an descending order according to the feature scores
            idx = SPEC.feature_ranking(score, **kwargs_style)
        elif operator == "trace_ratio_main":  # 14
            idx, _, _ = trace_ratio.trace_ratio(X_train, y_train, num_fea, style='fisher')
        elif operator == "ll_l21_main":  # 15
            Y_train = construct_label_matrix_pan(y_train)
            Weight, obj, value_gamma = ll_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "ls_l21_main":  # 16
            # obtain the feature weight matrix
            Y_train = construct_label_matrix_pan(y_train)
            Weight, obj, value_gamma = ls_l21.proximal_gradient_descent(X_train, Y_train, 0.1, verbose=False)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "MCFS_main":  # 17
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            num_cluster = 2  # specify the number of clusters, it is usually set as the number of classes in the ground truth
            # obtain the feature weight matrix
            Weight = MCFS.mcfs(X_train, n_selected_features=num_fea, W=W, n_clusters=num_cluster)
            # sort the feature scores in an ascending order according to the feature scores
            idx = MCFS.feature_ranking(Weight)
        elif operator == "NDFS_main":  # 18
            # construct affinity matrix
            kwargs_W = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
            W = construct_W.construct_W(X_train, **kwargs_W)
            # obtain the feature weight matrix
            Weight = NDFS.ndfs(X_train, W=W, n_clusters=2)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "RFS_main":  # 19
            # obtain the feature weight matrix
            Y_train = construct_label_matrix_pan(y_train)
            Weight = RFS.rfs(X_train, Y_train, gamma=0.1)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "UDFS_main":  # 20
            # obtain the feature weight matrix
            Weight = UDFS.udfs(X_train, gamma=0.1, n_clusters=2)
            # sort the feature scores in an ascending order according to the feature scores
            idx = feature_ranking(Weight)
        elif operator == "CFS_main":  # 21
            idx = CFS.cfs(X_train, y_train)
        elif operator == "chi_square_main":  # 22
            # obtain the chi-square score of each feature
            score = chi_square.chi_square(X_train, y_train)
            # rank features in descending order according to score
            idx = chi_square.feature_ranking(score)
        elif operator == "f_score_main":  # 23
            # obtain the chi-square score of each feature
            score = f_score.f_score(X_train, y_train)
            # rank features in descending order according to score
            idx = f_score.feature_ranking(score)
        elif operator == "gini_index_main":  # 24
            # obtain the chi-square score of each feature
            score = gini_index.gini_index(X_train, y_train)
            # rank features in descending order according to score
            idx = gini_index.feature_ranking(score)
        elif operator == "t_score_main":  # 25
            # obtain the chi-square score of each feature
            score = t_score.t_score(X_train, y_train)
            # rank features in descending order according to score
            idx = t_score.feature_ranking(score)
        elif operator == "alpha_investing_main":  # 26
            idx = alpha_investing.alpha_investing(X_train, y_train, 0.5, 0.5)
        elif operator == "decision_tree_backward_main":  # 27
            idx = decision_tree_backward.decision_tree_backward(X_train, y_train, num_fea)
        elif operator == "decision_tree_forward_main":  # 28
            idx = decision_tree_forward.decision_tree_forward(X_train, y_train, num_fea)
        elif operator == "svm_backward_main":  # 29
            idx = svm_backward.svm_backward(X_train, y_train, num_fea)
        elif operator == "svm_forward_main":  # 30
            idx = svm_forward.svm_forward(X_train, y_train, num_fea)

        return idx

    def heat_map_auc(self,name1, idx, X_train, y_train, X_test, y_test, clf, **kwargs):
        # information_theoretical_based feature selection
        if 'feanum' in kwargs.keys():
            # refer_num_fea = kwargs['feanum']
            num_fea = kwargs['feanum']

        if 'FLAG_DataBalance' in kwargs.keys():
            FLAG_DataBalance = kwargs['FLAG_DataBalance']
        else:
            FLAG_DataBalance = False
        ############## test ##############
        featuresidex = []

        ########################################################################
        # # 做p值后的idx， 根据x_train选择的特征值索引找到对应的x_test
        # X_train_selected = X_train[:, idx[0:num_fea]]
        # if len(Pindex) == 0:
        #     X_test_selected = X_test[:, idx[0:num_fea]]
        #     featuresidex.append(idx[0:num_fea])
        # else:
        #     Pfeatures = np.array(Pfeatures)
        #     select_features = Pfeatures[idx[0:num_fea]]
        #     # 在columnsname里找到选择的特征的对应索引
        #     select_features_idx = []
        #     for i in range(len(select_features)):
        #         select_idx = np.where(columnsname==select_features[i])
        #         select_features_idx.append(select_idx[0][0])
        #     X_test_selected = X_test[:, select_features_idx]
        #     # 存储原来的idx
        #     featuresidex.append(select_features_idx)
        #########################################################################

        # 原本的idx
        X_train_selected = X_train[:, idx[0:num_fea]]
        X_test_selected = X_test[:, idx[0:num_fea]]
        # 存储选择出来的特征的idx
        featuresidex.append(idx[0:num_fea])

        # 数据均衡
        # sm = SMOTE(kind='regular')  # kind = ['regular', 'borderline1', 'borderline2', 'svm']
        # X_selected_resampled, y_selected_resampled = sm.fit_sample(X_train_selected, y_train)

        ADA = ADASYN(ratio=0.5)
        X_selected_resampled, y_selected_resampled = ADA.fit_sample(X_train_selected, y_train)

        # 网格搜索
        if name1 == "Logistic Regression":
            # Grid_Dict = {'C': range(1, 10), 'solver':['liblinear','lbfgs']}
            Grid_Dict = {}
        elif name1 == "SVM":
            Grid_Dict = {}  # 'kernel':['rbf','linear','poly'],'max_iter':[50,100]
        elif name1 == "Naive Bayes":
            Grid_Dict = {}
        elif name1 == "KNN":
            # Grid_Dict = {'n_neighbors':[5, 7]} # 'n_neighbors': range(2, 15), 'p':range(1, 10)
            Grid_Dict = {}
        elif name1 == "Decision Tree":
            # Grid_Dict = {'random_state': range(0, 15)}
            Grid_Dict = {}
        elif name1 == "Extra Trees":
            # Grid_Dict = {'min_samples_split': range(2, 6)}
            Grid_Dict = {}
        elif name1 == "Bagging":
            Grid_Dict = {}
        elif name1 == "Random Forest":
            Grid_Dict = {}  # 速度很慢，'random_state': range(0, 10)
        elif name1 == "AdaBoost":
            Grid_Dict = {}  # 速度很慢，'learning_rate': [0.1, 1, 10]
        elif name1 == "GradientBoosting":
            Grid_Dict = {}  # 'learning_rate': [0.01, 0.1, 1] ，'random_state': range(0, 10)# 速度很慢
        elif name1 == "XGBoost":
            Grid_Dict = {}
        elif name1 == "LightGBM":
            Grid_Dict = {}
        elif name1 == "Catboost":
            Grid_Dict = {}
        elif name1 == "ExtraTrees":
            Grid_Dict = {}
        elif name1 == "LDA":
            Grid_Dict = {}
        elif name1 == "QDA":
            Grid_Dict = {}
        elif name1 == "MLP":
            Grid_Dict = {}

        Classifier = GridSearchCV(clf, cv=3, param_grid=Grid_Dict)
        Classifier.fit(X_selected_resampled, y_selected_resampled)

        ############### test ################
        y_predict = Classifier.predict(X_test_selected)
        sen_o, spe_o = self.cal_sen_spe(y_test, y_predict)
        acc_o = accuracy_score(y_test, y_predict)
        y_predict_prob = Classifier.predict_proba(X_test_selected)
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob[:, 1])
        auc_o = auc(fpr, tpr)

        return auc_o, acc_o, sen_o, spe_o


if __name__ == '__main__':
    #   读数据
    data_path = 'F:/SUBJECT/ZHONG/EOC/data_washing/excel_disp/standerlized/'
    file_name = 'boxcox_adjusted_TJ.csv'
    temp_data = pd.read_csv(data_path  + file_name, encoding='gbk').values

    out_name = 'Perfomance.csv'

    #  数据处理
    X = temp_data[:, 2:]
    y = temp_data[:, 1]
    DP = Data_processing()
    DP.trainning_data_processing(X)
    # joblib.dump(DP, 'F:/SUBJECT/ZHONG/EOC/Result/CLF_saving/data_processing/Processing.model')

    X = DP.transform(X, imp=True, boxcox=False, standered=False)

    # 模型训练
    Fusion_num = 5
    n_fold = 5


    MYclf = MCDF_CLF()
    skf = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)

    Perform = np.zeros([4, Fusion_num+1])
    for train_id, test_id in skf.split(X, y):

        X_train = X[train_id, :]
        X_test = X[test_id, :]
        y_train = y[train_id]
        y_test = y[test_id]
        MYclf.fit(X_train, y_train, sample_weight=0.5)
        y_pre, y_pro = MYclf.predict_and_prob(X_test)



        for i in range(Fusion_num):
            temp_X_test = X_test[:, MYclf.feaselect_id[MYclf.fea_id[i], :]]
            exec('temp_pre = MYclf.basic_clf_' + str(i) + '.predict(temp_X_test)')
            exec('temp_prob = MYclf.basic_clf_' + str(i) + '.predict_proba(temp_X_test)')
            sen_o, spe_o = MYclf.cal_sen_spe(y_test, temp_pre)
            acc_o = accuracy_score(y_test, temp_pre)
            fpr, tpr, thresholds = roc_curve(y_test, temp_prob[:, 1])
            auc_o = auc(fpr, tpr)

            Perform[0, i] += auc_o / n_fold
            Perform[1, i] += acc_o / n_fold
            Perform[2, i] += sen_o / n_fold
            Perform[3, i] += spe_o / n_fold

        sen_o, spe_o = MYclf.cal_sen_spe(y_test, y_pre)
        acc_o = accuracy_score(y_test, y_pre)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        auc_o = auc(fpr, tpr)

        Perform[0, -1] += auc_o / n_fold
        Perform[1, -1] += acc_o / n_fold
        Perform[2, -1] += sen_o / n_fold
        Perform[3, -1] += spe_o / n_fold

    final_head = ['CLF1', 'CLF2', 'CLF3', 'CLF4', 'CLF5', 'MCF']
    final_out = pd.DataFrame(data=Perform,columns=final_head)
    final_out.to_csv(data_path + out_name, index=False, encoding='gbk')




