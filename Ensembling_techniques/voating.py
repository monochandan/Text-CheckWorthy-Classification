from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

class Ensemble:
        def __init__(self, X_train, y_train, X_test, y_test, dtc_params, xgb_params, rfc_params, knn_params, gdb_params, lgbm_params, adb_params, scaler = None):
                self.X_train = X_train
                self.y_train = y_train
                self.X_test = X_test
                self.y_test = y_test
                self.dtc_params = dtc_params
                self.xgb_params = xgb_params
                self.rfc_params = rfc_params
                self.knn_params = knn_params
                self.gdb_params = gdb_params
                self.lgbm_params = lgbm_params
                self.adb_params = adb_params
                self.scaler = StandardScaler()

        # def set_xtrain(self, data):
        #         # check if the data is numpy.ndarray
        #         if not isinstance(data, np.ndarray):
        #                 self.X_train = np.array(data)
        #         else:
        #                 self.X_train = data
                        
        # def set_ytrain(self, data):
        #       if not isinstance(data, pd.Series):
        #               self.y_train = pd.Series(data)
        #       else:
        #               self.y_train = data

        # def set_xtest(self, data):
        #         # check if the data is numpy.ndarray
        #         if not isinstance(data, np.ndarray):
        #                 self.X_test = np.array(data)
        #         else:
        #                 self.X_test = data
                        
        # def set_ytest(self, data):
        #       if not isinstance(data, pd.Series):
        #               self.y_test = pd.Series(data)
        #       else:
        #               self.y_test = data
                      
        # def set_dtc_params(self, param_dict = None):
        #         self.dtc_params = param_dict
        # def get_dtc_params(self):
        #         return self.dtc_params
        # def set_xgb_params(self, param_dict = None):
        #         self.xgb_params = param_dict
        # def get_xgb_params(self):
        #         return self.xgb_params
        # def set_knn_params(self, param_dict = None):
        #         self.knn_params = param_dict
        # def get_knn_params(self):
        #         return self.knn_params
        # def set_rfc_params(self, param_dict = None):
        #         self.rfc_params = param_dict       
        # def get_rfc_params(self):
        #         return self.rfc_params
                
                      
        #@staticmethod
        def __classifiers(self, name = None): # private method
                random_state = 42
                if name == 'dt':
                        # params = Ensemble.get_dtc_params()
                        return DecisionTreeClassifier(random_state = 42,**self.dtc_params)
                if name == 'rf':
                        # params = Ensemble.get_rfc_params()
                        return RandomForestClassifier(random_state = 42, **self.rfc_params)
                if name == 'knn':
                        # params = Ensemble.get_knn_params()
                        return KNeighborsClassifier(**self.knn_params)
                if name == 'xgb':
                        # params = Ensemble.get_xgb_params()
                        return XGBClassifier(random_state = 42, **self.xgb_params)
                if name == 'gdb':
                        return GradientBoostingClassifier(random_state = 42, **self.gdb_params)
                if name == 'lgbm':
                        return LGBMClassifier(random_state = 42, **self.lgbm_params)
                if name == 'adb':
                        return AdaBoostClassifier(random_state = 42, **self.adb_params)
                
                raise ValueError(f"Unknown classifier name: {name}")

        # def __scale_data(self):
        #         self.X_train = self.scaler.fit_transform(self.X_train)
        #         self.X_test = self.scaler.transform(self.X_test)
                
                
        # weak classifiers , to check individually
        def __DecisionTreeClassifier(self):
                print(f"DT")
                decision_tree = self.__classifiers(name = 'dt')

                decision_tree.fit(self.X_train, self.y_train)

                # y_pred = decision_tree.predict(self.X_test)
                self.__check_metrics_save_model(decision_tree)
                
        def __RandomForestClassifier(self):
                print(f"RFC")
                rfc = self.__classifiers(name='rf')

                rfc.fit(self.X_train, self.y_train)

                # y_pred = rfc.predict(self.X_test)
                self.__check_metrics_save_model(rfc)

                
        def __KNeighborsClassifier(self):
                print(f"KNNS")
                knn = self.__classifiers(name = 'knn')
                # if need to scall the data
                # scaled_X_train = self.scale_train_data()
                # scaled_X_test = self.scale_test_data()
                
                knn.fit(self.X_train, self.y_train)
                # y_pred = knn.predict(self.X_test)
                self.__check_metrics_save_model(knn)
                
        def __XGBClassifier(self):
                print(f"XGB")
                xgb = self.__classifiers(name='XGB')

                xgb.fit(self.X_train, self.y_train)
                # y_pred = xgb.predict(self.X_test)
                self.__check_metrics_save_model(xgb)

        def __GradientBoostingClassifier(self):
                print(f"GDB")
                gdb = self.__classifiers(name = 'gdb')

                gdb.fit(self.X_train, self.y_train)
                # y_pred = gdb.predict(self.X_test)
                self.__check_metrics_save_model(gdb)

        def __LightGBMClassifier(self):
                print(f"LGBM")
                lgbm = self.__classifiers(name = 'lgbm')

                lgbm.fit(self.X_train, self.y_train)
                # y_pred = lgbm.predict(self.X_test)
                self.__check_metrics_save_model(lgbm)

        def __AdaBoostClassifier(self):
                print(f"AdaBooster")
                adb = self.__classifiers(name = 'adb')

                adb.fit(self.X_train, self.y_train)
                # y_pred = adb.predict(self.X_test)
                self.__check_metrics_save_model(adb)
                
        def __VotingClassifier(self):
                # instantiate different set of classifiers
                classifiers_dict = {
                'classifier_names':['dt', 'knn', 'rf', 'xgb'],
                'classifier_names_2':['rf', 'xgb', 'dt', 'knn', 'gdb', 'lgbm', 'adb'],
                'classifier_names_3':['gdb', 'lgbm', 'adb'],
                'classifier_names_4':['dt', 'knn', 'rf', 'xgb', 'adb']
                }
                for k, v in classifiers_dict.items():
                        print(f"Classifiers used: {classifiers_dict[k]}")
                        # classifier_name = [] # initiailzing the list for every list in the dictionary
                        # model_name = "_".join(v)
                        # for name in v:
                        #         classifier_name.append(name) # appending the names from the list of dictionary
                        #         model_name += name
                        #         model_name += "_"
                        classifiers = {name: self.__classifiers(name = name) for name in v}

                        vc = VotingClassifier(
                                estimators = [(name, clf) for name, clf in classifiers.items()],
                                voting = 'soft',
                        )
                        vc.fit(self.X_train,self.y_train)
                        #y_pred = vc.predict(self.X_test)
                        self.__check_metrics_save_model(vc)

                # decision_tree = self.__classifiers(name='dt')
                # knn = self.__classifiers(name='knn')
                # rfc = self.__classifiers(name='rf')
                # xgb = self.__classifiers(name='XGB')

                # initialize voatingClassifier
                # vc = VotingClassifier(estimators = [('decision_tree', decision_tree), ('knn', knn), ('rfc', rfc), ('xgb', xgb)], voting = 'soft')
                # model fit
                # vc.fit(self.X_train,self.y_train)
                # predict 
                # y_pred_train = vc.predict(self.X_train)

               # y_pred = vc.predict(self.X_test)

                # self.__check_metrics(y_pred)
                
        def __check_metrics_save_model(self, model):
                # y_pred = self.__VoatingClassifier()
                y_pred = model.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average = 'weighted')
                recall = recall_score(self.y_test, y_pred, average = 'weighted')
                f1 = f1_score(self.y_test, y_pred, average = 'weighted')
                f1_val = round(f1, 3)
                model_name = f"_f1_{f1_val}"
                print(f"Savnig this model.......")
                file_path = f'/content/drive/MyDrive/Thesis/models/{model_name}.pkl'
                with open(file_path, 'wb') as f:
                        pickle.dump(model, f)
                print(f"saved successfully....{model_name}")
                print(f"Model Prerformance: {model_name}")
                # print(f"Model:{model_name}")
                print(f"Accuracy: {acc:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1 Score: {f1:.3f}")
                
                

        def run_pipeline(self):
                ######################################################
                print("......Voatinng Classifier.....")
                try:

                        self.__VotingClassifier()
                except Exception as e:
                        print(f"Error in voating classifier: {e}")


                #########################################################
                print("******Decision Tree Classifier******")
                try:
                        
                
                        self.__DecisionTreeClassifier()
                except Exception as e:
                        print(f"Error in Decision Tree Classifier: {e}")

                #########################################################
                print("------KNN Classifier-----")
                try:

                        self.__KNeighborsClassifier()
                except Exception as e:
                        print(f"Error in KNN Classifier: {e}")

                ########################################################
                print(",,,,,,XGB Classifier,,,,,,")
                try:

                        self.__XGBClassifier()
                except Exception as e:
                        print(f"Error in XGB Classifier: {e}")

                ###########################################################
                print("!!!!!!Random Forest Classifier!!!!!!!")
                try:
                        self.__RandomForestClassifier()
                except Exception as e:
                        print(f"Error in Random Forest Classifier: {e}")

                ############################################################
                print("~~~~~~~Gradient Boosting Classifier~~~~~~~")
                try:

                        self.__GradientBoostingClassifier()
                except Exception as e:
                        print(f"Error in Gradient Boosting Classifier: {e}")

                #########################################################
                print("~~~~~~~~Light GBM Classifier~~~~~~~~")
                try:

                        self.__LightGBMClassifier()
                except Exception as e:
                        print(f"Error in Light GBM Classifier: {e}")

                ####################################################
                print("~~~~~~~~~~~Ada Boost Classifier~~~~~~~~~~~")
                try:

                        self.__AdaBoostClassifier()
                except Exception as e:
                        print(f"Error in Ada Boost Classifier: {e}")
                #########################################################
                        
                
