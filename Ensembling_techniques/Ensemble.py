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
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import joblib
# 31 9:00
class Ensemble:
        def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, dtc_params, xgb_params, 
                     rfc_params, knn_params, gdb_params, lgbm_params, adb_params, lr_params, k=5, scaler = None):
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
                self.X_test = X_test
                self.y_test = y_test
                self.dtc_params = dtc_params
                self.xgb_params = xgb_params
                self.rfc_params = rfc_params
                self.knn_params = knn_params
                self.gdb_params = gdb_params
                self.lgbm_params = lgbm_params
                self.adb_params = adb_params
                self.lr_params = lr_params
                self.k = k
                # self.scaler = StandardScaler()

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
                
        # individual classification methods             
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
                        return LGBMClassifier(andom_state = 42, **self.lgbm_params)
                if name == 'adb':
                        return AdaBoostClassifier(random_state = 42, **self.adb_params)
                if name == 'lr':
                        return LogisticRegression(random_state = 42, **self.lr_params)
                
                raise ValueError(f"Unknown classifier name: {name}")

        # def __scale_data(self):
        #         self.X_train = self.scaler.fit_transform(self.X_train)
        #         self.X_test = self.scaler.transform(self.X_test)
                
                
        # weak classifiers , to check individually
        def __DecisionTreeClassifier(self):
                print(f"DT")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                decision_tree = self.__classifiers(name = 'dt')

                decision_tree.fit(self.X_train, self.y_train)

                # y_pred = decision_tree.predict(self.X_test)
                self.__check_metrics_save_model(decision_tree, None, name= "DecisionTreeClassifier")

                return decision_tree
                
        def __RandomForestClassifier(self):
                print(f"RFC")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                rfc = self.__classifiers(name='rf')

                rfc.fit(self.X_train, self.y_train)

                # y_pred = rfc.predict(self.X_test)
                self.__check_metrics_save_model(rfc, None, name= "RandomForestClassifier")

                return rfc
                
        def __KNeighborsClassifier(self):#, X_train = None, y_train = None):
                print(f"KNNS")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                knn = self.__classifiers(name = 'knn')
                # if need to scall the data
                # scaled_X_train = self.scale_train_data()
                # scaled_X_test = self.scale_test_data()
                
                knn.fit(self.X_train, self.y_train)
                # y_pred = knn.predict(self.X_test)
                self.__check_metrics_save_model(knn, None, name = "KNeighborsClassifier")

                return knn
                
        def __XGBClassifier(self): #X_train = None, y_train = None):
                print(f"XGB")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                xgb = self.__classifiers(name='xgb')

                xgb.fit(self.X_train, self.y_train)
                # y_pred = xgb.predict(self.X_test)
                self.__check_metrics_save_model(xgb, None, name = "XGBClassifier")

                return xgb

        def __GradientBoostingClassifier(self):#, X_train = None, y_train = None):
                print(f"GDB")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                gdb = self.__classifiers(name = 'gdb')

                gdb.fit(self.X_train, self.y_train)
                # y_pred = gdb.predict(self.X_test)
                self.__check_metrics_save_model(gdb, None, name = "GradientBoostingClassifier")

                return gdb

        def __LightGBMClassifier(self):# X_train = None, y_train = None):
                print(f"LGBM")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                lgbm = self.__classifiers(name = 'lgbm')

                lgbm.fit(self.X_train, self.y_train)
                # y_pred = lgbm.predict(self.X_test)
                self.__check_metrics_save_model(lgbm, None, name = "LightGBMClassifier") 

                return lgbm

        def __AdaBoostClassifier(self):# X_train = None, y_train = None):
                print(f"AdaBooster")
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                adb = self.__classifiers(name = 'adb')

                adb.fit(self.X_train, self.y_train)
                # y_pred = adb.predict(self.X_test)
                self.__check_metrics_save_model(adb, None, name = "AdaBoostClassifier") 

                return adb

        def __LogisticRegressionClassifier(self): #X_train = None, y_train = None):
                print(f"Logistic Regression")

                # needed to check when k_fold function are using, that time i need to change the X_train and y_train
                # if X_train is not None and y_train is not None:
                #         self.X_train = X_train
                #         self.y_train = y_train
                lr = self.__classifiers(name = 'lr')

                lr.fit(self.X_train, self.y_train)

                self.__check_metrics_save_model(lr, None, name = "logistic_regression")

                return lr
        
        def __BlendingClassifier(self):
                # base model - any model type
                weak_learners = [['rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb']]
                #  [['dt', 'knn', 'rf', 'xgb'],
                # #                  ['gdb','lgbm','adb', 'xgb']]
                # meta model - linear model
                final_learner = self.__classifiers('lr')
        
                # self.voating_classifier(final_learner = final_learner_0)
                train_meta_model = None
                test_meta_model = None
                name = "blending_"
                for lst in weak_learners: # loop over the list of meta models
                      # for l in lst:                     
                        for model in lst:
                          # fit the individual
                          name += model + "_" 
                          val_pred, test_pred = self.__blending_train_level_0(model)

                          # the validation prediction will be the input data for training the mata model
                          if isinstance(train_meta_model, np.ndarray):
                                train_meta_model = np.vstack((train_meta_model, val_pred))
                          else:
                                train_meta_model = val_pred

                          # the test prediction will be the input data for testing the meta model
                          if isinstance(test_meta_model, np.ndarray):
                                test_meta_model = np.vstack((test_meta_model, test_pred))
                          else:
                                test_meta_model = test_pred
                        # representation of the outputs from each base model
                        # each column - ouput of each base model
                        train_meta_model = train_meta_model.T
                        test_meta_model = test_meta_model.T
                        self.__train_level_1(final_learner, train_meta_model, test_meta_model, name)

        def __blending_train_level_0(self, model):

                # fit on train data
                fitted_model = self.__classifiers(model)
                fitted_model.fit(self.X_train, self.y_train) # changed

                # predict on validation
                validation_prediction = fitted_model.predict(self.X_val)
                # predict on test data
                test_prediction = fitted_model.predict(self.X_test)

                return validation_prediction, test_prediction
                          
        def __StackingClassifier(self):
        # DecisionTreeClassifier(random_state = 42,**self.dtc_params)
        # RandomForestClassifier(random_state = 42, **self.rfc_params)
        # KNeighborsClassifier(**self.knn_params)
        # XGBClassifier(random_state = 42, **self.xgb_params)
        # GradientBoostingClassifier(random_state = 42, **self.gdb_params)
        # LGBMClassifier(andom_state = 42, **self.lgbm_params)
        # AdaBoostClassifier(random_state = 42, **self.adb_params)
        # LogisticRegression(random_state = 42, **self.lr_params)
                # weak_learners = [(RandomForestClassifier(random_state = 42, **self.rfc_params)), 
                #                   (XGBClassifier(random_state = 42, **self.xgb_params)), 
                #                   (DecisionTreeClassifier(random_state = 42,**self.dtc_params)), 
                #                   (KNeighborsClassifier(**self.knn_params)), 
                #                   (GradientBoostingClassifier(random_state = 42, **self.gdb_params)), 
                #                   (LGBMClassifier(random_state = 42, **self.lgbm_params)), 
                #                   (AdaBoostClassifier(random_state = 42, **self.adb_params))]
                weak_learners = [['rf', 'xgb', 'dt', 'knn', 'gdb', 'lgbm', 'adb']]
                # [['dt', 'knn', 'rf', 'xgb'],
                #                  ['gdb','lgbm','adb', 'xgb'],
                #                  ,
                #                         ] 
                final_learner = self.__classifiers('lr')
        
                # self.voating_classifier(final_learner = final_learner_0)
                train_meta_model = None
                test_meta_model = None
                name = "stacking_"
                for lst in weak_learners:
                      # for l in lst:  
                        print(lst)                   
                        for model in lst:
                          print(model)
                          # fit the individual 
                          # val_pred, test_pred = self.__train_level_0(model)
                          name += model + "_"
                          
                          prediction_clf = self.__k_fold_cross_validation(model)

                          test_predictions_clf = self.__stacking_traini_level_0(model)

                          if isinstance(train_meta_model, np.ndarray):
                                  train_meta_model = np.vstack((train_meta_model, prediction_clf))
                                  print(f"Train Meta Model type: {type(train_meta_model)}\n")
                                  print(f"Train Meta Model shape: {train_meta_model.shape}\n")
                          else:
                                  train_meta_model = prediction_clf
                                  print(f"Train Meta Model type: {type(train_meta_model)}\n")
                                  print(f"Train Meta Model shape: {train_meta_model.shape}\n")
                          if isinstance(test_meta_model, np.ndarray):
                                  test_meta_model = np.vstack((test_meta_model, test_predictions_clf))
                                  print(f"Test Meta Model type: {type(test_meta_model)}\n")
                                  print(f"Test Meta Model shape: {test_meta_model.shape}\n")
                          else:
                                  test_meta_model = test_predictions_clf
                                  print(f"Test Meta Model type: {type(test_meta_model)}\n")
                                  print(f"Test Meta Model shape: {test_meta_model.shape}\n")
                        train_meta_model = train_meta_model.T
                        print(f"After Transpose, Train Meta Model type: {type(train_meta_model)}\n")
                        print(f"After Transpose Train Meta Model shape: {train_meta_model.shape}\n")
                        test_meta_model = test_meta_model.T
                        print(f"After transpose Test Meta Model type: {type(test_meta_model)}\n")
                        print(f"After Transpose Test Meta Model shape: {test_meta_model.shape}\n")
                        self.__train_level_1(final_learner, train_meta_model, test_meta_model,name)

                
        
        def __k_fold_cross_validation(self, clf):
                print("In  __k_fold_cross_validation(self, clf)\n")
                predictions_clf = []
                # Error in stacking classifier: list indices must be integers or slices, not list
                y_train = np.array(self.y_train) #
                print(f"model: {clf}")
                batch_size = int(len(self.X_train)/self.k)
                print(f"Batch Size: {batch_size}")
                for fold in range(self.k):
                        print(f"fold:{fold}\n")
                        if fold == (self.k - 1):
                                # test = self.X_train[(batch_size * fold):, :]
                                batch_start = batch_size * fold
                                batch_finish = self.X_train.shape[0]
                        else:
                                # test = self.X_train[(batch_size * fold): (batch_size * (fold + 1)), :]
                                batch_start = batch_size * fold
                                batch_finish = batch_size * (fold + 1)
                        print(f"Batch Start:{batch_start}\n")
                        print(f"Batch Finish:{batch_finish}\n")
                        # test & training samples for each fold iteration
                        fold_x_test = self.X_train[batch_start:batch_finish, :]
                        print(f"fold_x_test shape: {fold_x_test.shape}\n")
                        print(f"fold_x_test type: {type(fold_x_test)}\n")
                        fold_x_train = self.X_train[[index for index in range(self.X_train.shape[0]) if
                                                        index not in range(batch_start, batch_finish)], :]
                        
                        print(f"fold_x_train shape: {fold_x_train.shape}\n")
                        print(f"fold_x_train type: {type(fold_x_train)}\n")

                        # test & training targets for each fold iteration
                        print(f"type of y_train : {type(self.y_train)}\n")
                        fold_y_test = y_train[batch_start:batch_finish]#
                        print(fold_y_test)
                        print(f"fold_y_test type: {type(fold_y_test)}\n")
                        fold_y_train = y_train[
                                [index for index in range(self.X_train.shape[0]) if index not in range(batch_start, batch_finish)]]#
                        print(fold_y_train)
                        print(f"fold_y_train type: {type(fold_y_train)}\n")
                        # Fit current classifier
                        # clf.fit(fold_x_train, fold_y_train)
                        model = self.__classifiers(name=clf)
                        print(f"model: {model}")
                        model.fit(fold_x_train, fold_y_train)
                        # model = self.__classifiers(name=clf, fold_x_train=fold_x_train, fold_y_train=fold_y_train)
                        fold_y_pred = model.predict(fold_x_test)
                        print(f"fold_y_predict shape: {fold_y_pred.shape}\n")
                        print(f"fold_y_prediction: {fold_y_pred[0:5]}\n")
                        # Store predictions for each fold_x_test
                        if isinstance(predictions_clf, np.ndarray):
                                predictions_clf = np.concatenate((predictions_clf, fold_y_pred))
                        else:
                                predictions_clf = fold_y_pred
                        print(f"Prediction_clf : {predictions_clf[0:5]}\n")
                return predictions_clf

                       
        def __stacking_traini_level_0(self, clf):
                print("\nIn __stacking_traini_level_0(self, model)\n")
                model = self.__classifiers(clf)
                print(f"model: {model}")
                model.fit(self.X_train, self.y_train)##
                y_pred = model.predict(self.X_test)
                print(y_pred[0:5])

                # self.__check_metrics_save_model(None,y_pred)
                return y_pred

      

        def __train_level_1(self, final_learner, train_meta_model, test_meta_model, name):
                print("\nIn __train_level_1(self, final_learner, train_meta_model, test_meta_model, name)\n")
                print(f"Final Learner: {final_learner}\n")
                print(f"Train Meta Model: {train_meta_model}\n")
                print(f"Test Meta Model: {test_meta_model}\n")
                print(f"Models name: {name}\n")
                final_learner.fit(train_meta_model, self.y_val) # validation data was used in train_meta model 
                # final_learner_pred = final_learner.predict(test_meta_model)
                # Error in Blending classifier: X has 4666 features, but LogisticRegression is expecting 6 features as input.
                # comment this line
                # self.__check_metrics_save_model(final_learner, None)# this one is taking the original model which need 4666 features, but from the base 6 models , now we have 6 features.
                # we need to predict here than send the predicted outpu to the function.

                test_preds = final_learner.predict(test_meta_model)
                print(f"Test Prediction: {test_preds[0:5]}\n")
                pd.DataFrame(test_preds).to_csv(f"test_preds_{name}.csv", index=False)
                test_preds = np.array([1 if i == 'Yes' else 0 for i in test_preds])
                print(f"Calling __check_metrics_save_model(final_learner, test_preds, name)\n")
                self.__check_metrics_save_model(final_learner, test_preds, name)     
                #return 0

                              
                        
                        
                                                                                        
        

        def __VotingClassifier(self):
                # instantiate different set of classifiers
                classifiers_dict = {
                'classifier_names':['dt', 'knn', 'rf', 'xgb'],
                'classifier_names_2':['rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb'],
                'classifier_names_3':['gdb', 'lgbm', 'adb'],
                'classifier_names_4':['dt', 'knn', 'rf', 'xgb', 'adb'],
                'classifier_names_5':['rf', 'xgb', 'dt'],
                'classifier_names_6':['xgb', 'lgbm', 'adb', 'lr'],
                'classifier_names_7':['dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf']
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
                        self.__check_metrics_save_model(vc, None, name = "VotingClassifier_" + "_".join(v))

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
                
        def __check_metrics_save_model(self, model = None, y_pred = None, name = None):
                # y_pred = self.__VoatingClassifier()   
                timestamp = datetime.now().strftime("%Y%m%d_")

                if model is None and y_pred is None:
                       raise ValueError("Either model or y_pred must be provided.")
                elif y_pred is None:
                        y_pred = model.predict(self.X_test)

                acc = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average = 'weighted')
                recall = recall_score(self.y_test, y_pred, average = 'weighted')
                f1 = f1_score(self.y_test, y_pred, average = 'weighted')
                f1_val = round(f1, 3)
                name += f"{model.__class__.__name__}_{timestamp}_{f1_val}"
                # model_name += name 
                print(f"Savnig this model.......")
                file_path = f'/content/drive/MyDrive/Thesis/models/{name}.pkl'
                with open(file_path, 'wb') as f:
                        pickle.dump(model, f)
                # joblib.dump(model, file_path)
                print(f"saved successfully....{name}")
                print(f"Model Prerformance: {name}")
                # print(f"Model:{model_name}")
                print(f"Accuracy: {acc:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1 Score: {f1:.3f}")
                
                

        def run_pipeline(self):
        
                ########################################################
                print("******Decision Tree Classifier******")
                try:
                        
                
                        self.__DecisionTreeClassifier()
                except Exception as e:
                        print(f"Error in Decision Tree Classifier: {e}")
                print('\n\n')
                #########################################################
                print("------KNN Classifier-----")
                try:

                        self.__KNeighborsClassifier()
                except Exception as e:
                        print(f"Error in KNN Classifier: {e}")
                print('\n\n')
                ########################################################
                print(",,,,,,XGB Classifier,,,,,,")
                try:

                        self.__XGBClassifier()
                except Exception as e:
                        print(f"Error in XGB Classifier: {e}")
                print('\n\n')
                ###########################################################
                print("!!!!!!Random Forest Classifier!!!!!!!")
                try:
                        self.__RandomForestClassifier()
                except Exception as e:
                        print(f"Error in Random Forest Classifier: {e}")
                print('\n\n')
                ############################################################
                print("~~~~~~~Gradient Boosting Classifier~~~~~~~")
                try:

                        self.__GradientBoostingClassifier()
                except Exception as e:
                        print(f"Error in Gradient Boosting Classifier: {e}")
                print('\n\n')
                #########################################################
                print("~~~~~~~~Light GBM Classifier~~~~~~~~")
                try:

                        self.__LightGBMClassifier()
                except Exception as e:
                        print(f"Error in Light GBM Classifier: {e}")
                print('\n\n')
                ####################################################
                print("~~~~~~~~~~~Ada Boost Classifier~~~~~~~~~~~")
                try:

                        self.__AdaBoostClassifier()
                except Exception as e:
                        print(f"Error in Ada Boost Classifier: {e}")
                print('\n\n')
                #########################################################
                try:
                        self.__LogisticRegressionClassifier()
                except Exception as e:
                        print(f"Error in logistic regression classifier: {e}")
                print('\n\n')
                #####################################################
                print("......Voatinng Classifier.....")
                try:

                        self.__VotingClassifier()
                except Exception as e:
                        print(f"Error in voating classifier: {e}")
                print('\n\n')
                ####################################################
                print("......Blending Classifier.....")
                try:

                        self.__BlendingClassifier()
                except Exception as e:
                        print(f"Error in Blending classifier: {e}")
                print('\n\n')
                # #####################################################
                # print("......Stacking Classifier.....")
                # try:

                #         self.__StackingClassifier()
                # except Exception as e:
                #         print(f"Error in stacking classifier: {e}")
                # print('\n\n')
                # print("Pipeline completed successfully!")



# if __name__ == "__main__":
#         cl = Ensemble()
#         cl.run_pipeline()
                        
                
