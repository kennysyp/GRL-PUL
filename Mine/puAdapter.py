import numpy as np

class PUAdapter(object):
    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False):
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio

        if precomputed_kernel:
            self.fit = self.__fit_precomputed_kernel
        else:
            self.fit = self.__fit_no_precomputed_kernel

        self.estimator_fitted = False


    def __str__(self):
        return 'Estimator:' + str(self.estimator) + '\n' + 'p(s=1|y=1,x) ~= ' + str(self.c) + '\n' + \
            'Fitted: ' + str(self.estimator_fitted)


    def __fit_precomputed_kernel(self, X, y):
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        if len(positives) <= hold_out_size:
            raise Exception('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]

        #Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:,keep]

        #New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass

        print(f"hold_out_predictions_shape2:{len(hold_out_predictions)}")
        print(f"hold_out_predictions_type2:{type(hold_out_predictions)}")
        c = np.mean(hold_out_predictions)  # 对保留集合进行预测
        print(f"hold_out_predictions_shape3:{hold_out_predictions.shape}")
        print(f"hold_out_predictions_type3:{type(hold_out_predictions)}")
        self.c = c

        self.estimator_fitted = True


    def __fit_no_precomputed_kernel(self, X, y):
        positives = np.where(y == 1)[0]
        print(f"len(positives):{len(positives)}")
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))

        if len(positives) <= hold_out_size:
            raise Exception('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]

        X = np.delete(X, hold_out,0)
        #print(f"y_shape1:{y.shape}")
        y = np.delete(y, hold_out)
        #print(f"y_shape2:{y.shape}")

        self.estimator.fit(X, y)

        #print(f"X_hold_out_shape:{X_hold_out.shape}")
        #print(f"X_hold_out_type:{type(X_hold_out)}")
        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:,1]
        except:
            pass

        #print(f"hold_out_predictions_shape2:{hold_out_predictions.shape}")
        #print(f"hold_out_predictions_type2:{type(hold_out_predictions)}")
        c = np.mean(hold_out_predictions)  # 对保留集合进行预测
        #print(f"hold_out_predictions:{hold_out_predictions}")
        #print(f"hold_out_predictions_shape3:{hold_out_predictions.shape}")
        #print(f"hold_out_predictions_type3:{type(hold_out_predictions)}")
        self.c = c

        self.estimator_fitted = True


    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')
        probabilistic_predictions = self.estimator.predict_proba(X)
        try:
            probabilistic_predictions = probabilistic_predictions[:,1]
        except:
            pass
        return probabilistic_predictions / self.c


    def predict(self, X, treshold=0.58):
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')
        #return np.array([1 if p > treshold else 0 for p in self.predict_proba(X)])
        return self.predict_proba(X)

