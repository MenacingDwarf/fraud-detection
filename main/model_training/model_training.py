import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
import pickle


from django.contrib.auth.models import User
from main.models import Transaction
from main.serializers import TransactionSerializer


def retrain_model():
    df = pd.read_csv('main/model_training/input/PS_20174392719_1491204439457_log.csv', nrows=1000000)
    df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig': 'newBalanceOrig', \
                            'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest': 'newBalanceDest'})

    X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

    randomState = 5
    np.random.seed(randomState)

    Y = X['isFraud']
    del X['isFraud']

    # Eliminate columns shown to be irrelevant for analysis in the EDA
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

    # Binary-encoding of labelled data in 'type'
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)  # convert dtype('O') to dtype(int)

    X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0),
          ['oldBalanceDest', 'newBalanceDest']] = - 1

    X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0),
          ['oldBalanceOrig', 'newBalanceOrig']] = np.nan

    X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
    X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest

    weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
    clf = XGBClassifier(max_depth=3, scale_pos_weight=weights,
                        n_jobs=4)
    clf.fit(X, Y)
    pickle.dump(clf, open("main/model_training/trained_model.dat", "wb"))

    test_bank = User.objects.get(username='test_bank')
    i = 0
    for index, transaction_data in X.iterrows():
        transaction_data["type"] = int(transaction_data["type"])
        transaction_data["step"] = int(transaction_data["step"])
        transaction_data["identifier"] = index
        transaction = TransactionSerializer(data=dict(transaction_data))
        if transaction.is_valid():
            print(Y.iloc[i])
            transaction.save(ownerBank=test_bank, trained=True, prediction=Y.iloc[i])

        i += 1


def prepare_data(transaction):
    transaction = {x: y[0] for x, y in transaction.items()}
    new_transaction = {}
    trans_type = transaction["type"]
    new_transaction["step"] = int(transaction["step"])
    new_transaction["type"] = transaction["amount"]
    new_transaction["amount"] = float(transaction["amount"])
    new_transaction["oldBalanceOrig"] = float(transaction["oldBalanceOrig"])
    new_transaction["newBalanceOrig"] = float(transaction["newBalanceOrig"])
    new_transaction["oldBalanceDest"] = float(transaction["oldBalanceDest"])
    new_transaction["newBalanceDest"] = float(transaction["newBalanceDest"])

    if trans_type != 'TRANSFER' and trans_type != 'CASH_OUT':
        return False
    new_transaction["type"] = 1 if trans_type == 'CASH_OUT' else 0

    if (new_transaction["oldBalanceDest"] == 0) & (new_transaction["newBalanceDest"] == 0) & (new_transaction["amount"] != 0):
        new_transaction["oldBalanceDest"] = -1
        new_transaction["newBalanceDest"] = -1

    if (new_transaction["oldBalanceOrig"] == 0) & (new_transaction["newBalanceOrig"] == 0) & (new_transaction["amount"] != 0):
        new_transaction["oldBalanceOrig"] = np.nan
        new_transaction["newBalanceOrig"] = np.nan

    new_transaction['errorBalanceOrig'] = new_transaction["newBalanceOrig"] + new_transaction["amount"] - new_transaction[
        "oldBalanceOrig"]
    new_transaction['errorBalanceDest'] = new_transaction["oldBalanceDest"] + new_transaction["amount"] - new_transaction[
        "newBalanceDest"]
    return new_transaction


def predict_transaction(transaction):
    x_dict = {x: [y] for x, y in transaction.items()}

    model = pickle.load(open("main/model_training/trained_model.dat", "rb"))

    return bool(model.predict(pd.DataFrame.from_dict(x_dict))[0])


# retrain_model()
transaction_descr = {"step": 205, "type": 'CASH_OUT', "amount": 23543.56,
                     "oldBalanceOrig": np.nan, "newBalanceOrig": np.nan,
                     "oldBalanceDest": 203469.86, "newBalanceDest": 227013.42,
                     "errorBalanceOrig": np.nan, "errorBalanceDest": -2.910383e-11
                     }

if __name__ == '__main__':
    retrain_model()