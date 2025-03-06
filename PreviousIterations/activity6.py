import pandas as pd
import re
pd.options.mode.chained_assignment = None
#142 NC
#129 C

def classifier(df):
    dfc = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
    dfnc = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
    df1 = df.copy()
    df1['TP'] = dfc.sum(axis=1)
    df1['FP'] = dfnc.sum(axis=1)
    df1['TP-FP'] = df1['TP'] - df1['FP']
    df1['TP-FP']=df1['TP-FP'].astype('int')
    dfB = df1.nlargest(1, 'TP-FP')
    df1 = df1.drop(['TP', 'FP', 'TP-FP'], axis=1)

    dfA = dfB.loc[:, ~(dfB == 0).any()]

    dfA = df1[df1.columns[~df1.columns.isin(list(dfA.columns))]]
    dfAc = dfA[dfA.columns.drop(list(dfA.filter(regex='^NC[0-9]+$')))]
    dfAnc = dfA[dfA.columns.drop(list(dfA.filter(regex='^C[0-9]+$')))]

    dfA['TP'] = dfAc.sum(axis=1)
    dfA['FP'] = dfAnc.sum(axis=1)
    dfA['TP-FP'] = dfA['TP'] - dfA['FP']
    dfA['TP-FP']=dfA['TP-FP'].astype('int')
    dfA = dfA.nlargest(1, 'TP-FP')

    return((list(dfB.index.values)[0],list(dfA.index.values)[0]))

def evaluator(df, f1, f2):
    df1 = df.copy()
    df1 = df1.loc[[f1,f2],:]
    df1 = df1.T
    df1[df1.columns[0]] = df1[df1.columns[0]].astype('bool')
    df1[df1.columns[1]] = df1[df1.columns[1]].astype('bool')
    df1['P']= df1[df1.columns[0]] | df1[df1.columns[1]]
    df1['A']= df1.index.values
    df1['A'] = df1['A'].str.replace("^NC[0-9]+$", '0', regex=True)
    df1['A'] = df1['A'].str.replace("^C[0-9]+$", '1', regex=True)
    map = {'0': False, '1': True}
    df1['A']=df1['A'].map(map)
    df1['E'] = ["TP" if x and y else "FN" if ((not x) and y) else "TN" if ((not x) and (not y)) else "FP" for x,y in zip(df1['P'],df1['A'])]
    
    dictionary = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} | df1['E'].value_counts().to_dict()
    TP = dictionary['TP']
    FP = dictionary['FP']
    FN = dictionary['FN']
    TN = dictionary['TN']
    return((TP,FP,TN,FN))

def metric(TP, FP, TN, FN):
    ACC = (TP+TN)/(TP + FP + TN + FN)
    SENS = TP/(TP+FN)
    SELECT = TN/(FP+TN)
    PPV = TP/(TP+FP)
    FNR = FN/(TP+FN)
    FDR = FP/(FP+TP)
    FOR = FN/(FN+TN)
    # print("Predicted    0   1")
    # print("Actual")
    # print("0            " + str(TN) + "   " + str(FP))
    # print("1            " + str(FN) + "   " + str(TP))
    print('TP: ' + str(TP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))
    print('FP: ' + str(FP))
    print('Accuracy: ' + str(ACC))
    print('Sensitivity: ' + str(SENS))
    print('Specificity: ' + str(SELECT))
    print('Precision: ' + str(PPV))
    print('Miss Rate: ' + str(FNR))
    print('False Discovery Rate: ' + str(FDR))
    print('False Omission Rate: ' + str(FOR))
    return((ACC, SENS, SELECT, PPV, FNR, FDR, FOR))

df = pd.read_csv('mutations_1.csv')
df = df.T
df.columns = df.iloc[0]
df = df.drop(df.index[0])

df1 = df.sample(frac=0.33, random_state=1,axis=1)
df2 = df.drop(columns=df1.columns).sample(frac=0.5, random_state=1, axis=1)
df3 = df.drop(columns=df1.columns).drop(columns=df2.columns)


f1 = classifier((df.drop(columns=df1.columns)))
eval1 = evaluator(df1, f1[0], f1[1])
metric1 = metric(eval1[0],eval1[1],eval1[2],eval1[3])

f2 = classifier((df.drop(columns=df2.columns)))
eval2 = evaluator(df2, f2[0], f2[1])
metric2 = metric(eval2[0],eval2[1],eval2[2],eval2[3])

f3 = classifier((df.drop(columns=df3.columns)))
eval3 = evaluator(df3, f3[0], f3[1])
metric3 = metric(eval3[0],eval3[1],eval3[2],eval3[3])


ACC = (metric1[0] + metric2[0] + metric3[0])/3
SENS = (metric1[1] + metric2[1] + metric3[1])/3
SELECT = (metric1[2] + metric2[2] + metric3[2])/3
PPV = (metric1[3] + metric2[3] + metric3[3])/3
FNR = (metric1[4] + metric2[4] + metric3[4])/3
FDR = (metric1[5] + metric2[5] + metric3[5])/3
FOR = (metric1[6] + metric2[6] + metric3[6])/3
print(f1)
print(f2)
print(f3)
print('Accuracy: ' + str(ACC))
print('Sensitivity: ' + str(SENS))
print('Specificity: ' + str(SELECT))
print('Precision: ' + str(PPV))
print('Miss Rate: ' + str(FNR))
print('False Discovery Rate: ' + str(FDR))
print('False Omission Rate: ' + str(FOR))


