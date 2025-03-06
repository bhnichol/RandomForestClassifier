import pandas as pd
import numpy as np
import math
import re
pd.options.mode.chained_assignment = None

def classifier(df):
    dfc = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
    dfnc = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
    df1 = df.copy()
    df2 = df.copy()
    nT = len(df1.columns)
    nC = len(dfc.columns)
    nNC = len(dfnc.columns)

    df1['n(TL)'] = df1.sum(axis=1).astype('float')
    df1['n(TR)'] = nT - df1['n(TL)'].astype('float')
    df1['n(TL, C)'] = dfc.sum(axis=1).astype('float')
    df1['n(TL, NC)'] = df1['n(TL)'] - df1['n(TL, C)']
    df1['n(TR, C)'] = nC - df1['n(TL, C)']
    df1['n(TR, NC)'] = nNC - df1['n(TL, NC)']
    df1['PL'] = df1['n(TL)'] / nT
    df1['PR'] = df1['n(TR)'] / nT

    df1['P(C|TL)'] = df1['n(TL, C)'].divide(df1['n(TL)']).replace(np.inf, 0)
    df1['P(NC|TL)'] = df1['n(TL, NC)'].divide(df1['n(TL)']).replace(np.inf, 0)
    df1['P(C|TR)'] = df1['n(TR, C)'].divide(df1['n(TR)']).replace(np.inf, 0)
    df1['P(NC|TR)'] = df1['n(TR, NC)'].divide(df1['n(TR)']).replace(np.inf, 0)
    df1['H(TL)'] = -(df1['P(C|TL)']*(np.log2(df1['P(C|TL)'].replace(0, np.nan)))+df1['P(NC|TL)']*(np.log2(df1['P(NC|TL)'].replace(0, np.nan))))
    df1['H(TR)'] = -(df1['P(C|TR)']*(np.log2(df1['P(C|TR)'].replace(0, np.nan)))+df1['P(NC|TR)']*(np.log2(df1['P(NC|TR)'].replace(0, np.nan))))
    HT = -((nC/nT)*(np.log2(nC/nT))+(nNC/nT)*(np.log2(nNC/nT)))
    df1['H(s,t)'] = (df1['PL']*df1['H(TL)'])+(df1['PR']*df1['H(TR)'])
    df1['gain'] = HT - df1['H(s,t)']

    df1 = df1.nlargest(10, 'gain')

    df2 = df.copy()
    df2['gain'] = df1['gain']
    df2 = df2.nlargest(1, 'gain')
    df2 = df2.drop('gain', axis=1)

    df1 = df1[df1.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
    df1 = df1[df1.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
    
    dfB = df2.loc[:, ~(df2 == 1).any()]
    dfA = df2.loc[:, ~(df2 == 0).any()]

    dfc = dfA[dfA.columns.drop(list(dfA.filter(regex='^NC[0-9]+$')))]
    dfnc = dfA[dfA.columns.drop(list(dfA.filter(regex='^C[0-9]+$')))]
    if len(dfc.columns) > len(dfnc.columns):
        AEval = True
    else:
        AEval = False
    dfc = dfB[dfB.columns.drop(list(dfB.filter(regex='^NC[0-9]+$')))]
    dfnc = dfB[dfB.columns.drop(list(dfB.filter(regex='^C[0-9]+$')))]
    if len(dfc.columns) > len(dfnc.columns):
        BEval = True
    else:
        BEval = False
    return((list(dfB.index.values)[0], dfA.columns, dfB.columns, AEval, BEval))


def evaluator(df, f1, f2, f3, AEval, A2Eval, BEval, B2Eval):
    df1 = df.copy()
    df1 = df1.loc[[f1,f2,f3],:]
    df1 = df1.T
    df1[df1.columns[0]] = df1[df1.columns[0]].astype('bool')
    df1[df1.columns[1]] = df1[df1.columns[1]].astype('bool')
    df1[df1.columns[2]] = df1[df1.columns[2]].astype('bool')
    df1['P']= (df1[df1.columns[0]] & df1[df1.columns[1]] & AEval) | ((~df1[df1.columns[0]]) & df1[df1.columns[2]] & BEval) | (df1[df1.columns[0]] & (~df1[df1.columns[1]]) & A2Eval) | ((~df1[df1.columns[0]]) & (~df1[df1.columns[2]]) & B2Eval) 
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

print("FOLD 1:")
root1 = classifier((df.drop(columns=df1.columns)))
left1 = classifier((df.drop(columns=df1.columns).drop(columns=root1[2])))
right1 = classifier((df.drop(columns=df1.columns).drop(columns=root1[1])))
eval1 = evaluator(df1, root1[0], left1[0], right1[0], left1[3], left1[4], right1[3], right1[4])
metric1 = metric(eval1[0],eval1[1],eval1[2],eval1[3])
print(root1[0])
print(left1[0])
print(right1[0])

print("FOLD 2:")
root2 = classifier((df.drop(columns=df2.columns)))
left2 = classifier((df.drop(columns=df2.columns).drop(columns=root2[2])))
right2 = classifier((df.drop(columns=df2.columns).drop(columns=root2[1])))
eval2 = evaluator(df2, root2[0], left2[0], right2[0], left2[3], left2[4], right2[3], right2[4])
metric2 = metric(eval2[0],eval2[1],eval2[2],eval2[3])
print(root2[0])
print(left2[0])
print(right2[0])

print("FOLD 3:")
root3 = classifier((df.drop(columns=df3.columns)))
left3 = classifier((df.drop(columns=df3.columns).drop(columns=root3[2])))
right3 = classifier((df.drop(columns=df3.columns).drop(columns=root3[1])))
eval3 = evaluator(df3, root3[0], left3[0], right3[0], left3[3], left3[4], right3[3], right3[4])
metric3 = metric(eval3[0],eval3[1],eval3[2],eval3[3])
print(root3[0])
print(left3[0])
print(right3[0])

print("Average Statistics:")
ACC = (metric1[0] + metric2[0] + metric3[0])/3
SENS = (metric1[1] + metric2[1] + metric3[1])/3
SELECT = (metric1[2] + metric2[2] + metric3[2])/3
PPV = (metric1[3] + metric2[3] + metric3[3])/3
FNR = (metric1[4] + metric2[4] + metric3[4])/3
FDR = (metric1[5] + metric2[5] + metric3[5])/3
FOR = (metric1[6] + metric2[6] + metric3[6])/3
print('Accuracy: ' + str(ACC))
print('Sensitivity: ' + str(SENS))
print('Specificity: ' + str(SELECT))
print('Precision: ' + str(PPV))
print('Miss Rate: ' + str(FNR))
print('False Discovery Rate: ' + str(FDR))
print('False Omission Rate: ' + str(FOR))