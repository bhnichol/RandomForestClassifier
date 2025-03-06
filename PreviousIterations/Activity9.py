import pandas as pd
import numpy as np
from collections import Counter
import math
import re
import emoji
pd.options.mode.chained_assignment = None

# Change this variable to change size of forest
NUM_TREES = 25


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def classifier(df):
    dfc = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
    dfnc = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
    df1 = df.copy()
    df1 = df1.sample(round(math.sqrt(len(df1.index))), axis=0, replace=False)
    df2 = df1.copy()
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
    df1['Q(s|t)'] = abs(df1['P(C|TL)'] - df1['P(C|TR)']) + abs(df1['P(NC|TL)'] - df1['P(NC|TR)'])
    df1['Phi'] = 2 * df1['PL'] * df1['PR'] * df1['Q(s|t)']
    df1 = df1.nlargest(10, 'Phi')
    df2['Phi'] = df1['Phi']
    df2 = df2.nlargest(1, 'Phi')
    df2 = df2.drop('Phi', axis=1)

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
    if TP+FP == 0:
        PPV = 0
        FDR = 0
    else:
        PPV = TP/(TP+FP)
        FDR = FP/(FP+TP)
    SENS = TP/(TP+FN)
    FNR = FN/(TP+FN)
    SELECT = TN/(FP+TN)
    
    FNR = FN/(TP+FN)
    
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

bootstrap = []
outofbag = []
trees = []
roots = []
lefts = []
rights = []

for i in range(NUM_TREES):
    dfboot = df.copy()
    dfbag = df.copy()
    dfboot = dfboot.sample(len(dfboot.columns), replace=True, axis=1)
    dfbag = dfbag.drop(dfboot.columns, axis=1)
    bootstrap.append(dfboot)
    outofbag.append(dfbag)

for sample in bootstrap:
    root = classifier(sample)
    left = classifier(sample.drop(columns=root[2]))
    right = classifier(sample.drop(columns=root[1]))
    trees.append((root[0], left[0], right[0], left[3], left[4], right[3], right[4]))
    roots.append(root[0])
    lefts.append(left[0])
    rights.append(right[0])

print(color.BOLD + "Forest Summary:" + color.END)
roots = Counter(roots)
lefts = Counter(lefts)
rights = Counter(rights)
print(color.YELLOW + color.BOLD + color.UNDERLINE + "Root Nodes" + color.END)
for key, value in roots.items():
    print(color.YELLOW + key + color.END , value)
print(color.RED + color.BOLD + color.UNDERLINE + "Right Nodes" + color.END)
for key, value in rights.items():
    print(color.RED + key + color.END, value)
print(color.GREEN + color.BOLD + color.UNDERLINE + "Right Nodes" + color.END)
for key, value in lefts.items():
    print(color.GREEN + key + color.END, value)
print(color.BOLD + "Out-Of-Bag Summary" + color.END)
print(color.BOLD + "Sizes:" + color.END)
length = 0
for bag in outofbag:
    print(bag.shape)
    length+= len(bag.columns)
length = length / len(outofbag)
print(color.BOLD + "Average Length: " + color.END, length)

TP = 0
FP = 0
TN = 0
FN = 0

Predicted = False
for sample in outofbag[0].columns:
    counterC = 0
    counterNC = 0  
    if 'NC' in sample:
        Actual = False
    else:
        Actual = True
    for tree in trees:
        # Predicted = (bool(df.loc[tree[0], sample]) and bool(df.loc[tree[1], sample]) and tree[3]) or (bool(df.loc[tree[0], sample]) and ~(bool(df.loc[tree[1], sample])) and tree[4]) or (~(bool(df.loc[tree[0], sample])) and bool(df.loc[tree[2], sample]) and tree[5]) or (~(bool(df.loc[tree[0], sample])) and ~(bool(df.loc[tree[2], sample])) and tree[6])
        if (bool(df.loc[tree[0], sample]) and bool(df.loc[tree[1], sample]) and tree[3]):
            counterC+=1
        elif (bool(df.loc[tree[0], sample]) and (not bool(df.loc[tree[1], sample])) and tree[4]):
            counterC+=1
        elif ((not bool(df.loc[tree[0], sample])) and bool(df.loc[tree[2], sample]) and tree[5]):
            counterC+=1
        elif ((not bool(df.loc[tree[0], sample])) and (not bool(df.loc[tree[2], sample])) and tree[6]):
            counterC+=1
        else:
            counterNC+=1
    
    if counterC >= counterNC:
        Predicted = True
    else:
        Predicted = False
    #print(sample, Actual, Predicted)
    if (Actual and Predicted):
        TP +=1
    elif ((not Actual) and Predicted):
        FP +=1
    elif ((not Actual) and (not Predicted)):
        TN +=1
    else:
        FN +=1

# print('TP: ' + str(TP))
# print('FN: ' + str(FN))
# print('TN: ' + str(TN))
# print('FP: ' + str(FP))
metrics = metric(TP , FP, TN, FN)
sample = input("Enter Sample Name (Q to exit): ")
while sample != 'Q':
    if sample in df:
        counterC = 0
        counterNC = 0
        for tree in trees:
            # Predicted = (bool(df.loc[tree[0], sample]) and bool(df.loc[tree[1], sample]) and tree[3]) or (bool(df.loc[tree[0], sample]) and ~(bool(df.loc[tree[1], sample])) and tree[4]) or (~(bool(df.loc[tree[0], sample])) and bool(df.loc[tree[2], sample]) and tree[5]) or (~(bool(df.loc[tree[0], sample])) and ~(bool(df.loc[tree[2], sample])) and tree[6])
            if (bool(df.loc[tree[0], sample]) and bool(df.loc[tree[1], sample]) and tree[3]):
                counterC+=1
            elif (bool(df.loc[tree[0], sample]) and (not bool(df.loc[tree[1], sample])) and tree[4]):
                counterC+=1
            elif ((not bool(df.loc[tree[0], sample])) and bool(df.loc[tree[2], sample]) and tree[5]):
                counterC+=1
            elif ((not bool(df.loc[tree[0], sample])) and (not bool(df.loc[tree[2], sample])) and tree[6]):
                counterC+=1
            else:
                counterNC+=1
        print(counterC)
        print(counterNC)
        if counterC > counterNC:
            print(sample + " is predicted to Cancer", "\U0001F972")
        else:
            print(sample + " is predicted to Non-Cancer", emoji.emojize(':thumbs_up:'))
    else:
        print("Sample does not exist in dataset!")
    sample = input("Enter Sample Name (Q to exit): ")



