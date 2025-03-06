import pandas as pd
import numpy as np
from collections import Counter
import math
import emoji
pd.options.mode.chained_assignment = None

NUM_TREES = 31
TREE_DEPTH = 1

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


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.data = None
        self.choice = None
    def print_tree(self):
        print(self.data, self.choice)
        if self.left != None:
            self.left.print_tree()
        if self.right != None:
            self.right.print_tree()
    def evaluator(self, sample, df):
        if self.right.data != None and self.left.data != None:
            if bool(df.loc[self.data, sample]):
                return self.left.evaluator(sample,df)
            else:
                return self.right.evaluator(sample,df)
        else:
            return self.choice

        


def classifier(df, depth, maxDepth, node):
    dfc = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
    dfnc = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
    df1 = df.copy()
    df1 = df1.sample(len(df1.index), axis=0, replace=False)
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

    node.data = list(dfB.index.values)[0]
    node.left = Node()
    node.left.choice = AEval
    node.right = Node()
    node.right.choice = BEval

    if depth == maxDepth:
        return
    else:
        classifier(df.drop(columns=dfA.columns), depth+1, maxDepth, node.left)
        classifier(df.drop(columns=dfB.columns), depth+1, maxDepth, node.right)

# def evaluator(sample, df, node):
#     print("START EVAL")
#     temp = node
#     while temp.right.data != None and temp.left.data != None:
#         print(temp.left.data)
#         print(temp.right.data)
#         if bool(df.loc[temp.data, sample]):
#             temp = node.left
#         else:
#             temp = node.right
    
#     return temp.choice

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
    if FN+TN == 0:
        FOR = 0
    else:
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

df = pd.read_csv('mutations_2.csv')
df = df.T
df.columns = df.iloc[0]
df = df.drop(df.index[0])

bootstrap = []
outofbag = []
trees = []
print("CREATING BOOT AND BAGS")
for i in range(NUM_TREES):
    dfboot = df.copy()
    dfbag = df.copy()
    dfboot = dfboot.sample(len(dfboot.columns), replace=True, axis=1)
    dfbag = dfbag.drop(dfboot.columns, axis=1)
    bootstrap.append(dfboot)
    outofbag.append(dfbag)
print("CREATING TREES")
for sample in bootstrap:
    root = Node()
    classifier(sample,0,TREE_DEPTH,root)
    trees.append(root)
    #root.print_tree()

print(color.BOLD + "Forest Summary:" + color.END)
features = []
for tree in trees:
    treeStack = []
    currNode = tree
    while treeStack or currNode.data != None:
        if currNode.data != None:
            features.append(currNode.data)
            treeStack.append(currNode)
            currNode = currNode.left
        else:
            prevNode = treeStack.pop()
            currNode = prevNode.right

features = Counter(features)
features = features.most_common()
print(color.YELLOW + color.BOLD + color.UNDERLINE + "Features" + color.END)
for key, value in features:
    print(color.YELLOW + key + color.END + ": ", value)



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
print("EVALUATING OUT OF BAG")
Predicted = False
for sample in outofbag[0].columns:
    counterC = 0
    counterNC = 0  
    if 'NC' in sample:
        Actual = False
    else:
        Actual = True
    for root in trees:
        if root.evaluator(sample,df):
            counterC+=1
        else:
            counterNC+=1

    if counterC >= round(0.17*NUM_TREES):
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

metrics = metric(TP , FP, TN, FN)

sample = input("Enter Sample Name (Q to exit): ")
while sample != 'Q':
    if sample in df:
        counterC = 0
        counterNC = 0
        for root in trees:
            # Predicted = (bool(df.loc[tree[0], sample]) and bool(df.loc[tree[1], sample]) and tree[3]) or (bool(df.loc[tree[0], sample]) and ~(bool(df.loc[tree[1], sample])) and tree[4]) or (~(bool(df.loc[tree[0], sample])) and bool(df.loc[tree[2], sample]) and tree[5]) or (~(bool(df.loc[tree[0], sample])) and ~(bool(df.loc[tree[2], sample])) and tree[6])
            if root.evaluator(sample,df):
                counterC+=1
            else:
                counterNC+=1
                
        print(counterC)
        print(counterNC)
        if counterC > round(0.17*counterNC):
            print(sample + " is predicted to Cancer", "\U0001F972")
        else:
            print(sample + " is predicted to Non-Cancer", emoji.emojize(':thumbs_up:'))
    else:
        print("Sample does not exist in dataset!")
    sample = input("Enter Sample Name (Q to exit): ")
