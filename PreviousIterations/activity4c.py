import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
#142 NC
#129 C
df = pd.read_excel('mutations.xlsx')
df = df.T
df.columns = df.iloc[0]
df = df.drop(df.index[0])

dfA = df[['NC1', 'C24', 'C25', 'C27', 'C28', 'C29', 'C41', 'C66', 'C84', 'C102', 'C104', 'C123', 'C124']]
dfB = df[df.columns[~df.columns.isin(['NC1', 'C24', 'C25', 'C27', 'C28', 'C29', 'C41', 'C66', 'C84', 'C102', 'C104', 'C123', 'C124'])]]

#Group A
dfAc = dfA[dfA.columns.drop(list(dfA.filter(regex='^NC[0-9]+$')))]
dfAnc = dfA[dfA.columns.drop(list(dfA.filter(regex='^C[0-9]+$')))]

dfA['TP'] = dfAc.sum(axis=1)
dfA['FP'] = dfAnc.sum(axis=1)
dfA['TP-FP'] = dfA['TP'] - dfA['FP']
dfA['TP-FP']=dfA['TP-FP'].astype('int')
dfA10 = dfA.nlargest(10, 'TP-FP')
dfA = dfA.nlargest(1, 'TP-FP')

dfA2 = dfA.loc[:, ~(dfA == 1).any()]
dfA1 = dfA.loc[:, ~(dfA == 0).any()]

dfA = dfA.drop(['TP', 'FP', 'TP-FP'], axis=1)
dfA.loc[len(dfA)] = dfA.columns
dfA.columns = range(len(dfA.columns))
dfA = dfA.replace(to_replace='^C',value = 1, regex=True)
dfA = dfA.replace(to_replace='^N',value = 0, regex=True)
dfA = dfA.T
confusion_matrix1 = pd.crosstab(dfA[1], dfA['RNF43_GRCh37_17:56435161-56435161_Frame-Shift-Del_DEL_C-C--'], rownames=['Actual'], colnames=['Predicted'])

# print(confusion_matrix1)

#Group B
dfBc = dfB[dfB.columns.drop(list(dfB.filter(regex='^NC[0-9]+$')))]
dfBnc = dfB[dfB.columns.drop(list(dfB.filter(regex='^C[0-9]+$')))]

dfB['TP'] = dfBc.sum(axis=1)
dfB['FP'] = dfBnc.sum(axis=1)
dfB['TP-FP'] = dfB['TP'] - dfB['FP']
dfB['TP-FP']=dfB['TP-FP'].astype('int')
dfB10 = dfB.nlargest(10, 'TP-FP')
dfB = dfB.nlargest(1, 'TP-FP')

dfB2 = dfB.loc[:, ~(dfB == 1).any()]
dfB1 = dfB.loc[:, ~(dfB == 0).any()]
print(dfB1)
dfB = dfB.drop(['TP', 'FP', 'TP-FP'], axis=1)
dfB.loc[len(dfB)] = dfB.columns
dfB.columns = range(len(dfB.columns))
dfB = dfB.replace(to_replace='^C',value = 1, regex=True)
dfB = dfB.replace(to_replace='^N',value = 0, regex=True)
dfB = dfB.T
confusion_matrix2 = pd.crosstab(dfB[1], dfB['PTEN_GRCh37_10:89692905-89692905_Missense-Mutation_SNP_G-G-A_G-G-T_G-G-C'], rownames=['Actual'], colnames=['Predicted'])

# print(confusion_matrix2)



# writer = pd.ExcelWriter("activity4c.xlsx", engine="xlsxwriter")

# dfA10.to_excel(writer, sheet_name="Sheet1")
# dfB10.to_excel(writer, sheet_name="Sheet2")

# writer.close()


sample = input("Enter Sample Name (Q to exit): ")
while sample != 'Q':
    if sample in dfA1.columns or sample in dfB1.columns:
        print(sample + " is predicted to have cancer.")
    elif sample in dfA2.columns or sample in dfB2.columns:
        print(sample + " is predicted to be non-cancerous.")
    else:
        print(sample + " Not in data set!")
    sample = input("Enter Sample Name (Q to exit): ")

