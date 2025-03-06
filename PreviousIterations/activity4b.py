import pandas as pd
import numpy as np
#142 NC
#129 C
df = pd.read_excel('mutations.xlsx')
df = df.T
df.columns = df.iloc[0]
df = df.drop(df.index[0])
dfc = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
dfnc = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]

df['TP'] = dfc.sum(axis=1)
df['FP'] = dfnc.sum(axis=1)
df['TP-FP'] = df['TP'] - df['FP']

df10 = df.nlargest(10, 'TP-FP')
df10 = df10[df10.columns.drop(list(df10.filter(regex='^NC[0-9]+$')))]
df10 = df10[df10.columns.drop(list(df10.filter(regex='^C[0-9]+$')))]

df = df.nlargest(1, 'TP-FP')

dfB = df.loc[:, ~(df == 1).any()]
dfA = df.loc[:, ~(df == 0).any()]

df = df.drop(['TP', 'FP', 'TP-FP'], axis=1)
df.loc[len(df)] = df.columns
df.columns = range(len(df.columns))
df = df.replace(to_replace='^C',value = 1, regex=True)
df = df.replace(to_replace='^N',value = 0, regex=True)
df = df.T

confusion_matrix1 = pd.crosstab(df[1], df['RNF43_GRCh37_17:56435161-56435161_Frame-Shift-Del_DEL_C-C--'], rownames=['Actual'], colnames=['Predicted'])
print(df10)
print(dfA)
print(dfB)
print(confusion_matrix1)

# writer = pd.ExcelWriter("activity4b.xlsx", engine="xlsxwriter")

# df10.to_excel(writer, sheet_name="Sheet1")
# dfA.to_excel(writer, sheet_name="Sheet2")
# dfB.to_excel(writer, sheet_name="Sheet3")

# writer.close()