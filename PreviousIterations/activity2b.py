import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('mutations.xlsx')
# df['MPS'] = df.sum(axis=1)
# df['MPS'] = df['MPS'].astype('int')
# df.plot.scatter(x='class', y='MPS')
df = df.T
df.columns = df.iloc[0]
df = df.drop(df.index[0])
df['T'] = df.sum(axis=1)
df['T'] = df['T'].astype('int')
# df['class'] = df.index
# df.plot.scatter(x='class',y='T')
# plt.show()
df['C'] = df[df.filter(regex=('^C'),axis=1).columns].sum(axis=1)
df['NC'] = df[df.filter(regex=('^N'), axis=1).columns].sum(axis=1)
df['%C'] = df['C'].div(len(df.filter(regex=('^C'), axis=1).columns))
df['%NC'] = df['NC'].div(len(df.filter(regex=('^N'), axis=1).columns))
df['%C - %NC'] = df['%C'] - df['%NC']
df['%C / %NC'] = df['%C'] / df['%NC']

df['C'] = df['C'].astype('int')
df['NC'] = df['NC'].astype('int')

df = df[df.columns.drop(list(df.filter(regex='^NC[0-9]+$')))]
df = df[df.columns.drop(list(df.filter(regex='^C[0-9]+$')))]
df.replace([np.inf, -np.inf], 0, inplace=True)

Table1 = df.nlargest(10,'T')
Table2 = df.nlargest(10, 'C')
Table3 = df.nlargest(10, 'NC')
Table4 = df.nlargest(10, '%C')
Table5 = df.nlargest(10, '%NC')
Table6 = df.nlargest(10, '%C - %NC')
Table7 = df.nlargest(10, '%C / %NC')

# writer = pd.ExcelWriter("activity2b.xlsx", engine="xlsxwriter")

# Table1.to_excel(writer, sheet_name="Sheet1")
# Table2.to_excel(writer, sheet_name="Sheet2")
# Table3.to_excel(writer, sheet_name="Sheet3")
# Table4.to_excel(writer, sheet_name="Sheet4")
# Table5.to_excel(writer, sheet_name="Sheet5")
# Table6.to_excel(writer, sheet_name="Sheet6")
# Table7.to_excel(writer, sheet_name="Sheet7")

# writer.close()