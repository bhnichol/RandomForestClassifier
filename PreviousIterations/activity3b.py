import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('mutations.xlsx')
df = df.T
# df.columns = df.iloc[0]
# df = df.drop(df.index[0])
df = df.replace(to_replace='^C',value = 1, regex=True)
df = df.replace(to_replace='^N',value = 0, regex=True)
df['T'] = df.sum(axis=1)
df['T'] = df['T'].astype('int')
df = df.nlargest(3,'T')
df1 = df.drop(['KRAS_GRCh37_12:25398284-25398284_Missense-Mutation_SNP_C-C-A_C-C-T_C-C-G'])
df1 = df1[df1.columns.drop(['T'])]
df2 = df.drop(['PPP2R1A_GRCh37_19:52715971-52715971_Missense-Mutation_SNP_C-C-G_C-C-T'])
df2 = df2[df2.columns.drop(['T'])]
df1 = df1.T
df2 = df2.T
confusion_matrix1 = pd.crosstab(df1['class'], df1['PPP2R1A_GRCh37_19:52715971-52715971_Missense-Mutation_SNP_C-C-G_C-C-T'], rownames=['Actual'], colnames=['Predicted'])
confusion_matrix2 = pd.crosstab(df2['class'], df2['KRAS_GRCh37_12:25398284-25398284_Missense-Mutation_SNP_C-C-A_C-C-T_C-C-G'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix1)
print(confusion_matrix2)
TP1 = 0
FP1 = 24
PercentTP1 = TP1 / 271
PercentFP1 = FP1 / 271
TPFP1 = TP1 - FP1
print(TPFP1)
print(PercentTP1-PercentFP1)

TP2 = 16
FP2 = 6
PercentTP2 = TP2 / 271
PercentFP2 = FP2 / 271
TPFP2 = TP2-FP2
print(TPFP2)
print(PercentTP2-PercentFP2)

X = ['TP,FP', 'TN,FN']
# y11 = [0, 118]
# y12 = [24, 129]
# plt.bar(X, y11, color='r')
# plt.bar(X, y12, bottom=y11, color='b')
# plt.show()

# y11 = [16, 136]
# y12 = [6, 113]
# plt.bar(X, y11, color='r')
# plt.bar(X, y12, bottom=y11, color='b')
# plt.show()
colors = ['#FF0000', '#0000FF', '#FFFF00',
          '#ADFF2F', '#FFA500']
Labels = ['TP', 'FP', 'TN', 'FN']
# Values = [0, 24, 118, 129]
# plt.pie(Values, colors=colors, labels=Labels,
#         autopct='%1.1f%%', pctdistance=0.85)
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.title('PPP2R1A_GRCh37_19:52715971-52715971_Missense-Mutation_SNP_C-C-G_C-C-T')
# plt.show()

# Values = [16, 6, 136, 113]
# plt.pie(Values, colors=colors, labels=Labels,
#         autopct='%1.1f%%', pctdistance=0.85)
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.title('KRAS_GRCh37_12:25398284-25398284_Missense-Mutation_SNP_C-C-A_C-C-T_C-C-G')
# plt.show()








