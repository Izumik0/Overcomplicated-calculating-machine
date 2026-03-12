import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os

paser = argparse.ArgumentParser()
paser.add_argument('-i','--input', help='input file, z użycia SMM.py', required=True)
paser.add_argument('-o','--output', help='output file', required=False)

# Ładowanie plików
args = paser.parse_args()
df_plik=pd.read_csv(args.input, sep=';')
df_plik.columns=df_plik.columns.str.strip()
n_klt=df_plik['Klatka']
ok_cnt=df_plik['%OK']


d_ma={"Klt":np.array(n_klt),"%OK":np.array(ok_cnt)}
df_ma_data=pd.DataFrame(data=ok_cnt)

# MA - calculation

ma=df_ma_data.rolling(min_periods=1, window=1000).mean()

# Histogram
plt.figure(figsize=(10, 10))
sns.histplot(ok_cnt,bins=20, color='blue', edgecolor='black', kde=True)
plt.xlabel('Procent zgodności więzów [%]')
plt.ylabel('Liczba klatek')
plt.title('Rozkład zgodności')
plt.show()

# Dryf zgodności w czasie - z średnią kroczącą
sns.lineplot(data=df_plik[::20], y='%OK', x='Klatka', color='blue')
plt.plot(ma, color='red')
plt.ylabel('Procent zgodności więzów [%]')
plt.xlabel('Liczba klatek')
plt.title('Zgodność w czasie')
plt.show()