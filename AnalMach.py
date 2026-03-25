from operator import truediv
from os.path import exists
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os

paser = argparse.ArgumentParser()
paser.add_argument('-in','--inputNOE', help='input file, z NOE', required=False)
paser.add_argument('-ik','--inputKat', help='input file z Katami', required=False)
paser.add_argument('-s','--step', type=int ,default=1 ,help='co ile klatek ma brak program aby utworzyć wykres', required=False)
paser.add_argument('-o','--output', help='output file', required=False)
args = paser.parse_args()
# Tworzenie folderu z wszystkimi wykresami
os.makedirs(args.output, exist_ok = True)
# Ładowanie plików

df_plik=pd.read_csv(args.inputNOE, sep=';')
df_plik.columns=df_plik.columns.str.strip()
n_klt=df_plik['Klatka']
ok_cnt=df_plik['%OK']
dk=pd.read_csv(args.inputKat, sep=';')
dk=dk.iloc[::args.step, :].reset_index(drop=True)
surowe_x = dk['Klatka'].astype(str).str.replace(',', '.')
x_data_k = pd.to_numeric(surowe_x, errors='coerce')
kolumny_do_analizy = [col for col in dk.columns if col != 'Klatka']
d_ma={"Klt":np.array(n_klt),"%OK":np.array(ok_cnt)}
df_ma_data=pd.DataFrame(data=ok_cnt)

# MA - calculation

ma=df_ma_data.rolling(min_periods=1, window=500, center=True).mean()

# Histogram - z NOE
sciezka_noe_hist=os.path.join(args.output, 'histogram_noe.png')
plt.figure(figsize=(10, 10))
sns.histplot(ok_cnt,bins=20, color='blue', edgecolor='black', kde=True)
plt.xlabel('Procent zgodności więzów [%]')
plt.ylabel('Liczba klatek')
plt.title('Rozkład zgodności')
plt.savefig(sciezka_noe_hist, dpi=300)
plt.close()

# Dryf zgodności w czasie - z średnią kroczącą - z NOE
sciezka_noe_dryf=os.path.join(args.output, 'dryf_noe.png')
sns.lineplot(data=df_plik[::args.step], y='%OK', x='Klatka', color='blue')
plt.plot(ma, color='red')
plt.ylabel('Procent zgodności więzów [%]')
plt.xlabel('Liczba klatek')
plt.title('Zgodność w czasie')
plt.savefig(sciezka_noe_dryf, dpi=300)
plt.close()

# Wykresy z kątów
for nazwa_kolumny in kolumny_do_analizy:

    surowe_y = dk[nazwa_kolumny].astype(str).str.replace(',', '.')
    y_data = pd.to_numeric(surowe_y, errors='coerce')

    # Usuwamy ewentualne puste rzędy
    valid_idx = y_data.dropna().index
    y_plot = y_data.loc[valid_idx]
    x_plot = x_data_k.loc[valid_idx]
    if y_data.empty:
        continue

    # Tworzymy płótno: 1 wiersz, 2 kolumny (szerokość 14, wysokość 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- WYKRES LEWY: DRYF W CZASIE ---
    ax1.plot(x_plot, y_plot, color='#1f77b4', linewidth=1.2, alpha=0.85)
    ax1.set_title(f"Przebieg w czasie: {nazwa_kolumny}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Klatka symulacji", fontsize=10)
    ax1.set_ylabel("Wartość", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # WYMUSZENIE MAKSYMALNIE 8 ZNACZNIKÓW NA OSIACH (żeby tekst się nie zlał!)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # --- WYKRES PRAWY: HISTOGRAM ---
    ax2.hist(y_plot, bins=40, color='#2ca02c', edgecolor='black', alpha=0.75)
    ax2.set_title(f"Rozkład (Histogram): {nazwa_kolumny}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Wartość", fontsize=10)
    ax2.set_ylabel("Częstość (liczba klatek)", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Ogranicznik osi X dla histogramu
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8))

    # 6. Bezpieczny zapis do pliku
    # Zamieniamy wszystkie dziwne znaki w nazwie kolumny na podkreślniki, by system to przełknął
    nazwa_pliku = "".join([c if c.isalnum() else "_" for c in nazwa_kolumny])

    sciezka_wykresu = os.path.join(args.output, f"{nazwa_pliku}.png")
    plt.tight_layout()
    plt.savefig(sciezka_wykresu, dpi=150)  # Wysoka jakość do publikacji

    # Czyszczenie pamięci (bez tego skrypt zawiesi komputer po 50 wykresach)
    plt.close(fig)
