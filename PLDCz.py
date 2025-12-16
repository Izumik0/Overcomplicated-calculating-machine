import argparse
import os
import sys
import itertools
import numpy as np
import pandas as pd
import mdtraj

# /////////////////////////////////Argumenty dla konsoli////////////////////////////////////////////////////

parser = argparse.ArgumentParser(description='Obliczanie więzów między atomami w strukturach DNA według informacji z NOE')

#utworzenie flag dla programu
parser.add_argument('-s', '--struct', type=str, required=True, help='Ścieżka do pliku struktury (np. structure.pdb)')
parser.add_argument('-n', '--noe',    type=str, required=True, help='Ścieżka do pliku z danymi NOE (.txt)')
parser.add_argument('-o', '--out',    type=str, default='wyniki_noe', help='Nazwa folderu na wyniki (domyślnie: wyniki_noe)')

args = parser.parse_args()

#załadowanie pliku .pdb
traj=mdtraj.load(args.struct)

#Plik NOE lokalizacje
#    (0) - Nr. nukleotydu na którym jest pierwszy atom z pary
#    (1) - Pierwszy atom z pary
#    (2) - Nr. nukelotydu na którym jest drugi atom z pary
#    (3) - Drugi atom z pary
#    (4) - Wartość odległości eksperymentalna
#    (5) - Wartość odległości teoretyczna I think

#Wczytanie pliku ze strukturą z pliku .pdb - robocze póki co
kolumny_pdb = [
    (0, 6), #Kolumna z typem co to jest czyli w naszym przypadku przeszukujemy ATOMS
    (6, 11), #indeks atomu
    (12, 16), #nazwa atomu
    (22, 26), #nr. nukleotydu
]
nazwy_kolumn_pdb=['Rekord', 'Indeks', 'Atom', 'Nr. Nukleotydu']
dpdb = pd.read_fwf(args.struct, colspecs=kolumny_pdb, names=nazwy_kolumn_pdb)
dpdb = dpdb[dpdb['Rekord'].isin(['ATOM'])]

#czyszczenie danych - wrzucamy na dane na str zeby ładnie porównywać
dpdb['Atom'] = dpdb['Atom'].str.strip()
dpdb['Nr. Nukleotydu'] = dpdb['Nr. Nukleotydu'].str.strip()


#tworzenie mapy z pogrupowanymi indeksami atomów
mapa_indeksow=dpdb.groupby(['Nr. Nukleotydu', 'Atom'])['Indeks'].apply(list).to_dict()
#print(mapa_indeksów)

#tworzenie folderu do którego wszystkie pliki z macierzami będą wrzucone
os.makedirs(args.out, exist_ok=True)

pary=[]

with (open(args.noe, 'r', encoding='utf-8') as NOE_file):
    for line in NOE_file:
        nr_N_1 = line.split()[0]
        atom_1 = line.split()[1]
        nr_N_2 = line.split()[2]
        atom_2 = line.split()[3]
        war_porow_str=line.split()[5]

        #the black box - czyli sobie znajduje, kombinuje i liczy odległości
        indeks_1=np.array(mapa_indeksow.get((nr_N_1, atom_1), [])) #znajdujemy indeksy dla pierwszego atomu
        indeks_2=np.array(mapa_indeksow.get((nr_N_2, atom_2), [])) #znajdujey indeksy dla drugiego atomu
        pary_str= itertools.product(indeks_1, indeks_2)
        pary_maciez=np.array(list(pary_str))
        pary=pary_maciez.astype(int)-1 #i znów na int (+ jest tu łatanie duck tape'em)
        war_porow_f=float(war_porow_str)

        #oblicza odległości na podstawie par które zostały utworzone
        #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
        odleglosci=mdtraj.compute_distances(traj, pary) # tu jest odległość w nm
        odleglosci_a= odleglosci*10 #odległości w arstrongach czy jak się to nazywa

        #print("Odległości między atomami:\n", "Nr. Nukleotydu:", nr_N_1, "Atom:", atom_1, "Nr. nuk:", nr_N_2, "atom", atom_2, "\n", odleglosci_a)

        #przygotowanie wyników do zapisu
        wynik_do_zapisu=odleglosci_a.reshape(len(indeks_1), len(indeks_2))
        roznica_od_exp=(wynik_do_zapisu-war_porow_f)
        #print(" Rożnice Odległości między atomami:\n", "Nr. Nukleotydu:", nr_N_1, "Atom:", atom_1, "Nr. nuk:", nr_N_2, "atom", atom_2, "\n", roznica_od_exp)

        #zapis macierzy do pliku i robienie kolorków
        etyk_wierszy=[f"{nr_N_1}_{atom_1}_{i}" for i in indeks_1]
        etyk_kolumn=[f"{nr_N_2}_{atom_2}_{i}" for i in indeks_2]

        df_zapis_sim= pd.DataFrame(wynik_do_zapisu, index=etyk_wierszy, columns=etyk_kolumn)
        df_zapis_roz= pd.DataFrame(roznica_od_exp, index=etyk_wierszy, columns=etyk_kolumn)

        nazwa_pliku=f"{nr_N_1}_{atom_1}_vs_{nr_N_2}_{atom_2}.xlsx"
        sciezka=os.path.join(args.out, nazwa_pliku)

        with pd.ExcelWriter(sciezka, engine='xlsxwriter') as writer:
            #zapis do Excela
            df_zapis_sim.to_excel(writer, sheet_name='Symulacja (nm)')
            df_zapis_roz.to_excel(writer, sheet_name='Exp_diff')

            #Załadowanie skoroszytu
            workbook = writer.book
            worksheet_roz = writer.sheets['Exp_diff']

            #jak wygląda formatowanie
            format_g = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

            #ustawienie tego jakie rzeczy podlegają formatowaniu
            max_row = len(df_zapis_roz)
            max_col = len(df_zapis_roz.columns)

            #formatowanie
            worksheet_roz.conditional_format(1, 1, max_row, max_col, {
                'type': 'cell',
                'criteria': 'between',
                'minimum': -1,
                'maximum': 1,
                'format': format_g
            })


        print(f"Zadanie ukończono, macierze zapisano w: {args.out}")

#liczy zajebiście bo porównałem z VMD i wmiare pasuje to wszystko - ale potem więcej posprawdzam tych par atomów
#Zapis do pliku też jest git ja poprostu nie na tą pare spojrzałem ;p