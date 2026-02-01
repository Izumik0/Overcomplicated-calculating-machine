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

# Plik NOE lokalizacje
#    (0) - Nr. nukleotydu na którym jest pierwszy atom z pary
#    (1) - Pierwszy atom z pary
#    (2) - Nr. nukelotydu na którym jest drugi atom z pary
#    (3) - Drugi atom z pary
#    (4) - Wartość odległości eksperymentalna
#    (5) - Wartość odległości teoretyczna I think

# Wczytanie pliku ze strukturą z pliku .pdb - robocze póki co
kolumny_pdb = [
    (0, 6),  # Kolumna z typem co to jest czyli w naszym przypadku przeszukujemy ATOMS
    (6, 11),  # indeks atomu
    (12, 16),  # nazwa atomu
    (17, 20),  # jaki to nukleotyd (adeina, guanina czy inny)
    (22, 30),  # nr. cząsteczki
]


mapa_poprawek={
    "H5'":"1H5'", "H5''":"2H5'",
    "H2'":"1H2'", "H2''":"2H2'",
    "H5'1":"1H5'", "H5'2":"2H5'",
    "H2'1":"1H2'", "H2'2":"2H2'",
    "C5M":"C7", "H71":"H51",
    "H72":"H52", "H73":"H53",
}
nazwy_kolumn_pdb=['Rekord', 'Indeks', 'Atom', 'Nukleotyd' ,'Nr. Nukleotydu']
dpdb = pd.read_fwf(args.struct, colspecs=kolumny_pdb, names=nazwy_kolumn_pdb)
dpdb = dpdb[dpdb['Rekord'].isin(['ATOM'])]

#czyszczenie danych - wrzucamy na dane na str zeby ładnie porównywać
dpdb['Atom'] = dpdb['Atom'].astype(str).str.strip()
dpdb['Nr. Nukleotydu'] = dpdb['Nr. Nukleotydu'].astype(str).str.strip()
dpdb['Nukleotyd'] = dpdb['Nukleotyd'].astype(str).str.strip()
dpdb['Atom'] = dpdb['Atom'].replace(mapa_poprawek)

#usuwanie wszystkiego co nie jest Guaniną
dpdb = dpdb[dpdb['Nukleotyd'].isin(['DG'])]

#tworzenie mapy z pogrupowanymi indeksami atomów
mapa_indeksow=dpdb.groupby(['Nr. Nukleotydu', 'Atom'])['Indeks'].apply(list).to_dict()


#tworzenie folderu do którego wszystkie pliki z macierzami będą wrzucone
os.makedirs(args.out, exist_ok=True)

#Słownik do sprawdzania atomów bliźniaczych
atom_blizniaki = {
    # DNA / RNA (cukier)
    "1H2'": "2H2'", "2H2'": "1H2'",
    "H21":"H22", "H22":"H21",
    "1H5'":"2H5'", "2H5'":"1H5'",
    "1H2":  "2H2",  "2H2":  "1H2",  # Guanina
    "1H4":  "2H4",  "2H4":  "1H4",  # Cytozyna
    "1H6":  "2H6",  "2H6":  "1H6",  # Adenina

}
#szukanie bliźniaków i pobieranie dla nich danych z mapy
def pobierz_grupe_indeksow(nr_res, nazwa_atom, mapa):

    indeksy=list(mapa.get((nr_res, nazwa_atom), []))

    if nazwa_atom in atom_blizniaki:
        nazwa_blizniaka=atom_blizniaki[nazwa_atom]
        indeksy_blizniaka = list(mapa.get((nr_res, nazwa_blizniaka), []))
        indeksy.extend(indeksy_blizniaka)
    return list(set(indeksy))


#początek tworzenia trajektorii w VMD
nazwa_tcl = "visualize_NOE.tcl"
sciezka_tcl = os.path.join(args.out, nazwa_tcl)

f_tcl = open(sciezka_tcl, "w", encoding="utf-8")
f_tcl.write("# Skrypt wizualizacji par NOE dla VMD\n")
f_tcl.write("# Użycie w VMD: Extensions -> Tk Console -> wpisz: source visualize_NOE.tcl\n")


brak_par=[]
pary=[]
wyniki_lista=[]

with (open(args.noe, 'r', encoding='utf-8') as NOE_file):
    for line in NOE_file:
        nr_N_1 = line.split()[0]
        atom_1 = line.split()[1]
        nr_N_2 = line.split()[2]
        atom_2 = line.split()[3]
        war_porow_str=line.split()[5]

        #the black box - czyli sobie znajduje, kombinuje i liczy odległości
        indeks_1=np.array(pobierz_grupe_indeksow(nr_N_1, atom_1, mapa_indeksow)) #znajdujemy indeksy dla pierwszego atomu
        indeks_2=np.array(pobierz_grupe_indeksow(nr_N_2, atom_2, mapa_indeksow)) #znajdujey indeksy dla drugiego atomu

        #system pomijania par które się nie pojawiły i ich dokumentowania w terminalu(poki co)
        if len(indeks_1)==0 or len(indeks_2)==0:
            print(f"Nie można znaleźć pary: {nr_N_1}_{atom_1} i {nr_N_2}_{atom_2}, przechodze dalej do kolejnej pary z pliku")
            brak_par.append(f"Nie znaleziono par: {nr_N_1}_{atom_1} i {nr_N_2}_{atom_2}")
            continue

        #powrót do black boxa
        pary_str= itertools.product(indeks_1, indeks_2)
        pary_maciez=np.array(list(pary_str))
        pary=pary_maciez.astype(int)-1 #i znów na int (+ jest tu łatanie duck tape'em)
        war_porow_f=float(war_porow_str)

        #oblicza odległości na podstawie par które zostały utworzone
        #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
        odleglosci=mdtraj.compute_distances(traj, pary) # tu jest odległość w nm
        odleglosci_a= odleglosci*10 #odległości w arstrongach czy jak się to nazywa

        #przygotowanie wyników do zapisu
        wynik_do_zapisu=odleglosci_a.reshape(len(indeks_1), len(indeks_2))
        roznica_od_exp=(wynik_do_zapisu-war_porow_f)


        if np.any(roznica_od_exp < 0):
            for i, (idx1, idx2) in enumerate(pary):

                # 1. Pobieramy dane tylko dla tej konkretnej pary (i-ta kolumna)
                dane_dla_pary = odleglosci_a[:, i]

                # 2. Liczymy statystyki dla tej konkretnej pary
                srednia_pary = np.mean(dane_dla_pary)
                roznica = srednia_pary - war_porow_f

                # 3. Pobieramy precyzyjne nazwy atomów z topologii (żeby wiedzieć co to jest)
                atom_obj_1 = traj.topology.atom(idx1)
                atom_obj_2 = traj.topology.atom(idx2)

                real_name_1 = f"{atom_obj_1.name}"  # np. 2H2'
                real_res_1 = f"{atom_obj_1.residue.resSeq}"

                real_name_2 = f"{atom_obj_2.name}"  # np. H8
                real_res_2 = f"{atom_obj_2.residue.resSeq}"

                # 4. ZAPIS DO LISTY WYNIKÓW (Każdy wariant jako osobny wiersz)
                wyniki_lista.append({
                    # Informacja z pliku wejściowego (czego szukaliśmy)
                    'Input_Query': f"{nr_N_1} {atom_1} - {nr_N_2} {atom_2}",

                    # Konkretne atomy w tym wariancie
                    'Res1': real_res_1,
                    'Atom1': real_name_1,
                    'Res2': real_res_2,
                    'Atom2': real_name_2,

                    # Wyniki
                    'Exp_Target': war_porow_f,
                    'Sim_Mean': srednia_pary,
                    'Diff': roznica,

                    # Dodatkowa flaga: czy to jest dokładnie ten atom co w inpucie?
                    # (Pomaga w Excelu odróżnić oryginał od bliźniaka)
                    'Is_Twin': 'YES' if (atom_1 != real_name_1 or atom_2 != real_name_2) else 'NO'
                })

        maska_vmd = roznica_od_exp < 0
        if np.any(maska_vmd):
            wiersze, kolumny = np.where(maska_vmd)

            for r, c in zip(wiersze, kolumny):
                atom_id_1 = indeks_1[r]
                atom_id_2 = indeks_2[c]
                komenda = f"label add Bonds 0/{atom_id_1} 0/{atom_id_2}\n"
                f_tcl.write(komenda)


        #print(f"Zadanie ukończono, macierze zapisano w: {args.out}")

#liczy zajebiście bo porównałem z VMD i wmiare pasuje to wszystko - ale potem więcej posprawdzam tych par atomów
#Zapis do pliku też jest git ja poprostu nie na tą pare spojrzałem ;p

#Zapis do pliku
dr=pd.DataFrame(wyniki_lista)
sciezka_dr = os.path.join(args.out, "Pelny_Raport.csv")
dr.to_csv(sciezka_dr, sep=';', decimal=',', index=False, encoding='utf-8-sig')

print(f"✅ Zapisano raport CSV: {sciezka_dr}")

#Zapis raportu o zaginionych atomach
czst_struk=os.path.basename(args.struct)
n_raportu=f"Raport z zaginionych par w pliku: {czst_struk}.txt"
s_raport=os.path.join(args.out, n_raportu)
with (open(s_raport, 'w', encoding='utf-8')) as s_file:
    for elem in brak_par:
        s_file.write(f"{elem}\n")

f_tcl.close()
print("Powstał plik z wizualizacją VMD w:", {sciezka_tcl})
