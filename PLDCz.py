import argparse
import os
import sys
import itertools
import numpy as np
import pandas as pd
import mdtraj



# /////////////////////////////////Argumenty dla konsoli////////////////////////////////////////////////////

parser = argparse.ArgumentParser(
    description='Obliczanie więzów między atomami w strukturach DNA według informacji z NOE')

#utworzenie flag dla programu
parser.add_argument('-s', '--struct', type=str, required=True, help='Ścieżka do pliku struktury (np. structure.pdb)')
parser.add_argument('-n', '--noe', type=str, required=True, help='Ścieżka do pliku z danymi NOE (.txt)')
parser.add_argument('-o', '--out', type=str, default='wyniki_noe',
                    help='Nazwa folderu na wyniki (domyślnie: wyniki_noe)')

args = parser.parse_args()

#załadowanie pliku .pdb
traj = mdtraj.load(args.struct)

#Plik NOE lokalizacje
#    (0) - Nr. nukleotydu na którym jest pierwszy atom z pary
#    (1) - Pierwszy atom z pary
#    (2) - Nr. nukelotydu na którym jest drugi atom z pary
#    (3) - Drugi atom z pary
#    (4) - Wartość odległości eksperymentalna
#    (5) - Wartość odległości teoretyczna I think

#Wczytanie pliku ze strukturą z pliku .pdb - robocze póki co
kolumny_pdb = [
    (0, 6),  #Kolumna z typem co to jest czyli w naszym przypadku przeszukujemy ATOM
    (6, 11),  #indeks atomu
    (12, 16),  #nazwa atomu
    (22, 30),  #nr. cząsteczki
]

#limit gramiczny dla odchyleń w pród - to się jeszcze dowiemy
limit_granicy = 0.5

mapa_poprawek = {
    "H5'": "1H5'", "H5''": "2H5'",
    "H2'": "1H2'", "H2''": "2H2'",
    "H5'1": "1H5'", "H5'2": "2H5'",
    "H2'1": "1H2'", "H2'2": "2H2'",
    "C5M": "C7", "H71": "H51",
    "H72": "H52", "H73": "H53",
}
nazwy_kolumn_pdb = ['Rekord', 'Indeks', 'Atom', 'Nr. Nukleotydu']
dpdb = pd.read_fwf(args.struct, colspecs=kolumny_pdb, names=nazwy_kolumn_pdb)
dpdb = dpdb[dpdb['Rekord'].isin(['ATOM'])]

#czyszczenie danych - wrzucamy na dane na str zeby ładnie porównywać
dpdb['Atom'] = dpdb['Atom'].astype(str).str.strip()
dpdb['Nr. Nukleotydu'] = dpdb['Nr. Nukleotydu'].astype(str).str.strip()
dpdb['Atom'] = dpdb['Atom'].replace(mapa_poprawek)

#tworzenie mapy z pogrupowanymi indeksami atomów
mapa_indeksow = dpdb.groupby(['Nr. Nukleotydu', 'Atom'])['Indeks'].apply(list).to_dict()

#tworzenie folderu do którego wszystkie pliki z macierzami będą wrzucone
os.makedirs(args.out, exist_ok=True)

#Słownik do sprawdzania atomów bliźniaczych
atom_blizniaki = {
    # DNA / RNA (cukier)
    "1H2'": "2H2'", "2H2'": "1H2'",
    "H21": "H22", "H22": "H21",
    "1H5'": "2H5'", "2H5'": "1H5'",
    "1H2": "2H2", "2H2": "1H2",  # Guanina
    "1H4": "2H4", "2H4": "1H4",  # Cytozyna
    "1H6": "2H6", "2H6": "1H6",  # Adenina

}


#szukanie bliźniaków i pobieranie dla nich danych z mapy
def pobierz_grupe_indeksow(nr_res, nazwa_atom, mapa):
    indeksy = list(mapa.get((nr_res, nazwa_atom), []))

    if nazwa_atom in atom_blizniaki:
        nazwa_blizniaka = atom_blizniaki[nazwa_atom]
        indeksy_blizniaka = list(mapa.get((nr_res, nazwa_blizniaka), []))
        indeksy.extend(indeksy_blizniaka)
    return list(set(indeksy))


#początek tworzenia trajektorii dla dobrych w VMD
nazwa_tcl_ok = "spelnione_vizu.tcl"
sciezka_tcl_ok = os.path.join(args.out, nazwa_tcl_ok)
f_tcl_ok = open(sciezka_tcl_ok, "w", encoding="utf-8")
f_tcl_ok.write("# Skrypt wizualizacji par NOE dla VMD\n")
f_tcl_ok.write("# Użycie w VMD: Extensions -> Tk Console -> wpisz: source visualize_NOE.tcl\n")

#tworzenie trajektorii dla tych co sa na granicy
nazwa_tcl_border = "vmd_borderline.tcl"
sciezka_tcl_border = os.path.join(args.out, nazwa_tcl_border)
f_tcl_border = open(sciezka_tcl_border, "w", encoding="utf-8")
f_tcl_border.write("# Pary na granicy (0 < Diff <= 0.5)\n")
f_tcl_border.write("color Labels Bonds orange\n")

#trajektorie tych niespełnionych
nazwa_tcl_bad = "vmd_bad.tcl"
sciezka_tcl_bad = os.path.join(args.out, nazwa_tcl_bad)
f_tcl_bad = open(sciezka_tcl_bad, "w", encoding="utf-8")
f_tcl_bad.write("# Pary niespełnione (Diff > 0.5)\n")
f_tcl_bad.write("color Labels Bonds red\n")

brak_par = []
pary = []
wyniki_lista = []
lista_najlepszych_roznic = []

with (open(args.noe, 'r', encoding='utf-8') as NOE_file):
    for line in NOE_file:
        nr_N_1 = line.split()[0]
        atom_1 = line.split()[1]
        nr_N_2 = line.split()[2]
        atom_2 = line.split()[3]
        war_porow_str = line.split()[5]

        #the black box - czyli sobie znajduje, kombinuje i liczy odległości
        indeks_1 = np.array(
            pobierz_grupe_indeksow(nr_N_1, atom_1, mapa_indeksow))  #znajdujemy indeksy dla pierwszego atomu
        indeks_2 = np.array(
            pobierz_grupe_indeksow(nr_N_2, atom_2, mapa_indeksow))  #znajdujey indeksy dla drugiego atomu

        #system pomijania par które się nie pojawiły i ich dokumentowania w terminalu(poki co)
        if len(indeks_1) == 0 or len(indeks_2) == 0:
            print(
                f"Nie można znaleźć pary: {nr_N_1}_{atom_1} i {nr_N_2}_{atom_2}, przechodze dalej do kolejnej pary z pliku")
            brak_par.append(f"Nie znaleziono par: {nr_N_1}_{atom_1} i {nr_N_2}_{atom_2}")
            continue

        #powrót do black boxa
        pary_str = itertools.product(indeks_1, indeks_2)
        pary_maciez = np.array(list(pary_str))
        pary = pary_maciez.astype(int) - 1  #i znów na int (+ jest tu łatanie duck tape'em)
        war_porow_f = float(war_porow_str)

        #oblicza odległości na podstawie par które zostały utworzone
        #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
        odleglosci = mdtraj.compute_distances(traj, pary)  # tu jest odległość w nm
        odleglosci_a = odleglosci * 10  #odległości w arstrongach czy jak się to nazywa

        #przygotowanie wyników do zapisu
        wynik_do_zapisu = odleglosci_a.reshape(len(indeks_1), len(indeks_2))
        roznica_od_exp = (wynik_do_zapisu - war_porow_f)

        if np.any(roznica_od_exp < 0):
            for i, (idx1, idx2) in enumerate(pary):
                # 1. Pobieramy dane tylko dla tej konkretnej pary (i-ta kolumna)
                dane_dla_pary = odleglosci_a[:, i]

                # 2. Liczymy statystyki dla tej konkretnej pary
                srednia_pary=np.mean(dane_dla_pary)
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
                    'Index PDB1': idx1,
                    'Res2': real_res_2,
                    'Atom2': real_name_2,
                    'Index PDB2': idx2,

                    # Wyniki
                    'Exp_Target': war_porow_f,
                    'Sim_Mean': srednia_pary,
                    'Diff': roznica,

                    # Dodatkowa flaga: czy to jest dokładnie ten atom co w inpucie?
                    # (Pomaga w Excelu odróżnić oryginał od bliźniaka)
                    'Is_Twin': 'YES' if (atom_1 != real_name_1 or atom_2 != real_name_2) else 'NO'
                })


        # --- PĘTLA PO WARIANTACH (np. 1H2', 2H2') ---
        for i, (idx1, idx2) in enumerate(pary):
            srednie_dla_par = np.mean(odleglosci_a, axis=0)
            roznice_wszystkich = srednie_dla_par - war_porow_f
            #Do vmd ogarnianie par
            ind_win = np.argmin(roznice_wszystkich)
            #vmd_gap = np.min(roznice_wszystkich)

            id_at_1_win = pary[ind_win][0]
            id_at_2_win = pary[ind_win][1]
            naj_exp=roznice_wszystkich[ind_win]


        # Dodajemy ją do głównej listy statystyk (do raportu końcowego)
        lista_najlepszych_roznic.append(naj_exp)

        if naj_exp < 0:
            f_tcl_ok.write(f"label add Bonds 0/{id_at_1_win} 0/{id_at_2_win}\n")

        elif 0 < naj_exp <= limit_granicy:
            f_tcl_border.write(f"label add Bonds 0/{id_at_1_win} 0/{id_at_2_win}\n")

        else:
            f_tcl_bad.write(f"label add Bonds 0/{id_at_1_win} 0/{id_at_2_win}\n")

total_noe = len(lista_najlepszych_roznic)

if total_noe > 0:
    # Konwersja na tablicę numpy dla łatwiejszego liczenia
    arr_diff = np.array(lista_najlepszych_roznic)

    # DEFINICJE KATEGORII:
    # 1. SPEŁNIONE: Różnica <= 0 (Symulacja jest bliżej lub równo z limitem)
    count_ok = np.sum(arr_diff <= 0)

    # 2. NA GRANICY (Lekkie naruszenie): Różnica między 0 a 0.5 Angstrema
    # Możesz zmienić 0.5 na inną tolerancję, np. 0.3

    count_borderline = np.sum((arr_diff > 0) & (arr_diff <= limit_granicy))

    # 3. NIESPEŁNIONE (Duże naruszenie): Różnica > 0.5 Angstrema
    count_bad = np.sum(arr_diff > limit_granicy)

    # Obliczanie procentów
    proc_ok = (count_ok / total_noe) * 100
    proc_borderline = (count_borderline / total_noe) * 100
    proc_bad = (count_bad / total_noe) * 100

    # --- WYPISANIE NA EKRAN ---
    print("\n" + "=" * 40)
    print(f"PODSUMOWANIE STATYSTYK NOE (Total: {total_noe})")
    print("=" * 40)
    print(f"✅ SPEŁNIONE (Diff <= 0):         {count_ok:4d} ({proc_ok:.1f}%)")
    print(f"⚠️ NA GRANICY (0 < Diff <= {limit_granicy}): {count_borderline:4d} ({proc_borderline:.1f}%)")
    print(f"❌ NIESPEŁNIONE (Diff > {limit_granicy}):    {count_bad:4d} ({proc_bad:.1f}%)")
    print("=" * 40 + "\n")

    # --- ZAPIS DO PLIKU TEKSTOWEGO ---
    sciezka_raportu = os.path.join(args.out, "Raport_Procentowy.txt")
    with open(sciezka_raportu, "w", encoding="utf-8") as f1:
        f1.write("RAPORT ZGODNOŚCI STRUKTURY Z WIĘZAMI NOE\n")
        f1.write("========================================\n")
        f1.write(f"Plik struktury: {args.struct}\n")
        f1.write(f"Liczba analizowanych więzów: {total_noe}\n\n")

        f1.write(f"1. SPEŁNIONE (Diff <= 0 A):\n")
        f1.write(f"   Liczba: {count_ok}\n")
        f1.write(f"   Procent: {proc_ok:.2f}%\n\n")

        f1.write(f"2. NA GRANICY (0 < Diff <= {limit_granicy} A):\n")
        f1.write(f"   Liczba: {count_borderline}\n")
        f1.write(f"   Procent: {proc_borderline:.2f}%\n\n")

        f1.write(f"3. NIESPEŁNIONE (Diff > {limit_granicy} A):\n")
        f1.write(f"   Liczba: {count_bad}\n")
        f1.write(f"   Procent: {proc_bad:.2f}%\n")

    print(f"Zapisano statystyki w pliku: {sciezka_raportu}")

else:
    print("Nie znaleziono żadnych więzów do analizy.")

#liczy zajebiście bo porównałem z VMD i wmiare pasuje to wszystko - ale potem więcej posprawdzam tych par atomów
#Zapis do pliku też jest git ja poprostu nie na tą pare spojrzałem ;p

#Zapis do pliku
dr = pd.DataFrame(wyniki_lista)
sciezka_dr = os.path.join(args.out, "Pelny_Raport.csv")
dr.to_csv(sciezka_dr, sep=';', decimal=',', index=False, encoding='utf-8-sig')

print(f"✅ Zapisano raport CSV: {sciezka_dr}")

#Zapis raportu o zaginionych atomach
czst_struk = os.path.basename(args.struct)
n_raportu = f"Raport z zaginionych par w pliku: {czst_struk}.txt"
s_raport = os.path.join(args.out, n_raportu)
with (open(s_raport, 'w', encoding='utf-8')) as s_file:
    for elem in brak_par:
        s_file.write(f"{elem}\n")

f_tcl_ok.close()
f_tcl_border.close()
f_tcl_bad.close()
print("Powstał plik z wizualizacją VMD w:", {sciezka_tcl_ok})
