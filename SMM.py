import argparse
import os
import sys
import itertools
import numpy as np
import pandas as pd
import mdtraj
import glob
from biopandas.pdb import PandasPdb
import concurrent.futures

# Argumenty dla konsoli
parser = argparse.ArgumentParser(
    description='Obliczanie więzów między atomami w strukturach DNA według informacji z NOE')

#utworzenie flag dla programu
parser.add_argument('-s', '--struct', type=str, required=True, help='Ścieżka do pliku struktury (np. structure.pdb)')
parser.add_argument('-n', '--noe', type=str, required=True, help='Ścieżka do pliku z danymi NOE (.txt)')
parser.add_argument('-o', '--out', type=str, default='wyniki_noe', help='Nazwa folderu na wyniki (domyślnie: wyniki_noe)')

args = parser.parse_args()


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
    czysty_nr = str(nr_res).strip()
    czysty_atom = str(nazwa_atom).strip()

    # Szukamy klucza tekstowego tak, jak go zapisaliśmy w pętli PDB
    klucz_glowny = f"{czysty_nr}_{czysty_atom}"
    indeksy = list(mapa.get(klucz_glowny, []))

    # Logika dodawania bliźniaków
    if czysty_atom in atom_blizniaki:
        nazwa_blizniaka = atom_blizniaki[czysty_atom]
        klucz_blizniaka = f"{czysty_nr}_{nazwa_blizniaka}"

        indeksy_blizniaka = list(mapa.get(klucz_blizniaka, []))
        indeksy.extend(indeksy_blizniaka)

    return list(set(indeksy))

    # Zwracamy listę bez duplikatów
    return list(set(indeksy))

# Analiza pliku pdb - fancy funkcja

def analiza_pdb (sciezka_pdb, linie_noe):
    try:
        traj=mdtraj.load(sciezka_pdb)
        dpdb=PandasPdb().read_pdb(sciezka_pdb).df['ATOM']
    except wykluczenia as e:
        print(f"Nie można wczytać {sciezka_pdb}:{e}")
        return sciezka_pdb, 0, 0, 0, "BŁĄD"

    # Mapa poprawek ozmaczen atomow
    mapa_poprawek = {
        "H5'": "1H5'", "H5''": "2H5'",
        "H2'": "1H2'", "H2''": "2H2'",
        "H5'1": "1H5'", "H5'2": "2H5'",
        "H2'1": "1H2'", "H2'2": "2H2'",
        "C5M": "C7", "H71": "H51",
        "H72": "H52", "H73": "H53",
    }

    # Czyszczenie Danych
    dpdb['residue_name'] = dpdb['residue_name'].str.strip()
    dpdb['atom_name'] = dpdb['atom_name'].str.strip()
    dpdb['residue_name'] = dpdb['residue_name'].str.replace(r'\d+$', '', regex=True)
    dpdb['atom_name'] = dpdb["atom_name"].replace(mapa_poprawek)

    mapa_indeksow = {}
    for index, row in dpdb.iterrows():
        nr_reszty = str(row['residue_number']).strip()
        nazwa_atomu = str(row['atom_name']).strip()
        klucz = f"{nr_reszty}_{nazwa_atomu}"

        if klucz not in mapa_indeksow:
            mapa_indeksow[klucz] = []

        # Pamiętaj o przesunięciu o -1, jeśli mdtraj liczy od 0, a PDB od 1!
        indeks_mdtraj = int(row['atom_number'])
        mapa_indeksow[klucz].append(indeks_mdtraj)

    lista_najlepszych_roz = []

    for line in linie_noe:
        nr_N_1 = line.split()[0]
        atom_1 = line.split()[1]
        nr_N_2 = line.split()[2]
        atom_2 = line.split()[3]
        war_porow_str = line.split()[5]

        #the black box - czyli sobie znajduje, kombinuje i liczy odległości
        indeks_1 = pobierz_grupe_indeksow(nr_N_1, atom_1, mapa_indeksow)
        indeks_2 = pobierz_grupe_indeksow(nr_N_2, atom_2, mapa_indeksow)

        # system pomijania pustych par
        if len(indeks_1) == 0 or len(indeks_2) == 0:
            continue

        # powrót do black boxa
        pary_str = list(itertools.product(indeks_1, indeks_2))
        pary_maciez = np.array(pary_str)
        pary = pary_maciez.astype(int) - 1  # i znów na int (+ jest tu łatanie duck tape'em)
        war_porow_f = float(war_porow_str)

        #oblicza odległości na podstawie par które zostały utworzone
        #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
        odleglosci = mdtraj.compute_distances(traj, pary)  # tu jest odległość w nm
        odleglosci_a = odleglosci * 10  #odległości w arstrongach czy jak się to nazywa

        for i, (indeks_1, indeks_2) in enumerate(pary):
            srednie_dla_par = np.mean(odleglosci_a, axis=0)
            roznice_wszystkich = srednie_dla_par - war_porow_f
            #Do vmd ogarnianie par
            ind_win = np.argmin(roznice_wszystkich)

            naj_exp=roznice_wszystkich[ind_win]


        # Dodajemy ją do głównej listy statystyk (do raportu końcowego)
        lista_najlepszych_roz.append(naj_exp)

    arr_diff = np.array(lista_najlepszych_roz)
    ilosc=len(arr_diff)

    limit_granicy=0.5
    if ilosc > 0:
        count_ok = np.sum(arr_diff <= 0)
        count_borderline = np.sum((arr_diff > 0) & (arr_diff <= limit_granicy))
        count_bad = np.sum(arr_diff > limit_granicy)
        proc_ok=(count_ok/ilosc)*100
        proc_borderline=(count_borderline/ilosc)*100
        proc_bad=(count_bad/ilosc)*100
    else:
        # Jeśli program nie znalazł żadnych więzów (ilosc == 0), wpisujemy same zera
        count_ok = 0.0
        count_borderline = 0.0
        count_bad = 0.0

    return sciezka_pdb, count_ok, proc_ok, count_borderline, proc_borderline, count_bad, proc_bad, "OK"

# Glowny mech dzialania skryptu
if __name__ == "__main__":

    # 1. Tu masz swoje wczytywanie argumentów (argparse)
    folder_pdb = args.struct
    plik_noe = args.noe
    folder_out = args.out

    # 2. Wczytywanie NOE
    print(f"Wczytuję plik NOE: {plik_noe} ...")
    with open(plik_noe, 'r') as f:
        linie_noe = [l.strip() for l in f.readlines() if l.strip()]

    # 3. Szukanie plików PDB
    szukana_sciezka = os.path.join(folder_pdb, "*.pdb")
    pliki_pdb = glob.glob(szukana_sciezka)
    print(f"Znaleziono {len(pliki_pdb)} plików PDB do analizy.")

    wyniki_zbiorcze = []

    print(f"🚀 Odpalam masową analizę {len(pliki_pdb)} plików na wielu rdzeniach...")

    # 4. Multiprocessing - Zwróć uwagę na WCIĘCIA! Wszystko jest przesunięte w prawo.
    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures = {executor.submit(analiza_pdb, sciezka, linie_noe): sciezka for sciezka in pliki_pdb}

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            sciezka = futures[future]
            try:
                wynik = future.result()
                wyniki_zbiorcze.append({
                    'Plik': os.path.basename(wynik[0]),
                    'OK': wynik[1],
                    '%OK': wynik[2],
                    'Na_granicy': wynik[3],
                    '%Na_granicy': wynik[4],
                    'Zle': wynik[5],
                    "%Zle": wynik[6],
                    'Status': wynik[7]
                })
                print(f"[{i}/{len(pliki_pdb)}] Przeanalizowano: {os.path.basename(sciezka)}")

            except Exception as exc:  # <--- TUTAJ POPRAWIONA LITERÓWKA (Exception)
                print(f"❌ Plik {os.path.basename(sciezka)} wygenerował błąd: {exc}")

    os.makedirs(folder_out, exist_ok=True)
    df_wyniki = pd.DataFrame(wyniki_zbiorcze)
    sciezka_raportu = os.path.join(folder_out, "ZBIORCZY_RAPORT_NOE.csv")
    df_wyniki.to_csv(sciezka_raportu, sep=';', index=False)

    print(f"\n✅ Zakończono! Zapisano zbiorczy raport: {sciezka_raportu}")



