import argparse, subprocess, concurrent.futures
import os, tracemalloc, sys, time, datetime
import itertools
import numpy as np
import pandas as pd
import mdtraj
import glob
from biopandas.pdb import PandasPdb
import matplotlib.pyplot as plt



# Argumenty dla konsoli
parser = argparse.ArgumentParser(
    description='Obliczanie więzów między atomami w strukturach DNA według informacji z NOE')

#utworzenie flag dla programu
parser.add_argument('-p', '--top', type=str, required=True, help='Ścieżka do topologii (pojedynczy plik .pdb)')
parser.add_argument('-t', '--traj', type=str, required=True, help='Ścieżka do trajektorii (.xtc)')
parser.add_argument('-w', '--tpr', type=str, required=True, help='Ścieszka do pliku .tpr/.gro symulacji (trzeba osuszyć)')
parser.add_argument('-n', '--noe', type=str, required=True, help='Ścieżka do pliku z danymi NOE (.txt)')
parser.add_argument('-o', '--out', type=str, default='wyniki_noe', help='Nazwa folderu na wyniki (domyślnie: wyniki_noe)')

args = parser.parse_args()

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

# Suszarka molekularna - osuszanie wody z trajektorii
def suszarka(xtc, tpr, out ):
    base=os.path.basename(xtc).replace('.xtc', '')
    suchy=os.path.join(out, f"{base}_dry.xtc")

    if os.path.exists(suchy):
        print(f"Sucha trajektoria już istanieje:{suchy}. Pomijamy suszenie")
        return suchy
    print(f"Odpalam GROMACS aby suszyć trajektorie {xtc}")

    komenda= f" echo 1 | gmx trjconv -f {xtc} -s {tpr} -o {suchy}"

    try:
        subprocess.run(komenda, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Trajektoria jest już sucha i zapisana jako: {suchy}")
        return suchy
    except subprocess.CalledProcessError:
        print("cos nie działa, sprawdz czy jest gromacs lub dobry plik")
        sys.exit(1)

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


# Analiza pliku pdb - fancy funkcja

def analiza_pdb (plik, linie_noe, mapa_indeksow):
    lista_najlepszych_roz=[]

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
            odleglosci = mdtraj.compute_distances(plik, pary)  # tu jest odległość w nm
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
        proc_ok=np.round((count_ok/ilosc)*100, 2)
        proc_borderline=np.round((count_borderline/ilosc)*100, 2)
        proc_bad=np.round((count_bad/ilosc)*100, 2)
    else:
        # Jeśli program nie znalazł żadnych więzów (ilosc == 0), wpisujemy same zera
        count_ok = 0.0
        count_borderline = 0.0
        count_bad = 0.0

    return count_ok, proc_ok, count_borderline, proc_borderline, count_bad, proc_bad, "OK"

# Glowny mech dzialania skryptu
if __name__ == "__main__":

    # 1. Start rejestru RAM - for debug purpuse
    tracemalloc.start()


    # 2. Wczytywanie NOE
    print(f"Wczytuję plik NOE: {args.noe} ...")
    with open(args.noe, 'r') as f:
        linie_noe = [l.strip() for l in f.readlines() if l.strip()]


    # Wczytanie trajektorii .xtc
    print (f"Ładowanie trajektorii {args.traj} na podstawie {args.top}")
    sucha_traj=suszarka(args.traj, args.tpr, args.out)
    loaded = mdtraj.load(sucha_traj, top=args.top)
    dpdb = PandasPdb().read_pdb(args.top).df['ATOM']

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

        mapa_indeksow[klucz].append(row['atom_number'])

    wyniki_nonsort=[]
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor()as executor:

        przyszle_wyniki={}
        for i in range(loaded.n_frames):
            poj_klatka=loaded[i]
            zadanie = executor.submit(analiza_pdb, poj_klatka, linie_noe, mapa_indeksow)
            przyszle_wyniki[zadanie]=i

        zrobione=0
        for zadanie in concurrent.futures.as_completed(przyszle_wyniki):
            numer_klatki=przyszle_wyniki[zadanie]
            wynik=zadanie.result()
            wyniki_nonsort.append((numer_klatki, wynik))
            zrobione += 1

            # --- MAGIA ETA ---
            elapsed_time = time.time() - start_time  # Ile czasu minęło od startu
            avg_time_per_file = elapsed_time / zrobione  # Średni czas na jeden plik
            files_left = loaded.n_frames - zrobione  # Ile plików zostało
            eta_seconds = int(avg_time_per_file * files_left)  # Przewidywany czas w sekundach

            current, peak = tracemalloc.get_traced_memory()

        # Formatujemy sekundy do ładnego HH:MM:SS
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            print(f"[{zrobione}/{loaded.n_frames}] | ETA: {eta_str} | Obecne zużycie RAM[MB]: {current/10**6:.2f} | Szczytowe zużycie RAM[MB]: {peak/10**6:.2f}")

        wyniki_sort=sorted(wyniki_nonsort, key=lambda x: x[0])
        wyniki_zbiorcze=[]
        klatki=[]
    for numer_klatki, wynik in wyniki_sort:
        wyniki_zbiorcze.append({
            'Klatka': f"{numer_klatki}",
            'OK': wynik[0],
            '%OK': wynik[1],
            'Na_granicy': wynik[2],
            '%Na_granicy': wynik[3],
            'Zle': wynik[4],
            "%Zle": wynik[5],
            'Status': wynik[6]
        })

    df_wyniki = pd.DataFrame(wyniki_zbiorcze)
    sciezka_raportu = os.path.join(args.out, "ZBIORCZY_RAPORT_NOE.csv")
    df_wyniki.to_csv(sciezka_raportu, sep=';', index=False)


# ------------------------------------- To zostanie zastąpione ----------------------------------------------------
    # HISTOGRAM
    plt.figure(figsize=[10, 10])
    plt.hist(df_wyniki['%OK'], bins=20, color='green', edgecolor='black')
    plt.title('Rozkład zgodności')
    plt.xlabel('Procent zgodności więzów')
    plt.ylabel('Liczba klatek')
    plt.savefig(os.path.join(args.out, "histogram.png"), dpi=300, bbox_inches='tight' )
    plt.close()


    # Dryf w czasie
    plt.scatter(klatki, df_wyniki['%OK'], c = 'red')
    plt.plot(klatki, df_wyniki['%OK'])
    plt.xlabel('Klatka')
    plt.ylabel('Procent zgodnosci więzów [%]')
    plt.title('Dryf w czasie')
    plt.savefig(os.path.join(args.out, "dryf.png"), dpi=600, bbox_inches='tight' )
    plt.close()

    print(f'Wszystkie pliki z analizy zapisano w: {args.out}')





