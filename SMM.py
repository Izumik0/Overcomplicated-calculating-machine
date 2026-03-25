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

    klucz_glowny = f"{czysty_nr}_{czysty_atom}"
    indeksy = list(mapa.get(klucz_glowny, []))

    if czysty_atom in atom_blizniaki:
        nazwa_blizniaka = atom_blizniaki[czysty_atom]
        klucz_blizniaka = f"{czysty_nr}_{nazwa_blizniaka}"
        indeksy_blizniaka = list(mapa.get(klucz_blizniaka, []))
        indeksy.extend(indeksy_blizniaka)

    # BARDZO WAŻNE: sorted() gwarantuje, że pary nie pomieszają się na wielu rdzeniach!
    return sorted(list(set(indeksy)))

# Analiza pliku pdb - fancy funkcja

def analiza_pdb (plik, linie_noe, mapa_indeksow):
    lista_najlepszych_roz=[]
    lista_sr_odl=[]

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
                lista_sr_odl.append(np.array([]))
                continue

            # powrót do black boxa
            pary_str = list(itertools.product(indeks_1, indeks_2))
            pary_maciez = np.array(pary_str)
            pary = pary_maciez.astype(int) - 1  # i znów na int (+ jest tu łatanie duck tape'em)
            war_porow_f = float(war_porow_str)

            #oblicza odległości na podstawie par które zostały utworzone
            #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
            odleglosci = mdtraj.compute_distances(plik, pary) # tu jest odległość w nm
            odleglosci_a = odleglosci*10
            odleglosci_a_f = (odleglosci * 10).flatten()  #odległości w arstrongach czy jak się to nazywa
            lista_sr_odl.append(odleglosci_a_f)

            for i, (indeks_1, indeks_2) in enumerate(pary):
                #srednie_dla_par = np.mean(odleglosci_a, axis=0)
                roznice_wszystkich = odleglosci_a[0] - war_porow_f
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

    return count_ok, proc_ok, count_borderline, proc_borderline, count_bad, proc_bad, lista_sr_odl, arr_diff

def atomy_dla_kata(nazwa_kata, nr_res, res_name):
    nr = int(nr_res)
    nazwa = nazwa_kata.upper()
    if nazwa == 'ALPHA': return [(nr-1, "O3'"), (nr, "P"), (nr, "O5'"), (nr, "C5'")]
    elif nazwa == 'BETA': return [(nr, "P"), (nr, "O5'"), (nr, "C5'"), (nr, "C4'")]
    elif nazwa == 'GAMMA': return [(nr, "O5'"), (nr, "C5'"), (nr, "C4'"), (nr, "C3'")]
    elif nazwa == 'NU1': return [(nr, "O4'"), (nr, "C1'"), (nr, "C2'"), (nr, "C3'")]
    elif nazwa == 'ZETA': return [(nr, "C3'"), (nr, "O3'"), (nr+1, "P"), (nr+1, "O5'")]
    elif nazwa == 'CHI':
        if res_name.upper() in ['A', 'G', 'DA', 'DG']:
            return [(nr, "O4'"), (nr, "C1'"), (nr, "N9"), (nr, "C4")]
        else:
            return [(nr, "O4'"), (nr, "C1'"), (nr, "N1"), (nr, "C2")]
    return []

def build_czworke(definicja_atomow, mapa_indeksow):
    czworka = []
    for nr, atom in definicja_atomow:
        klucz = f"{nr}_{atom}"
        indeksy = mapa_indeksow.get(klucz, [])
        if not indeksy: return None # Pomijamy, jeśli brakuje atomu (np. brzeg łańcucha)
        czworka.append(indeksy[0] - 1)
    return czworka

def analiza_katow(plik, linie_katy, mapa_indeksow):
    if not linie_katy:
        return 0, 0.0, 0, 0.0

    czworki_lista = []
    limity = []

    for line in linie_katy:
        kolumny = line.split()
        nr_res = kolumny[0]
        res_name = kolumny[1]
        nazwa_kata = kolumny[2].upper()
        limit_min = float(kolumny[3])
        limit_max = float(kolumny[4])

        definicja = atomy_dla_kata(nazwa_kata, nr_res, res_name)
        czworka = build_czworke(definicja, mapa_indeksow)

        if czworka:
            czworki_lista.append(czworka)
            limity.append((limit_min, limit_max))

    if not czworki_lista:
        return 0, 0.0, 0, 0.0

    katy_radiany = mdtraj.compute_dihedrals(plik, czworki_lista)
    katy_stopnie = np.rad2deg(katy_radiany[0])

    count_ok_k = 0
    count_bad_k = 0
    kat_lista=[]
    for kat, (l_min, l_max) in zip(katy_stopnie, limity):
        # Korekta ujemnych kątów (żeby np. -120 pasowało do przedziału 200-300 jeśli taki ustawiono)
        kat_znormalizowany = kat + 360 if (kat < 0 and l_min >= 0) else kat
        kat_lista.append(kat_znormalizowany)

        if l_min <= kat_znormalizowany <= l_max:
            count_ok_k += 1
        else:
            count_bad_k += 1

    ilosc_kant=len(czworki_lista)
    proc_ok_k=(count_ok_k/ilosc_kant)*100
    proc_bad_k=(count_bad_k/ilosc_kant)*100
    return count_ok_k, proc_ok_k, count_bad_k, proc_bad_k, kat_lista

def obl_core(anything, linie_noe, linie_katy, mapa_indeksow):
    wyniki_noe = analiza_pdb(anything, linie_noe, mapa_indeksow)
    wyniki_katy = analiza_katow(anything, linie_katy, mapa_indeksow)
    return wyniki_noe, wyniki_katy

# Glowny mech dzialania skryptu
if __name__ == "__main__":

    # 1. Start rejestru RAM - for debug purpuse
    tracemalloc.start()

    # 2. Wczytywanie NOE
    linie_noe = []
    linie_katy = []
    zdefiniowane_katy = {'ALPHA', 'BETA', 'GAMMA', 'ZETA', 'NU1', 'CHI'}
    print(f"Wczytuję plik NOE: {args.noe} ...")

    with open(args.noe, 'r') as f:
        for l in f.readlines():
            l=l.strip()
            if not l: continue
            kolumny = l.split()
            if len(kolumny) >= 3 and kolumny[2].upper() in zdefiniowane_katy:
                linie_katy.append(l)
            else: linie_noe.append(l)
    print(f"Znaleziono {len(linie_noe)} więzów odległościowych (NOE) oraz {len(linie_katy)} więzów kątowych.")

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
            zadanie = executor.submit(obl_core, loaded[i], linie_noe, linie_katy, mapa_indeksow)
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
        srednia_biezaca_noe = None
        srednia_biezaca_katy = None
        historia_katow_w_czasie = []
    for numer_klatki, wynik in wyniki_sort:
        wynik_noe, wynik_katy=wynik
        n = numer_klatki + 1  # Licznik klatek (1, 2, 3...)
        wartosci_sred=wynik_noe[6]
        # --- 1. Aktualizacja średniej dla NOE ---
        if srednia_biezaca_noe is None:
            # Kopiujemy tablice do nowej listy
            srednia_biezaca_noe = [np.copy(tablica) for tablica in wartosci_sred]
        else:
            for i in range(len(wartosci_sred)):
                # Bezpieczne nadpisywanie tablicy nową wartością średniej
                srednia_biezaca_noe[i] = srednia_biezaca_noe[i] + (
                            wartosci_sred[i] - srednia_biezaca_noe[i]) / n

        # --- 2. Aktualizacja średniej dla Kątów ---
        wartosci_katy_klatka = wynik_katy[-1] if len(wynik_katy) > 0 else []
        if type(wartosci_katy_klatka) is not str and len(wartosci_katy_klatka) > 0:

            # ZAPISUJEMY PEŁNĄ HISTORIĘ KLATKI DO WYKRESÓW:
            historia_katow_w_czasie.append(wartosci_katy_klatka)

            wartosci_katy_np = np.array(wartosci_katy_klatka)
            if srednia_biezaca_katy is None:
                srednia_biezaca_katy = np.copy(wartosci_katy_np)
            else:
                srednia_biezaca_katy += (wartosci_katy_np - srednia_biezaca_katy) / n

        wyniki_zbiorcze.append({
            'Klatka': f"{numer_klatki}",
            'OK': wynik_noe[0],
            '%OK': wynik_noe[1],
            'Na_granicy': wynik_noe[2],
            '%Na_granicy': wynik_noe[3],
            'Zle': wynik_noe[4],
            "%Zle": wynik_noe[5],
            'Kąty_Ok': wynik_katy[0],
            '% Kąty_Ok': wynik_katy[1],
            'Kąty_Żle': wynik_katy[2],
            '% Kąty_Złe': wynik_katy[3]
        })


    df_wyniki = pd.DataFrame(wyniki_zbiorcze)
    sciezka_raportu = os.path.join(args.out, "RAPORT_NOE.csv")
    df_wyniki.to_csv(sciezka_raportu, sep=';', index=False)

    print("\n📊 Generowanie Pełnego Raportu wszystkich wygenerowanych par...")

    # Wczytujemy ORYGINALNĄ topologię (sprzed czyszczenia), żeby wyciągnąć prawdziwe nazwy PDB np. H5'1
    oryginalne_pdb = PandasPdb().read_pdb(args.top).df['ATOM']
    indeks_do_nazwy = {}
    for index, row in oryginalne_pdb.iterrows():
        indeks_do_nazwy[row['atom_number']] = (str(row['residue_number']).strip(), str(row['atom_name']).strip())

    raport_sredni = []

    if srednia_biezaca_noe is not None:
        for linia, srednia_wszystkie in zip(linie_noe, srednia_biezaca_noe):
            kolumny = linia.split()
            if len(kolumny) < 6:
                continue

            nr_N_1, atom_1 = kolumny[0], kolumny[1]
            nr_N_2, atom_2 = kolumny[2], kolumny[3]
            war_porow_f = float(kolumny[5])

            input_query = f"{nr_N_1} {atom_1} - {nr_N_2} {atom_2}"

            # Ponownie generujemy pary w identycznej kolejności jak rdzeń obliczeniowy
            indeks_1 = pobierz_grupe_indeksow(nr_N_1, atom_1, mapa_indeksow)
            indeks_2 = pobierz_grupe_indeksow(nr_N_2, atom_2, mapa_indeksow)

            if not indeks_1 or not indeks_2:
                continue

            pary_str = list(itertools.product(indeks_1, indeks_2))
            roznice_grupy = [abs(float(mean) - war_porow_f) for mean in srednia_wszystkie]
            najlepszy_wynik_w_grupie = min(roznice_grupy) if roznice_grupy else None
            # Łączymy wygenerowaną parę z jej fizyczną średnią
            for (idx1, idx2), sim_mean in zip(pary_str, np.array(srednia_wszystkie).flatten()):
                # Odpytujemy słownik o prawdziwe nazwy z PDB dla danych indeksów
                res1, a_name1 = indeks_do_nazwy.get(idx1, (nr_N_1, atom_1))
                res2, a_name2 = indeks_do_nazwy.get(idx2, (nr_N_2, atom_2))


                diff = float(sim_mean) - war_porow_f
                is_twin = "YES" if (najlepszy_wynik_w_grupie is not None and abs(diff) == najlepszy_wynik_w_grupie) else "NO"
                # Zapisujemy rozdzielone dane, wiersz po wierszu
                raport_sredni.append({
                    'Input_Query': input_query,
                    'Res1': res1,
                    'Atom1': a_name1,
                    'Index PDB1': idx1,
                    'Res2': res2,
                    'Atom2': a_name2,
                    'Index PDB2': idx2,
                    'Exp_Target': war_porow_f,
                    'Sim_Mean': sim_mean,
                    'Diff': diff,
                    'Is_Best_Pair?': is_twin
                })

    # Tutaj możesz wstawić z powrotem stary kod do zapisywania kątów (jeśli ich wciąż używasz)
    if srednia_biezaca_katy is not None:
        for linia, srednia in zip(linie_katy, srednia_biezaca_katy):
            raport_sredni.append({
                'Typ': 'KAT (Stopnie)',
                'Definicja_w_pliku': linia,
                'średnii_kąt': round(srednia, 2)
            })

    if raport_sredni:
        df_srednie = pd.DataFrame(raport_sredni)
        sciezka_srednie = os.path.join(args.out, "Pelny_sr_Raport.csv")
        # Zapis parametrem decimal=',' zapewnia idealne wczytywanie ułamków przez polskiego Excela
        df_srednie.to_csv(sciezka_srednie, sep=';', index=False, decimal=',')
        print(f"✅ Zapisano pełny, rozdzielony raport więzów: {sciezka_srednie}")

    if historia_katow_w_czasie:
        print("\n📈 Generowanie historii kątów do wykresów...")

        # Zamieniamy naszą listę na macierz i transponujemy (.T),
        # żeby wiersze to były konkretne kąty, a kolumny to upływający czas (klatki)
        macierz_katow = np.array(historia_katow_w_czasie).T

        raport_historia_katow = []
        nazwy_kolumn = ["_".join(linia.split()[:4]) for linia in linie_katy]

        for idx_klatki, wartosci_w_klatce in enumerate(historia_katow_w_czasie):
            wiersz = {'Klatka': idx_klatki + 1}
            for nazwa_kata, wartosc in zip(nazwy_kolumn, wartosci_w_klatce):
                wiersz[nazwa_kata] = round(wartosc, 2)
            raport_historia_katow.append(wiersz)

        df_historia_katow = pd.DataFrame(raport_historia_katow)
        sciezka_hist_katow = os.path.join(args.out, "Historia_Katow_Wykresy.csv")

        # Zapis parametrem decimal=',' zapewnia wsparcie dla polskiego Excela
        df_historia_katow.to_csv(sciezka_hist_katow, sep=';', index=False, decimal=',')
        print(f"✅ Zapisano plik z historią kątów (gotowy na wykresy!): {sciezka_hist_katow}")

    print(f'Wszystkie pliki z analizy zapisano w: {args.out}')