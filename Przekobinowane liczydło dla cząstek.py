import itertools
import numpy as np
import pandas as pd
import mdtraj

traj=mdtraj.load('structure.pdb')

#Plik NOE lokalizacje
#    (0) #Nr. nukleotydu na którym jest pierwszy atom z pary
#    (1) #Pierwszy atom z pary
#    (2) #Nr. nukelotydu na którym jest drugi atom z pary
#    (3) #Drugi atom z pary

#Wczytanie pliku ze strukturą z pliku .pdb - robocze póki co
kolumny_pdb = [
    (0, 6), #Kolumna z typem co to jest czyli w naszym przypadku przeszukujemy ATOMS
    (6, 11), #indeks atomu
    (12, 16), #nazwa atomu
    (22, 26), #nr. nukleotydu
]
nazwy_kolumn_pdb=['Rekord', 'Indeks', 'Atom', 'Nr. Nukleotydu']
dpdb = pd.read_fwf('structure.pdb', colspecs=kolumny_pdb, names=nazwy_kolumn_pdb)
dpdb = dpdb[dpdb['Rekord'].isin(['ATOM'])]

#czyszczenie danych - wrzucamy na dane na str zeby ładnie porównywać
dpdb['Atom'] = dpdb['Atom'].str.strip()
dpdb['Nr. Nukleotydu'] = dpdb['Nr. Nukleotydu'].str.strip()



#tworzenie mapy z pogrupowanymi indeksami atomów
mapa_indeksow=dpdb.groupby(['Nr. Nukleotydu', 'Atom'])['Indeks'].apply(list).to_dict()
#print(mapa_indeksów)

#tworzenie folderu do którego wszystkie pliki z macierzami będą wrzucone


pary=[]

with open('NOE_clean_uni_names.txt', 'r', encoding='utf-8') as NOE_file:
    for line in NOE_file:
        nr_N_1 = line.split()[0]
        atom_1 = line.split()[1]
        nr_N_2 = line.split()[2]
        atom_2 = line.split()[3]

        #the black box - czyli sobie znajduje, kombinuje i liczy odległości
        indeks_1=np.array(mapa_indeksow.get((nr_N_1, atom_1), [])) #znajdujemy indeksy dla pierwszego atomu
        indeks_2=np.array(mapa_indeksow.get((nr_N_2, atom_2), [])) #znajdujey indeksy dla drugiego atomu
        pary_str= itertools.product(indeks_1, indeks_2)
        pary_maciez=np.array(list(pary_str))
        pary=pary_maciez.astype(int)-1 #i znów na int (+ jest tu łatanie duck tape'em)
        #print("Para", nr_N_1, atom_1, "&", nr_N_2 , atom_2, indeks_1, indeks_2)
        #print("Możliwe pary", pary)

        #oblicza odległości na podstawie par które zostały utworzone
        #odległości są przedstawiane w macierzy gdzie kolumny to te same atomy_1 z rórnych czasetek a wiersze to tak samo ale atomy_2
        odleglosci=mdtraj.compute_distances(traj, pary) # tu jest odległość w nm
        odleglosci_a= odleglosci*10 #odległości w arstrongach czy jak się to nazywa
        print("Odległości między atomami:\n", "Nr. Nukleotydu:", nr_N_1, "Atom:", atom_1, "Nr. nuk:", nr_N_2, "atom", atom_2, "\n", odleglosci_a)


#liczy zajebiście bo porównałem z VMD i wmiare pasuje to wszystko - ale potem więcej posprawdzam tych par atomów