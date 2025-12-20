Jak używać:

komenda: python PLDCz.py -s [plik ze strukturą] (musi być w fomacie .pdb) -n [plik z więzami] (musie być w formacie .txt) -o [jak ma nazywać się folder]

Jak odczytywać wyniki w tabelach .csv któ®e podaje program:
Oznaczenie atomu wygląda następująco:
        {nr_nukleotydu_na_którym_jest_atom}_{Atom_badany}_{Indeks_atomu_w_pliku_.pdb}
        np. 1_H8_13

plik new_to_old.py zamienia nazewnictwo atomów z nowego standardu pdb (po 2007) [np. H5''] na stary standard [np. 2H5']
użycie tego pliku jest niezbędne do zachowania spójności nazw jednak trzeba najpierw porównać jakie nazwy/standard jest wykorzystywany w pliku z więzami
