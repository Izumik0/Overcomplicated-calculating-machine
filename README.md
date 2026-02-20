Jak używać:  
Istnieją dwa pliki   
PLDCz.py - bierze pod uwage wszystkie atomy czyli daje nam informacje zwrotną o wszystkich więzach   
PLDCzG.py - bierze pod uwage tylko Guanine, która tworzy szkielet G-4   
jeżeli chodzi o komendę to PLDCzG.py używa się tak samo jak PLDCz.py   
SMM.py - obliczenia na wielu rdzeniach, szybsza wersja PLDCz.py, która jest zaprojektowana do sprawdzania pojedyńćzych klatek w symulacji 

komenda: python PLDCz.py -s [plik ze strukturą] (musi być w fomacie .pdb) -n [plik z więzami] (musie być w formacie .txt) -o [jak ma nazywać się folder] 

Program generuje zbiorczy raport dla danej struktury, raport o atomach/parach których nie znalazł oraz plik .tcl który umożliwia wizualizacje więzów, które spełniają warunki. 

już nie trzeba używac pliku old_to_new.py --> została wprowadzona poprawka która ujednolica zapis tak aby można go było porównac z więzami
