

Zosia:
- feature selection methods (korelacje X, korelacje X z Y)
- zwracanie rankingów zmiennych  LUB jednego genialnego rankingu TOP 30 zmiennych
- następne etapy pipeline'u (robione przez Matiego i Huberta) powinny uwzględniać ten ranking ale optymalizować jeszcze punkt odcięcia (np. wybieramy tylko top5 z top30 zmiennych)


Jeśli uzywamy random forest to Platt calibration.




Na koniec:
- Krzywe Lift i cap (krzywa słuzące do wyboru ludzi do których warto wysłać ofertę) - szukamy na tych krzywych jakiegoś puktu zgięcia (to chyba na TEST powinno być zrobione) - trzeba o tym jeszcze doczytać, nie wiemy czy da się to zastosować
- można też ewentualnie jakąś metodę łokcia
- albo frakcja z train przeniesiona na test

