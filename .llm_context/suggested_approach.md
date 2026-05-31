Oto kompletna notatka strategiczna, stworzona zgodnie z najlepszymi praktykami z konkursów data science. Zakładamy, że etap wstępnej redukcji 500 zmiennych masz już za sobą i dysponujesz wąską pulą obiecujących cech (np. top 5–15 zmiennych).

Teraz kluczem do wygranej jest idealna synergia między **podziałem danych, wyborem modelu, kalibracją prawdopodobieństwa a optymalizacją progu decyzyjnego**.

---

## Kompleksowa Strategia Modelowania i Optymalizacji Zysku

### 1. Architektura Walidacji (Validation Scheme)

W konkursach stabilna walidacja to różnica między wygraną a zaliczeniem potężnego spadku w tabeli (tzw. *shake-up*). Ponieważ zbiór treningowy liczy 5000 obserwacji, idealnym podejściem jest **Stratified $k$-fold Cross-Validation (np. $k=5$)**.

* 
**Dlaczego Stratified?** Musimy zachować identyczną proporcję klas ($1$ vs $0$) w każdym foldzie, co w treningu.


* **Logika podziału:** Każdy z 5 foldów służy po kolei jako zbiór walidacyjny ($V_k$). Modele trenujemy na pozostałych 4 foldach ($T_k$). Prawdziwym sekretem jest jednak to, co robimy z "out-of-fold predictions" (OOF). Prawdopodobieństwa wygenerowane dla całego zbioru treningowego na drodze OOF posłużą nam do bezpiecznego wyznaczenia optymalnego progu decyzyjnego, zapobiegając przeuczeniu (overfittingowi).

---

### 2. Wybór Modelu (Model Selection)

Nie szukaj jednego "idealnego" algorytmu. Wygrywające rozwiązania opierają się na **Ensemble Learningu** (różnorodności). Powinieneś przetestować i połączyć trzy rodziny modeli:

1. **AdaBoost / XGBoost:** Królowie danych tabelarycznych. Świetnie radzą sobie z nieliniowymi zależnościami i interakcjami między zmiennymi.
2. **Regresja Logistyczna (z regularyzacją L2):** Prosty model liniowy. Bardzo często ustępuje genom gradientowym pod kątem czystego AUC, ale ma ogromną zaletę: generuje świetnie skalibrowane, gładkie prawdopodobieństwa.
3. **Random Forest:** Zapewnia dobrą stabilność i nie ma tendencji do tak gwałtownego przeuczania się jak boosting.
4. SVM



**Logika połączenia:** Ostateczny model powinien być średnią ważoną (Ensemble) z prawdopodobieństw tych modeli.

---

### 3. Tuning Hiperparametrów pod metrykę biznesową

Standardowy tuning (np. przez Optuna) optymalizuje pod kątem *Log-Loss* lub *AUC*. Dla Ciebie te metryki są tylko drogowskazem, a nie celem.

* **Logika:** Użyj Optuny do optymalizacji modeli pod kątem log-loss, ale trzymaj bardzo rygorystyczne ograniczenia przeciw przeuczeniu (np. w LightGBM: wysokie `min_data_in_leaf`, niskie `learning_rate`, niska głębokość drzewa `max_depth`).
* W barierach tak kosztownych zmiennych (200 EUR/zmienna) model musi być prosty, by wzorce były powtarzalne na zbiorze testowym.



---

### 4. Kalibracja Prawdopodobieństwa (Probability Calibration)

Algorytmy takie jak XGBoost czy Random Forest optymalizują podział klas, co sprawia, że ich surowe wyniki (output) często **nie odzwierciedlają rzeczywistego prawdopodobieństwa**. Na przykład model może rzucać wartościami 0.9 dla pewnych obserwacji, choć realna szansa konwersji wynosi 65%.

Skoro Twój zysk zależy bezpośrednio od precyzji finansowej, musisz te prawdopodobieństwa "wyprostować".

* **Metoda:** Zastosuj **Platt Scaling** (kalibrację sigmoidalną) lub **Isotonic Regression** na predykcjach out-of-fold (OOF).
* Dzięki temu, jeśli model przypisze klientowi wartość $P=0.40$, będziesz mieć statystyczną pewność, że w tej grupie dokładnie 40% osób przyjmie ofertę.

---

### 5. Wybór Progu Decyzyjnego (Threshold) oraz Liczby Klientów

To najważniejszy, czysto logiczny etap projektu. Zapomnij o domyślnym progu $0.5$. Musisz wyznaczyć próg matematycznie na podstawie wzoru na oczekiwaną wartość zysku ($E(\text{Profit})$).

Mając dany zysk z TP (+10 EUR) i stratę z FP (-5 EUR), koszt wysłania oferty do klienta o prawdopodobieństwie konwersji $P$ wynosi:

$$E(\text{Profit}) = P \times 10 - (1 - P) \times 5$$

Chcemy, aby każda wysyłka przynosiła zysk większy niż zero ($E(\text{Profit}) > 0$):

$$10P - 5 + 5P > 0$$

$$15P > 5$$

$$P > \frac{1}{3} \approx 0.3333$$

#### Teoretyczny a Praktyczny Próg Decyzyjny:

Teoretycznie powinieneś wysłać ofertę do każdego, u kogo $P > 0.333$. **W praktyce konkursowej musisz jednak uwzględnić ryzyko błędu i koszt zmiennych.** Ponieważ zapłaciłeś już ogromną kwotę za sam wybór zmiennych (np. 5 zmiennych = 1000 EUR długu na start), Twój próg na zbiorze walidacyjnym musi być bezpieczniejszy.

#### Procedura wyboru ostatecznej liczby klientów na zbiorze TEST:

1. Na podstawie modeli przeszkolonych w Cross-Validation generujesz ostateczne prawdopodobieństwa dla 5000 klientów ze zbioru testowego ($x\_test.txt$). Uśredniasz wyniki z 5 foldów (tzw. *test-time augmentation/ensemble*).


2. Sortujesz wszystkich 5000 klientów od **najwyższego prawdopodobieństwa do najniższego**.
3. Przesuwasz się po posortowanej liście w dół i kalkulujesz skumulowany zysk.
4. Zatrzymujesz się w punkcie, w którym prawdopodobieństwo spada poniżej Twojego bezpiecznego progu (np. $P = 0.38$, dającego bufor na błąd generalizacji) **LUB** gdy osiągniesz limit 1000 klientów.


5. 
**Złota zasada:** Jeśli tylko 450 klientów spełnia kryterium wysokiej pewności zysku, do pliku `_obs.txt` wpisujesz **tylko te 450 indeksów**. Wysłanie oferty "na siłę" do kolejnych 550 osób z niskim prawdopodobieństwem wygeneruje masę False Positives i zrujnuje Twój końcowy wynik na Leaderboardzie.



---

### Podsumowanie workflow po selekcji zmiennych:

```
[Wybrane Zmienne] 
       │
       ▼
[5-Fold Stratified CV] ──► [Trening XGB/LGBM/LogReg] ──► [Predykcje OOF]
                                                               │
                                                               ▼
[Ostateczny Wybór na TEST] ◄── [Sortowanie po P i Odcięcie] ◄── [Kalibracja Prawdopodobieństwa]
(Max 1000 ID, próg P ~ 0.35+)

```

Takie podejście – oparte na precyzyjnym sortowaniu i matematycznym odcięciu zamiast sztywnych reguł – bezpośrednio replikuje systemy generujące najwyższy zysk (ROI) w realnych kampaniach direct marketingowych.