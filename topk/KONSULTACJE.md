# Konsultacje D03 (07.12) – co pokazać i jak odpowiadać

## 1. Demo (ok. 10 minut, w tej kolejności)

```bash
cd PRiR/topk
make build                       # wszystkie wersje, bez ostrzeżeń
python3 tests/check.py --quick   # tabela OK dla ref / omp / mpi / cuda / cublas
d=data/n100000_d128
reference/topk_ref $d/base.bin $d/queries.bin 100000 128 1000 10 --repeat 1 --warmup 0
openmp/topk_omp    $d/base.bin $d/queries.bin 100000 128 1000 10 --threads 16
mpirun -np 8 mpi/topk_mpi $d/base.bin $d/queries.bin 100000 128 1000 10
cuda/topk_cuda     $d/base.bin $d/queries.bin 100000 128 1000 10
cuda/topk_cublas   $d/base.bin $d/queries.bin 100000 128 1000 10
cat results/ranking_check.csv    # zgodność rankingów z referencją
```

Potem sprawozdanie: tabela przypadku głównego, wykresy S(p)/E(p) dla OpenMP i MPI,
wykres CUDA (bloki, cuBLAS), wpływ N, D, k, Q. Wszystkie liczby pochodzą z
`results/raw.csv` przez `scripts/plots.py` – można to pokazać (`make plots`).

## 2. Każda wersja w 2–3 zdaniach

- **Referencja:** dla każdego zapytania liczę kwadrat odległości do wszystkich N wektorów,
  `std::partial_sort` wybiera k najmniejszych par (odległość, indeks). Para porównuje się
  najpierw po odległości, potem po indeksie, więc remis wygrywa mniejszy indeks.
- **OpenMP:** przy dużej partii wątki dzielą się zapytaniami (`schedule(dynamic, 1)`), każdy ma
  własny kopiec k najlepszych – nie ma sekcji krytycznej. Przy jednym zapytaniu dzielą się bazą
  (`schedule(static)`), każdy ma lokalny kopiec, a na końcu scalam p·k kandydatów.
- **MPI:** `MPI_Scatterv` raz dzieli bazę, potem dla każdej partii `MPI_Bcast` zapytań, każdy
  proces liczy lokalne top-k na swojej części, `MPI_Gather` zbiera k kandydatów od każdego
  procesu dla każdego zapytania, a proces 0 je scala. Pomiar lokalny, na jednym komputerze.
- **CUDA:** jeden blok = jedno zapytanie. Zapytanie w pamięci współdzielonej, każdy wątek
  przegląda co T-ty wektor bazy i trzyma swoją listę k najlepszych, potem listy są scalane
  drzewiasto w log2(T) krokach. Baza jest kopiowana na GPU raz.
- **cuBLAS:** ‖q−b‖² = ‖q‖² + ‖b‖² − 2q·b; biblioteka liczy tylko iloczyny q·b (jedno SGEMM),
  normy, złożenie odległości i wybór top-k to moje kernele.

## 3. Spodziewane pytania i krótkie odpowiedzi

1. **Po co `MPI_Scatterv`, a nie `MPI_Scatter`?** N nie musi dzielić się przez p. `counts[r]` =
   N/p wektorów (+1 dla pierwszych N % p procesów), `displs[r]` = początek części procesu r.
   Oba liczone w liczbach `float`, więc mnożone przez D.
2. **Co dokładnie przesyła MPI?** Raz: część bazy do każdego procesu. Na partię: zapytania
   (Bcast, Q·D liczb) i z powrotem Q·k par (indeks, kwadrat odległości) od każdego procesu.
   Nigdy całej macierzy odległości Q×N.
3. **Czemu MPI na jednym komputerze jest wolniejsze od OpenMP?** Przy tej samej liczbie
   wykonawców (8) czasy są bliskie; komunikacja na partię to ok. 1 ms (tab. MPI). OpenMP
   wygrywa, bo używa 16 wątków (SMT), a MPI ma jeden proces na rdzeń. Przy N = 10^6 MPI odstaje
   bardziej: procesy czytają rozłączne części bazy, więc nie korzystają ze wspólnej pamięci
   podręcznej jak wątki OpenMP. Zysk z MPI byłby na kilku komputerach (więcej pamięci i rdzeni).
4. **Czym się różni czas rdzenia od pełnego?** Rdzeń to same obliczenia (MPI: najwolniejszy
   proces, CUDA: kernel). Pełny dodaje komunikację (Bcast, Gather, scalanie) albo kopie
   zapytań i wyników między CPU i GPU. Jednorazowe rozesłanie bazy jest w `base_ms`.
5. **Jak działa redukcja drzewiasta w kernelu?** Po przejrzeniu bazy każdy z T wątków ma
   posortowaną listę k kandydatów w shared memory. W kroku s wątki t < s scalają listę t
   z listą t+s (jak w merge sort, O(k)); s = T/2, T/4, …, 1. Po log2(T) krokach wynik ma wątek 0.
6. **Skąd limit k ≤ 64?** Każdy wątek trzyma k par po 8 B w shared memory: T·k·8 B + D·4 B na blok,
   a RTX 4060 pozwala na 99 KiB. Dla k = 64 i T = 256 to 128 KiB – za dużo, więc program
   zmniejsza T do 128. Lokalne tablice w kernelu mają rozmiar MAX_K = 64.
7. **Czemu P95 z zapytań obsługiwanych pojedynczo?** P95 opisuje opóźnienie jednego zapytania
   (tryb „online”). Czas partii podzielony przez Q to nie opóźnienie, tylko odwrotność
   przepustowości. Mierzę 100 zapytań w każdym z 7 powtórzeń, każda wersja tak samo.
8. **Jak obsługujecie remisy?** Wszędzie porównanie (odległość, indeks) – przy równej
   odległości wygrywa mniejszy indeks. Testy mają przypadki z duplikatami i kratką całkowitą.
9. **Co to E(p)?** E(p) = S(p)/p, S(p) = T(1)/T(p) przy tym samym problemie. Dla GPU nie
   liczę E(p) po wątkach – podaję czas, przepustowość i wpływ rozmiaru bloku.
10. **Dlaczego cuBLAS jest szybszy?** Mnożenie macierzy jest bardzo dobrze zoptymalizowane
    (kafelkowanie w pamięci współdzielonej i rejestrach), a mój kernel czyta bazę osobno dla każdego
    zapytania. Kosztem jest pamięć na macierz odległości (liczę kawałkami do 512 MiB)
    i mniejsza dokładność dla prawie identycznych wektorów.
11. **Po co strumienie CUDA?** Partia jest dzielona na 4 części; kopia zapytań jednej części
    może iść równolegle z kernelem innej. W pomiarach zysku nie ma (77,9 vs 78,0 ms), bo kopia
    1000 zapytań to 0,5 MB, a kernel liczy kilkadziesiąt ms - nie ma czego zasłaniać.
12. **Czemu P95 pojedynczego zapytania na GPU jest gorsze niż na CPU?** Własny kernel liczy
    jedno zapytanie jednym blokiem, czyli na jednym multiprocesorze; CPU dzieli zapytanie na
    wszystkie wątki. Kernel jest zrobiony pod przepustowość dużych partii (przy Q = 1 ma tylko
    ok. 500 QPS). cuBLAS ma najmniejsze P95, bo jedno zapytanie to mnożenie macierz-wektor na
    całej karcie. S(p) liczę względem OpenMP na 1 wątku, który ma praktycznie ten sam czas
    co referencja (0,98).
13. **Jak sprawdzacie poprawność?** `tests/check.py`: niezależna referencja numpy w float64,
    przypadki brzegowe, tolerancja 1e-4. Dla przypadku głównego `results/ranking_check.csv`
    porównuje rankingi wszystkich wersji z referencją.
14. **Czy wyniki da się odtworzyć?** `make bench` (albo `scripts/run_all.sh`) od zera: build,
    dane z ziarnem 42, testy, wszystkie pomiary z `experiments.conf`; `make plots` – wykresy
    i tabele. `results/env.txt` opisuje sprzęt i wersje programów.
15. **Wariant hybrydowy?** `topk_mpi_hybrid`: p procesów × t wątków (p·t = 16), wątki dzielą
    zapytania partii wewnątrz procesu. Uruchamiany z `--bind-to none`, żeby wątki procesu nie
    były przypięte do jednego rdzenia.

## 4. Ćwiczenia na żywo (do przećwiczenia przed konsultacją)

**a) Zmiana liczby wątków w bloku CUDA.**
```bash
for t in 64 128 256 512; do cuda/topk_cuda $d/base.bin $d/queries.bin 100000 128 1000 10 --threads $t | grep partia; done
cuda/topk_cuda $d/base.bin $d/queries.bin 100000 128 1000 64 | head -1   # k=64: blok zmniejszony do 128
```
Co powiedzieć: mniej wątków = mniej równoległości w bloku, więcej = dłuższe scalanie
(log2 T kroków) i więcej shared memory na blok, więc mniej bloków naraz na multiprocesorze.

**b) Zmiana w kernelu.** W `cuda/topk_cuda.cu` wyłączyć ścieżkę `float4` (warunek
`if (D % 4 == 0)` zamienić na `if (false)`), `make -C cuda`, porównać czas rdzenia.
Testy dalej przechodzą (`python3 tests/check.py --impl cuda --quick`).

**c) `MPI_Scatterv` na kartce.** N = 10, p = 3, D = 2: counts = [4, 3, 3] wektory, czyli
send_counts = [8, 6, 6] liczb, displs = [0, 8, 14]. Proces 1 dostaje wektory 4–6, globalny
indeks = `first[rank] + i`. Uruchomić `mpirun -np 3` dla N niepodzielnego przez 3 i pokazać,
że test „N nie dzieli się przez p” przechodzi.
