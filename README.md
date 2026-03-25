Napisz kompletny, produkcyjnie uporządkowany skrypt Python 3.11 do backtestu strategii analizy technicznej dla jednej wybranej spółki, przeznaczony do uruchamiania na Synology NAS.

Wymagania środowiskowe:
- Kod ma działać na Synology NAS z Python 3.11 uruchamianym z terminala lub harmonogramu zadań DSM.
- Unikaj ciężkich, problematycznych zależności kompilowanych, jeśli nie są konieczne.
- Preferowane biblioteki: pandas, numpy, yfinance, ta lub pandas_ta, dataclasses, json, pathlib, logging, itertools.
- Jeśli pandas_ta jest niedostępne, użyj biblioteki ta lub zaimplementuj brakujące wskaźniki ręcznie.
- Kod ma być zapisany jako pojedynczy plik .py i działać bez notebooka.
- Dodaj logowanie do pliku i do stdout.
- Dodaj podstawową obsługę błędów sieciowych i pustych danych.

Cel:
- Pobierz dane historyczne dla wskazanego tickera z Yahoo Finance.
- Okres analizy: od 2025-03-20 do 2026-03-20.
- Interwał: 1d.
- Użyj 7 wskaźników analizy technicznej:
  1. SMA fast
  2. SMA slow
  3. RSI
  4. MACD
  5. Bollinger Bands
  6. ATR
  7. Ichimoku Cloud

Dane wejściowe:
- ticker: parametr wejściowy, np. AAPL
- start_date = "2025-03-20"
- end_date = "2026-03-20"
- initial_capital = 10000.0
- commission_pct = 0.001

Parametry początkowe strategii:
- sma_fast = 20
- sma_slow = 50
- rsi_period = 14
- rsi_buy = 35
- rsi_sell = 65
- macd_fast = 12
- macd_slow = 26
- macd_signal = 9
- bb_period = 20
- bb_std = 2.0
- atr_period = 14
- atr_stop_mult = 2.0
- ichimoku_tenkan = 9
- ichimoku_kijun = 26
- ichimoku_senkou_b = 52
- cooldown_days = 3

Wymagania dotyczące Ichimoku:
- Wylicz:
  - Tenkan-sen
  - Kijun-sen
  - Senkou Span A
  - Senkou Span B
  - Chikou Span, jeśli potrzebny do logiki
- Dla sygnału BUY Ichimoku powinien wspierać wejście, np.:
  - cena nad chmurą lub
  - Tenkan > Kijun lub
  - bullish crossover Tenkan/Kijun
- Dla sygnału SELL Ichimoku powinien wspierać wyjście, np.:
  - cena pod chmurą lub
  - Tenkan < Kijun lub
  - bearish crossover Tenkan/Kijun

Logika sygnałów:
- Każdy z 7 wskaźników ma zwrócić osobny sygnał cząstkowy:
  - +1 = bullish
  -  0 = neutral
  - -1 = bearish
- Oblicz sumaryczny score dzienny.
- BUY:
  - gdy brak otwartej pozycji
  - oraz score >= +4
  - oraz co najmniej 4 z 7 wskaźników są bullish
  - oraz cooldown po poprzednim SELL już minął
- SELL:
  - gdy pozycja jest otwarta
  - oraz score <= -4
  - lub aktywuje się ATR stop loss
  - lub twardy sygnał odwrócenia z Ichimoku i MACD jednocześnie
- Transakcje wykonuj na cenie Close danego dnia.
- Jednocześnie może być otwarta tylko 1 pozycja.
- Brak shortów.

Zasady transakcji:
- Kupuj za 100% dostępnego kapitału po odjęciu prowizji.
- Przy sprzedaży uwzględnij prowizję.
- Dla każdej transakcji zapisz:
  - entry_date
  - entry_price
  - exit_date
  - exit_price
  - shares
  - pnl_abs
  - pnl_pct
  - hold_days
  - reason_exit
  - buy_score
  - sell_score

Backtest:
- Przejdź dzień po dniu przez dane.
- Dodaj wszystkie wskaźniki do DataFrame.
- Dla każdego dnia wylicz sygnały cząstkowe.
- Symuluj transakcje.
- Na końcu oblicz:
  - final_capital
  - total_return_pct
  - number_of_trades
  - win_rate
  - profit_factor
  - max_drawdown
  - best_trade_pct
  - worst_trade_pct
  - average_trade_pct
  - exposure_pct
  - buy_and_hold_return_pct

Walidacja i overfitting:
- Podziel dane na dwa etapy:
  1. in_sample: 2025-03-20 do 2025-12-31
  2. out_of_sample: 2026-01-01 do 2026-03-20
- Optymalizację parametrów wykonuj wyłącznie na in_sample.
- Po znalezieniu najlepszego setupu uruchom finalny backtest na całym okresie oraz osobno na out_of_sample.
- Nie optymalizuj bezpośrednio na całym okresie jako jedynym kryterium.

Optymalizacja:
- Jeżeli bazowy wynik in_sample jest ujemny lub niesatysfakcjonujący, uruchom prostą optymalizację grid search.
- Zmieniaj tylko te parametry:
  - sma_fast in [10, 15, 20, 30]
  - sma_slow in [40, 50, 100, 150]
  - rsi_buy in [25, 30, 35, 40]
  - rsi_sell in [60, 65, 70, 75]
  - bb_std in [1.8, 2.0, 2.2]
  - atr_stop_mult in [1.5, 2.0, 2.5, 3.0]
  - ichimoku_tenkan in [7, 9, 12]
  - ichimoku_kijun in [22, 26, 30]
  - ichimoku_senkou_b in [44, 52, 60]
- Pomiń kombinacje:
  - sma_fast >= sma_slow
  - ichimoku_tenkan >= ichimoku_kijun
  - ichimoku_kijun >= ichimoku_senkou_b
- Dla każdej kombinacji wykonaj pełny backtest in_sample.
- Zapisz wszystkie przetestowane kombinacje i ich wyniki do optimization_results.csv.

Kryterium akceptacji setupu:
- Setup uznaj za dodatni i zatrzymaj dalsze testy tylko wtedy, gdy jednocześnie:
  - total_return_pct > 0
  - number_of_trades >= 5
  - max_drawdown <= 20
  - profit_factor > 1.05
- Jeśli pierwszy taki setup zostanie znaleziony, można zatrzymać dalszą optymalizację.
- Jeśli żaden setup nie spełni warunków, wybierz najlepszy według kolejności:
  1. najwyższy total_return_pct
  2. niższy max_drawdown
  3. wyższy profit_factor

Wyniki i pliki:
- Zapisz do katalogu roboczego:
  - trades.csv
  - summary.json
  - best_setup.json
  - optimization_results.csv
  - equity_curve.csv
  - strategy.log
- Jeśli wynik dodatni i setup zaakceptowany:
  - zapisz best_setup.json
  - zakończ działanie sukcesem
- Jeśli wynik nie spełnia warunków:
  - zapisz najlepszy znaleziony setup mimo wszystko
  - zakończ działanie z czytelnym komunikatem

Wymagania techniczne kodu:
- Użyj @dataclass do parametrów strategii.
- Użyj pathlib.Path do ścieżek.
- Użyj logging.
- Zadbaj o czytelną strukturę funkcji.
- Dodaj typing hints.
- Dodaj sekcję if __name__ == "__main__":.
- Dodaj możliwość podania tickera i katalogu wyjściowego przez argparse.
- Dodaj domyślny output_dir, np. ./backtest_output.

Wymagane funkcje:
- download_data(ticker, start, end) -> pd.DataFrame
- split_data(df) -> tuple[pd.DataFrame, pd.DataFrame]
- add_indicators(df, params) -> pd.DataFrame
- score_signals(row, params, position_state) -> dict
- run_backtest(df, params, initial_capital=10000.0, commission_pct=0.001) -> tuple[pd.DataFrame, dict, pd.DataFrame]
- optimize_strategy(df_in_sample, base_params) -> tuple[StrategyParams, dict, pd.DataFrame]
- save_results(output_dir, trades_df, summary, best_params, optimization_df, equity_df) -> None

Dodatkowe wymagania pod Synology:
- Kod ma działać z wiersza poleceń:
  python3 strategy_backtest.py --ticker AAPL --output-dir /volume1/backtests/AAPL
- Nie używaj GUI.
- Nie używaj bibliotek wymagających środowiska notebookowego.
- Zadbaj, aby zapis plików był odporny na brak katalogu docelowego.
- Dodaj przykład polecenia uruchomienia oraz przykładowy wpis do harmonogramu zadań DSM jako komentarz na końcu pliku.

Na końcu wygeneruj pełny, kompletny kod, a nie szkic.