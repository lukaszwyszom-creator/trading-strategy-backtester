from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
	from ta.momentum import RSIIndicator
	from ta.trend import IchimokuIndicator, MACD, SMAIndicator
	from ta.volatility import AverageTrueRange, BollingerBands
except ImportError as exc:  # pragma: no cover - import guard for deployment
	raise SystemExit(
		"Brakuje biblioteki 'ta'. Zainstaluj zaleznosci z requirements.txt."
	) from exc


DEFAULT_START_DATE = "2025-03-20"
DEFAULT_END_DATE = "2026-03-20"
DEFAULT_INTERVAL = "1d"
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_PCT = 0.001
IN_SAMPLE_END_DATE = "2025-12-31"
OUT_OF_SAMPLE_START_DATE = "2026-01-01"

EXPECTED_TRADE_COLUMNS = [
	"entry_date",
	"entry_price",
	"exit_date",
	"exit_price",
	"shares",
	"pnl_abs",
	"pnl_pct",
	"hold_days",
	"reason_exit",
	"buy_score",
	"sell_score",
]


@dataclass(frozen=True)
class StrategyParams:
	sma_fast: int = 20
	sma_slow: int = 50
	rsi_period: int = 14
	rsi_buy: int = 35
	rsi_sell: int = 65
	macd_fast: int = 12
	macd_slow: int = 26
	macd_signal: int = 9
	bb_period: int = 20
	bb_std: float = 2.0
	atr_period: int = 14
	atr_stop_mult: float = 2.0
	ichimoku_tenkan: int = 9
	ichimoku_kijun: int = 26
	ichimoku_senkou_b: int = 52
	cooldown_days: int = 3


def setup_logging(output_dir: Path) -> logging.Logger:
	output_dir.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	for handler in list(logger.handlers):
		logger.removeHandler(handler)

	formatter = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

	file_handler = logging.FileHandler(output_dir / "strategy.log", encoding="utf-8")
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
	logger = logging.getLogger(__name__)
	attempts = 3
	inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

	for attempt in range(1, attempts + 1):
		try:
			logger.info("Pobieranie danych dla %s, proba %s/%s", ticker, attempt, attempts)
			data = yf.download(
				tickers=ticker,
				start=start,
				end=inclusive_end,
				interval=DEFAULT_INTERVAL,
				auto_adjust=False,
				progress=False,
				threads=False,
			)
			if data.empty:
				raise ValueError(f"Brak danych dla tickera {ticker}")

			if isinstance(data.columns, pd.MultiIndex):
				data.columns = data.columns.get_level_values(0)

			normalized = data.rename(columns={column: str(column).lower() for column in data.columns})
			required_columns = {"open", "high", "low", "close", "volume"}
			missing_columns = required_columns.difference(normalized.columns)
			if missing_columns:
				raise ValueError(f"Brak wymaganych kolumn danych: {sorted(missing_columns)}")

			normalized = normalized.loc[:, ["open", "high", "low", "close", "volume"]].copy()
			normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
			normalized = normalized.sort_index().dropna(subset=["open", "high", "low", "close"])
			if normalized.empty:
				raise ValueError(f"Dane po oczyszczeniu sa puste dla tickera {ticker}")
			return normalized
		except Exception as exc:
			logger.warning("Nie udalo sie pobrac danych: %s", exc)
			if attempt == attempts:
				raise
			time.sleep(2 * attempt)

	raise RuntimeError("Nieoczekiwany blad pobierania danych")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	in_sample = df.loc[(df.index >= DEFAULT_START_DATE) & (df.index <= IN_SAMPLE_END_DATE)].copy()
	out_of_sample = df.loc[
		(df.index >= OUT_OF_SAMPLE_START_DATE) & (df.index <= DEFAULT_END_DATE)
	].copy()
	return in_sample, out_of_sample


def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
	enriched = df.copy()

	enriched["sma_fast"] = SMAIndicator(
		close=enriched["close"], window=params.sma_fast, fillna=False
	).sma_indicator()
	enriched["sma_slow"] = SMAIndicator(
		close=enriched["close"], window=params.sma_slow, fillna=False
	).sma_indicator()

	enriched["rsi"] = RSIIndicator(
		close=enriched["close"], window=params.rsi_period, fillna=False
	).rsi()

	macd_indicator = MACD(
		close=enriched["close"],
		window_slow=params.macd_slow,
		window_fast=params.macd_fast,
		window_sign=params.macd_signal,
		fillna=False,
	)
	enriched["macd_line"] = macd_indicator.macd()
	enriched["macd_signal"] = macd_indicator.macd_signal()
	enriched["macd_hist"] = macd_indicator.macd_diff()

	bollinger = BollingerBands(
		close=enriched["close"], window=params.bb_period, window_dev=params.bb_std, fillna=False
	)
	enriched["bb_mid"] = bollinger.bollinger_mavg()
	enriched["bb_upper"] = bollinger.bollinger_hband()
	enriched["bb_lower"] = bollinger.bollinger_lband()

	enriched["atr"] = AverageTrueRange(
		high=enriched["high"],
		low=enriched["low"],
		close=enriched["close"],
		window=params.atr_period,
		fillna=False,
	).average_true_range()

	ichimoku = IchimokuIndicator(
		high=enriched["high"],
		low=enriched["low"],
		window1=params.ichimoku_tenkan,
		window2=params.ichimoku_kijun,
		window3=params.ichimoku_senkou_b,
		visual=False,
		fillna=False,
	)
	enriched["tenkan_sen"] = ichimoku.ichimoku_conversion_line()
	enriched["kijun_sen"] = ichimoku.ichimoku_base_line()
	enriched["senkou_span_a"] = ichimoku.ichimoku_a()
	enriched["senkou_span_b"] = ichimoku.ichimoku_b()
	enriched["chikou_span"] = enriched["close"].shift(params.ichimoku_kijun)

	enriched["prev_close"] = enriched["close"].shift(1)
	enriched["prev_macd_line"] = enriched["macd_line"].shift(1)
	enriched["prev_macd_signal"] = enriched["macd_signal"].shift(1)
	enriched["prev_tenkan_sen"] = enriched["tenkan_sen"].shift(1)
	enriched["prev_kijun_sen"] = enriched["kijun_sen"].shift(1)
	return enriched


def _safe_float(value: Any) -> float | None:
	if value is None or pd.isna(value):
		return None
	return float(value)


def _to_builtin(value: Any) -> Any:
	if isinstance(value, dict):
		return {key: _to_builtin(inner_value) for key, inner_value in value.items()}
	if isinstance(value, list):
		return [_to_builtin(item) for item in value]
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.floating, float)):
		if math.isfinite(float(value)):
			return float(value)
		return None
	if isinstance(value, (pd.Timestamp,)):
		return value.strftime("%Y-%m-%d")
	return value


def _acceptance_criteria(summary: dict[str, Any]) -> bool:
	return (
		summary.get("total_return_pct", 0.0) > 0.0
		and summary.get("number_of_trades", 0) >= 5
		and summary.get("max_drawdown", float("inf")) <= 20.0
		and summary.get("profit_factor", 0.0) > 1.05
	)


def _build_optimization_row(
	params: StrategyParams, summary: dict[str, Any], source: str, accepted: bool
) -> dict[str, Any]:
	row = asdict(params)
	row.update(
		{
			"source": source,
			"accepted": accepted,
			"final_capital": summary.get("final_capital", 0.0),
			"total_return_pct": summary.get("total_return_pct", 0.0),
			"number_of_trades": summary.get("number_of_trades", 0),
			"win_rate": summary.get("win_rate", 0.0),
			"profit_factor": summary.get("profit_factor", 0.0),
			"max_drawdown": summary.get("max_drawdown", 0.0),
			"best_trade_pct": summary.get("best_trade_pct", 0.0),
			"worst_trade_pct": summary.get("worst_trade_pct", 0.0),
			"average_trade_pct": summary.get("average_trade_pct", 0.0),
			"exposure_pct": summary.get("exposure_pct", 0.0),
			"buy_and_hold_return_pct": summary.get("buy_and_hold_return_pct", 0.0),
		}
	)
	return row


def _choose_better_candidate(
	current_best: tuple[StrategyParams, dict[str, Any]] | None,
	candidate_params: StrategyParams,
	candidate_summary: dict[str, Any],
) -> tuple[StrategyParams, dict[str, Any]]:
	if current_best is None:
		return candidate_params, candidate_summary

	best_params, best_summary = current_best
	candidate_rank = (
		candidate_summary.get("total_return_pct", float("-inf")),
		-candidate_summary.get("max_drawdown", float("inf")),
		candidate_summary.get("profit_factor", float("-inf")),
	)
	best_rank = (
		best_summary.get("total_return_pct", float("-inf")),
		-best_summary.get("max_drawdown", float("inf")),
		best_summary.get("profit_factor", float("-inf")),
	)
	if candidate_rank > best_rank:
		return candidate_params, candidate_summary
	return best_params, best_summary


def score_signals(
	row: pd.Series, params: StrategyParams, position_state: dict[str, Any]
) -> dict[str, Any]:
	signals: dict[str, int] = {
		"sma": 0,
		"rsi": 0,
		"macd": 0,
		"bollinger": 0,
		"atr": 0,
		"ichimoku": 0,
		"price_action": 0,
	}

	sma_fast = _safe_float(row.get("sma_fast"))
	sma_slow = _safe_float(row.get("sma_slow"))
	if sma_fast is not None and sma_slow is not None:
		signals["sma"] = 1 if sma_fast > sma_slow else -1 if sma_fast < sma_slow else 0

	rsi = _safe_float(row.get("rsi"))
	if rsi is not None:
		if rsi <= params.rsi_buy:
			signals["rsi"] = 1
		elif rsi >= params.rsi_sell:
			signals["rsi"] = -1

	macd_line = _safe_float(row.get("macd_line"))
	macd_signal = _safe_float(row.get("macd_signal"))
	prev_macd_line = _safe_float(row.get("prev_macd_line"))
	prev_macd_signal = _safe_float(row.get("prev_macd_signal"))
	if macd_line is not None and macd_signal is not None:
		signals["macd"] = 1 if macd_line > macd_signal else -1 if macd_line < macd_signal else 0

	close_price = _safe_float(row.get("close"))
	bb_upper = _safe_float(row.get("bb_upper"))
	bb_lower = _safe_float(row.get("bb_lower"))
	if close_price is not None and bb_upper is not None and bb_lower is not None:
		if close_price < bb_lower:
			signals["bollinger"] = 1
		elif close_price > bb_upper:
			signals["bollinger"] = -1

	prev_close = _safe_float(row.get("prev_close"))
	atr_value = _safe_float(row.get("atr"))
	if close_price is not None and prev_close is not None and atr_value is not None and atr_value > 0:
		threshold = atr_value * 0.25
		change = close_price - prev_close
		if change > threshold:
			signals["atr"] = 1
		elif change < -threshold:
			signals["atr"] = -1

	tenkan = _safe_float(row.get("tenkan_sen"))
	kijun = _safe_float(row.get("kijun_sen"))
	senkou_a = _safe_float(row.get("senkou_span_a"))
	senkou_b = _safe_float(row.get("senkou_span_b"))
	prev_tenkan = _safe_float(row.get("prev_tenkan_sen"))
	prev_kijun = _safe_float(row.get("prev_kijun_sen"))

	ichimoku_bull_cross = False
	ichimoku_bear_cross = False
	if (
		tenkan is not None
		and kijun is not None
		and prev_tenkan is not None
		and prev_kijun is not None
	):
		ichimoku_bull_cross = prev_tenkan <= prev_kijun and tenkan > kijun
		ichimoku_bear_cross = prev_tenkan >= prev_kijun and tenkan < kijun

	if close_price is not None and senkou_a is not None and senkou_b is not None and tenkan is not None and kijun is not None:
		cloud_top = max(senkou_a, senkou_b)
		cloud_bottom = min(senkou_a, senkou_b)
		ichimoku_bullish = close_price > cloud_top or tenkan > kijun or ichimoku_bull_cross
		ichimoku_bearish = close_price < cloud_bottom or tenkan < kijun or ichimoku_bear_cross
		if ichimoku_bullish and not ichimoku_bearish:
			signals["ichimoku"] = 1
		elif ichimoku_bearish and not ichimoku_bullish:
			signals["ichimoku"] = -1

	sma_fast_for_price = _safe_float(row.get("sma_fast"))
	if close_price is not None and sma_fast_for_price is not None:
		signals["price_action"] = (
			1 if close_price > sma_fast_for_price else -1 if close_price < sma_fast_for_price else 0
		)

	atr_stop_hit = False
	trailing_stop = position_state.get("trailing_stop")
	if position_state.get("in_position") and close_price is not None and trailing_stop is not None:
		atr_stop_hit = close_price <= float(trailing_stop)

	hard_reversal = signals["ichimoku"] == -1 and signals["macd"] == -1 and (
		ichimoku_bear_cross
		or (
			prev_macd_line is not None
			and prev_macd_signal is not None
			and macd_line is not None
			and macd_signal is not None
			and prev_macd_line >= prev_macd_signal
			and macd_line < macd_signal
		)
	)

	bullish_count = sum(1 for signal_value in signals.values() if signal_value == 1)
	bearish_count = sum(1 for signal_value in signals.values() if signal_value == -1)
	daily_score = int(sum(signals.values()))

	return {
		"signals": signals,
		"daily_score": daily_score,
		"bullish_count": bullish_count,
		"bearish_count": bearish_count,
		"atr_stop_hit": atr_stop_hit,
		"hard_reversal": hard_reversal,
	}


def _build_summary(
	price_df: pd.DataFrame,
	trades_df: pd.DataFrame,
	equity_df: pd.DataFrame,
	initial_capital: float,
) -> dict[str, Any]:
	final_capital = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else float(initial_capital)
	total_return_pct = ((final_capital / initial_capital) - 1.0) * 100.0 if initial_capital else 0.0

	number_of_trades = int(len(trades_df))
	if number_of_trades:
		winning_trades = int((trades_df["pnl_abs"] > 0).sum())
		win_rate = (winning_trades / number_of_trades) * 100.0
		gross_profit = float(trades_df.loc[trades_df["pnl_abs"] > 0, "pnl_abs"].sum())
		gross_loss = abs(float(trades_df.loc[trades_df["pnl_abs"] < 0, "pnl_abs"].sum()))
		if gross_loss > 0:
			profit_factor = gross_profit / gross_loss
		elif gross_profit > 0:
			profit_factor = float("inf")
		else:
			profit_factor = 0.0
		best_trade_pct = float(trades_df["pnl_pct"].max())
		worst_trade_pct = float(trades_df["pnl_pct"].min())
		average_trade_pct = float(trades_df["pnl_pct"].mean())
		exposure_pct = (
			float(trades_df["hold_days"].sum()) / max(len(price_df), 1)
		) * 100.0
	else:
		win_rate = 0.0
		profit_factor = 0.0
		best_trade_pct = 0.0
		worst_trade_pct = 0.0
		average_trade_pct = 0.0
		exposure_pct = 0.0

	if not equity_df.empty:
		running_peak = equity_df["equity"].cummax()
		drawdown = (equity_df["equity"] / running_peak) - 1.0
		max_drawdown = abs(float(drawdown.min()) * 100.0)
	else:
		max_drawdown = 0.0

	if not price_df.empty:
		buy_and_hold_return_pct = (
			(float(price_df["close"].iloc[-1]) / float(price_df["close"].iloc[0]) - 1.0) * 100.0
		)
	else:
		buy_and_hold_return_pct = 0.0

	return {
		"final_capital": round(final_capital, 2),
		"total_return_pct": round(total_return_pct, 2),
		"number_of_trades": number_of_trades,
		"win_rate": round(win_rate, 2),
		"profit_factor": profit_factor if not math.isfinite(profit_factor) else round(profit_factor, 4),
		"max_drawdown": round(max_drawdown, 2),
		"best_trade_pct": round(best_trade_pct, 2),
		"worst_trade_pct": round(worst_trade_pct, 2),
		"average_trade_pct": round(average_trade_pct, 2),
		"exposure_pct": round(exposure_pct, 2),
		"buy_and_hold_return_pct": round(buy_and_hold_return_pct, 2),
	}


def run_backtest(
	df: pd.DataFrame,
	params: StrategyParams,
	initial_capital: float = DEFAULT_INITIAL_CAPITAL,
	commission_pct: float = DEFAULT_COMMISSION_PCT,
	log_trades: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
	logger = logging.getLogger(__name__)
	enriched = add_indicators(df, params)

	cash = float(initial_capital)
	shares = 0.0
	last_sell_date: pd.Timestamp | None = None
	entry_date: pd.Timestamp | None = None
	entry_price = 0.0
	entry_capital = 0.0
	buy_score = 0
	highest_close_since_entry = 0.0
	last_signal_snapshot: dict[str, Any] = {
		"daily_score": 0,
		"atr_stop_hit": False,
		"hard_reversal": False,
	}

	trades: list[dict[str, Any]] = []
	equity_records: list[dict[str, Any]] = []

	for current_date, row in enriched.iterrows():
		close_price = float(row["close"])

		if shares > 0:
			highest_close_since_entry = max(highest_close_since_entry, close_price)
			atr_value = _safe_float(row.get("atr")) or 0.0
			trailing_stop = highest_close_since_entry - (atr_value * params.atr_stop_mult)
		else:
			trailing_stop = None

		position_state = {
			"in_position": shares > 0,
			"entry_price": entry_price if shares > 0 else None,
			"trailing_stop": trailing_stop,
		}
		signal_snapshot = score_signals(row, params, position_state)
		last_signal_snapshot = signal_snapshot

		if shares == 0:
			cooldown_complete = (
				last_sell_date is None
				or (current_date - last_sell_date).days >= params.cooldown_days
			)
			should_buy = (
				signal_snapshot["daily_score"] >= 4
				and signal_snapshot["bullish_count"] >= 4
				and cooldown_complete
				and close_price > 0
			)

			if should_buy:
				entry_capital = cash
				shares = (cash * (1.0 - commission_pct)) / close_price
				cash = 0.0
				entry_date = current_date
				entry_price = close_price
				buy_score = signal_snapshot["daily_score"]
				highest_close_since_entry = close_price
				if log_trades:
					logger.info(
						"BUY | %s | price=%.2f | shares=%.6f | score=%s",
						current_date.strftime("%Y-%m-%d"),
						close_price,
						shares,
						buy_score,
					)
		else:
			should_sell = (
				signal_snapshot["daily_score"] <= -4
				or signal_snapshot["atr_stop_hit"]
				or signal_snapshot["hard_reversal"]
			)
			if should_sell and entry_date is not None:
				if signal_snapshot["atr_stop_hit"]:
					reason_exit = "atr_stop_loss"
				elif signal_snapshot["hard_reversal"]:
					reason_exit = "ichimoku_macd_reversal"
				else:
					reason_exit = "score_reversal"

				exit_value = shares * close_price * (1.0 - commission_pct)
				pnl_abs = exit_value - entry_capital
				pnl_pct = (pnl_abs / entry_capital) * 100.0 if entry_capital else 0.0

				trades.append(
					{
						"entry_date": entry_date.strftime("%Y-%m-%d"),
						"entry_price": round(entry_price, 4),
						"exit_date": current_date.strftime("%Y-%m-%d"),
						"exit_price": round(close_price, 4),
						"shares": round(shares, 8),
						"pnl_abs": round(pnl_abs, 2),
						"pnl_pct": round(pnl_pct, 2),
						"hold_days": int((current_date - entry_date).days),
						"reason_exit": reason_exit,
						"buy_score": buy_score,
						"sell_score": signal_snapshot["daily_score"],
					}
				)

				if log_trades:
					logger.info(
						"SELL | %s | price=%.2f | pnl=%.2f | reason=%s",
						current_date.strftime("%Y-%m-%d"),
						close_price,
						pnl_abs,
						reason_exit,
					)

				cash = exit_value
				shares = 0.0
				last_sell_date = current_date
				entry_date = None
				entry_price = 0.0
				entry_capital = 0.0
				buy_score = 0
				highest_close_since_entry = 0.0

		market_value = shares * close_price
		equity_records.append(
			{
				"date": current_date.strftime("%Y-%m-%d"),
				"cash": round(cash, 2),
				"market_value": round(market_value, 2),
				"equity": round(cash + market_value, 2),
				"position_open": int(shares > 0),
				"daily_score": signal_snapshot["daily_score"],
			}
		)

	if shares > 0 and entry_date is not None and not enriched.empty:
		final_date = enriched.index[-1]
		final_close = float(enriched.iloc[-1]["close"])
		exit_value = shares * final_close * (1.0 - commission_pct)
		pnl_abs = exit_value - entry_capital
		pnl_pct = (pnl_abs / entry_capital) * 100.0 if entry_capital else 0.0
		trades.append(
			{
				"entry_date": entry_date.strftime("%Y-%m-%d"),
				"entry_price": round(entry_price, 4),
				"exit_date": final_date.strftime("%Y-%m-%d"),
				"exit_price": round(final_close, 4),
				"shares": round(shares, 8),
				"pnl_abs": round(pnl_abs, 2),
				"pnl_pct": round(pnl_pct, 2),
				"hold_days": int((final_date - entry_date).days),
				"reason_exit": "end_of_period",
				"buy_score": buy_score,
				"sell_score": last_signal_snapshot["daily_score"],
			}
		)
		cash = exit_value
		equity_records[-1]["cash"] = round(cash, 2)
		equity_records[-1]["market_value"] = 0.0
		equity_records[-1]["equity"] = round(cash, 2)
		equity_records[-1]["position_open"] = 0
		if log_trades:
			logger.info("Wymuszone zamkniecie pozycji na koncu okresu")

	trades_df = pd.DataFrame(trades, columns=EXPECTED_TRADE_COLUMNS)
	equity_df = pd.DataFrame(equity_records)
	summary = _build_summary(enriched, trades_df, equity_df, initial_capital)
	return trades_df, summary, equity_df


def optimize_strategy(
	df_in_sample: pd.DataFrame,
	base_params: StrategyParams,
	initial_capital: float = DEFAULT_INITIAL_CAPITAL,
	commission_pct: float = DEFAULT_COMMISSION_PCT,
) -> tuple[StrategyParams, dict[str, Any], pd.DataFrame]:
	logger = logging.getLogger(__name__)
	optimization_rows: list[dict[str, Any]] = []
	best_candidate: tuple[StrategyParams, dict[str, Any]] | None = None

	base_trades, base_summary, _ = run_backtest(
		df_in_sample,
		base_params,
		initial_capital=initial_capital,
		commission_pct=commission_pct,
		log_trades=False,
	)
	del base_trades
	base_accepted = _acceptance_criteria(base_summary)
	optimization_rows.append(
		_build_optimization_row(base_params, base_summary, source="baseline", accepted=base_accepted)
	)
	best_candidate = _choose_better_candidate(None, base_params, base_summary)
	if base_accepted:
		return base_params, base_summary, pd.DataFrame(optimization_rows)

	parameter_grid = itertools.product(
		[10, 15, 20, 30],
		[40, 50, 100, 150],
		[25, 30, 35, 40],
		[60, 65, 70, 75],
		[1.8, 2.0, 2.2],
		[1.5, 2.0, 2.5, 3.0],
		[7, 9, 12],
		[22, 26, 30],
		[44, 52, 60],
	)

	for index, combo in enumerate(parameter_grid, start=1):
		(
			sma_fast,
			sma_slow,
			rsi_buy,
			rsi_sell,
			bb_std,
			atr_stop_mult,
			ichimoku_tenkan,
			ichimoku_kijun,
			ichimoku_senkou_b,
		) = combo

		if sma_fast >= sma_slow:
			continue
		if ichimoku_tenkan >= ichimoku_kijun:
			continue
		if ichimoku_kijun >= ichimoku_senkou_b:
			continue

		candidate_params = replace(
			base_params,
			sma_fast=sma_fast,
			sma_slow=sma_slow,
			rsi_buy=rsi_buy,
			rsi_sell=rsi_sell,
			bb_std=bb_std,
			atr_stop_mult=atr_stop_mult,
			ichimoku_tenkan=ichimoku_tenkan,
			ichimoku_kijun=ichimoku_kijun,
			ichimoku_senkou_b=ichimoku_senkou_b,
		)
		_, candidate_summary, _ = run_backtest(
			df_in_sample,
			candidate_params,
			initial_capital=initial_capital,
			commission_pct=commission_pct,
			log_trades=False,
		)
		accepted = _acceptance_criteria(candidate_summary)
		optimization_rows.append(
			_build_optimization_row(candidate_params, candidate_summary, source="grid_search", accepted=accepted)
		)
		best_candidate = _choose_better_candidate(best_candidate, candidate_params, candidate_summary)

		if index % 250 == 0:
			logger.info("Przetestowano %s kombinacji optymalizacji", index)

		if accepted:
			logger.info("Znaleziono akceptowalny setup po %s kombinacjach", index)
			return candidate_params, candidate_summary, pd.DataFrame(optimization_rows)

	if best_candidate is None:
		raise RuntimeError("Optymalizacja nie zwrocila zadnego wyniku")

	best_params, best_summary = best_candidate
	return best_params, best_summary, pd.DataFrame(optimization_rows)


def save_results(
	output_dir: Path,
	trades_df: pd.DataFrame,
	summary: dict[str, Any],
	best_params: StrategyParams,
	optimization_df: pd.DataFrame,
	equity_df: pd.DataFrame,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	trades_df.to_csv(output_dir / "trades.csv", index=False)
	equity_df.to_csv(output_dir / "equity_curve.csv", index=False)
	optimization_df.to_csv(output_dir / "optimization_results.csv", index=False)

	summary_payload = _to_builtin(summary)
	with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
		json.dump(summary_payload, summary_file, indent=2, ensure_ascii=False)

	best_setup_payload = {
		"params": _to_builtin(asdict(best_params)),
		"accepted": summary_payload.get("accepted_setup", False),
		"selected_by": summary_payload.get("selected_by", "unknown"),
		"in_sample": summary_payload.get("in_sample", {}),
		"out_of_sample": summary_payload.get("out_of_sample", {}),
		"full_period": summary_payload.get("full_period", {}),
	}
	with (output_dir / "best_setup.json").open("w", encoding="utf-8") as best_setup_file:
		json.dump(best_setup_payload, best_setup_file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Backtest strategii analizy technicznej")
	parser.add_argument("--ticker", required=True, help="Ticker instrumentu, np. AAPL")
	parser.add_argument(
		"--output-dir",
		default="./backtest_output",
		help="Katalog wyjsciowy na wyniki backtestu",
	)
	parser.add_argument(
		"--initial-capital",
		type=float,
		default=DEFAULT_INITIAL_CAPITAL,
		help="Kapital poczatkowy",
	)
	parser.add_argument(
		"--commission-pct",
		type=float,
		default=DEFAULT_COMMISSION_PCT,
		help="Prowizja procentowa, np. 0.001 = 0.1%%",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	output_dir = Path(args.output_dir)
	logger = setup_logging(output_dir)
	ticker = args.ticker.upper().strip()

	logger.info("Start backtestu dla %s", ticker)
	try:
		full_df = download_data(ticker, DEFAULT_START_DATE, DEFAULT_END_DATE)
		in_sample_df, out_of_sample_df = split_data(full_df)

		if in_sample_df.empty:
			raise ValueError("Zbior in-sample jest pusty")
		if out_of_sample_df.empty:
			raise ValueError("Zbior out-of-sample jest pusty")

		base_params = StrategyParams()
		_, base_in_sample_summary, _ = run_backtest(
			in_sample_df,
			base_params,
			initial_capital=args.initial_capital,
			commission_pct=args.commission_pct,
			log_trades=False,
		)

		if _acceptance_criteria(base_in_sample_summary):
			best_params = base_params
			best_in_sample_summary = base_in_sample_summary
			optimization_df = pd.DataFrame(
				[_build_optimization_row(base_params, base_in_sample_summary, "baseline", True)]
			)
			selected_by = "baseline"
			logger.info("Bazowy setup spelnia kryteria akceptacji")
		else:
			logger.info("Bazowy setup nie spelnia kryteriow, start optymalizacji")
			best_params, best_in_sample_summary, optimization_df = optimize_strategy(
				in_sample_df,
				base_params,
				initial_capital=args.initial_capital,
				commission_pct=args.commission_pct,
			)
			selected_by = "grid_search"

		trades_df, full_summary, equity_df = run_backtest(
			full_df,
			best_params,
			initial_capital=args.initial_capital,
			commission_pct=args.commission_pct,
		)
		_, out_of_sample_summary, _ = run_backtest(
			out_of_sample_df,
			best_params,
			initial_capital=args.initial_capital,
			commission_pct=args.commission_pct,
		)

		accepted_setup = _acceptance_criteria(best_in_sample_summary)
		summary = {
			"ticker": ticker,
			"selected_by": selected_by,
			"accepted_setup": accepted_setup,
			"period": {
				"start_date": DEFAULT_START_DATE,
				"end_date": DEFAULT_END_DATE,
				"interval": DEFAULT_INTERVAL,
				"in_sample_end": IN_SAMPLE_END_DATE,
				"out_of_sample_start": OUT_OF_SAMPLE_START_DATE,
			},
			"initial_capital": float(args.initial_capital),
			"commission_pct": float(args.commission_pct),
			"best_params": asdict(best_params),
			"in_sample": best_in_sample_summary,
			"out_of_sample": out_of_sample_summary,
			"full_period": full_summary,
		}

		save_results(output_dir, trades_df, summary, best_params, optimization_df, equity_df)

		if accepted_setup and full_summary.get("total_return_pct", 0.0) > 0.0:
			logger.info("Backtest zakonczony sukcesem. Wyniki zapisane w %s", output_dir)
			return 0

		logger.warning(
			"Najlepszy setup zapisany, ale nie spelnia wszystkich warunkow akceptacji lub wynik koncowy nie jest dodatni."
		)
		return 1
	except Exception as exc:
		logger.exception("Backtest zakonczony bledem: %s", exc)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())


# Przyklad uruchomienia na Synology:
# python3 strategy_backtest.py --ticker AAPL --output-dir /volume1/backtests/AAPL
#
# Przykladowy wpis do Harmonogramu zadan DSM:
# /usr/bin/python3 /volume1/projects/trading-strategy-backtester/strategy_backtest.py \
#   --ticker AAPL \
#   --output-dir /volume1/backtests/AAPL
