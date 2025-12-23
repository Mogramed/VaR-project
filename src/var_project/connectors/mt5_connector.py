from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

import pandas as pd

from var_project.core.exceptions import MT5ConnectionError
from var_project.core.types import MT5Config


class MT5Connector:
    """
    Wrapper propre autour de MetaTrader5 (API Python).

    Règles :
    - init() une fois au début
    - fetch (bars/ticks) ensuite
    - shutdown() à la fin

    Remarque importante :
    - copy_rates_range() est parfois très sensible aux paramètres datetime (timezone, server time…)
      => Pour un projet stable, on privilégie fetch_last_n_bars() via copy_rates_from_pos().
    """

    def __init__(self, config: MT5Config):
        self.config = config
        self._mt5 = None
        self._initialized = False

    # -----------------------------
    # Connection lifecycle
    # -----------------------------
    def init(self) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as e:
            raise MT5ConnectionError(
                "MetaTrader5 package non installé. Fais: pip install MetaTrader5"
            ) from e

        self._mt5 = mt5

        ok = mt5.initialize()
        if not ok:
            raise MT5ConnectionError(f"mt5.initialize() a échoué: {mt5.last_error()}")

        # Optionnel : login explicite si fourni
        if self.config.login and self.config.password and self.config.server:
            logged = mt5.login(
                login=int(self.config.login),
                password=str(self.config.password),
                server=str(self.config.server),
            )
            if not logged:
                raise MT5ConnectionError(f"mt5.login() a échoué: {mt5.last_error()}")

        self._initialized = True

    def shutdown(self) -> None:
        if self._mt5 and self._initialized:
            self._mt5.shutdown()
        self._initialized = False

    def _require_ready(self) -> None:
        if not self._initialized or not self._mt5:
            raise MT5ConnectionError("MT5 non initialisé : appelle init() avant")

    # -----------------------------
    # Symbol utilities
    # -----------------------------
    def ensure_symbol(self, symbol: str) -> None:
        """
        Vérifie que le symbole existe et est visible dans MarketWatch.
        """
        self._require_ready()

        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise MT5ConnectionError(
                f"Symbole inconnu dans MT5: '{symbol}'. "
                "Vérifie qu'il apparaît bien dans Market Watch (pas de suffixe du broker)."
            )

        if not info.visible:
            ok = self._mt5.symbol_select(symbol, True)
            if not ok:
                raise MT5ConnectionError(
                    f"Impossible de sélectionner '{symbol}': {self._mt5.last_error()}"
                )

    # -----------------------------
    # Timeframe utilities
    # -----------------------------
    def get_timeframe(self, timeframe: str):
        """
        Convertit un string (ex: 'M5') vers la constante MT5.
        """
        self._require_ready()

        tf = timeframe.upper().strip()
        mapping = {
            "M1": self._mt5.TIMEFRAME_M1,
            "M2": self._mt5.TIMEFRAME_M2,
            "M3": self._mt5.TIMEFRAME_M3,
            "M4": self._mt5.TIMEFRAME_M4,
            "M5": self._mt5.TIMEFRAME_M5,
            "M10": self._mt5.TIMEFRAME_M10,
            "M15": self._mt5.TIMEFRAME_M15,
            "M30": self._mt5.TIMEFRAME_M30,
            "H1": self._mt5.TIMEFRAME_H1,
            "H2": self._mt5.TIMEFRAME_H2,
            "H4": self._mt5.TIMEFRAME_H4,
            "D1": self._mt5.TIMEFRAME_D1,
        }
        if tf not in mapping:
            raise ValueError(f"Timeframe inconnu: {timeframe}")
        return mapping[tf]

    def bars_per_day(self, timeframe: str) -> int:
        """
        Approximation du nombre de bougies par jour selon la timeframe.
        Utile pour convertir history_days -> n_bars.

        Ex:
        - M5 => 1440/5 = 288
        - H1 => 24
        - D1 => 1
        """
        tf = timeframe.upper().strip()
        minutes_map = {
            "M1": 1,
            "M2": 2,
            "M3": 3,
            "M4": 4,
            "M5": 5,
            "M10": 10,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H2": 120,
            "H4": 240,
            "D1": 1440,
        }
        if tf not in minutes_map:
            raise ValueError(f"Timeframe inconnu: {timeframe}")

        minutes = minutes_map[tf]
        return int(1440 / minutes)

    # -----------------------------
    # Fetch methods (bars)
    # -----------------------------
    def fetch_last_n_bars(
            self,
            symbol: str,
            timeframe: str,
            n_bars: int,
            chunk_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Version chuncked: récupère les n_bars dernières bougies en plusieurs appels
        (évite les erreurs MT5 quand count est trop grand).

        start_pos: 0 = bougie la plus récente, puis on augmente pour aller vers le passé.
        """
        self._require_ready()
        self.ensure_symbol(symbol)

        tf = self.get_timeframe(timeframe)
        n_bars = int(n_bars)
        if n_bars <= 0:
            raise ValueError("n_bars doit être > 0")

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size doit être > 0")

        frames = []
        start_pos = 0
        remaining = n_bars

        # On tente avec une taille de chunk raisonnable, et si MT5 refuse,
        # on réessaie plus petit (fallback).
        fallback_chunks = [chunk_size, 2000, 1000, 500, 200, 100]

        while remaining > 0:
            wanted = min(remaining, chunk_size)

            rates = None
            used = None
            for cs in fallback_chunks:
                used = min(wanted, cs)
                rates = self._mt5.copy_rates_from_pos(symbol, tf, start_pos, used)
                if rates is not None:
                    break

            if rates is None:
                raise MT5ConnectionError(
                    f"copy_rates_from_pos() a renvoyé None après fallback chunks {fallback_chunks}. "
                    f"last_error={self._mt5.last_error()} "
                    f"(symbol={symbol}, timeframe={timeframe}, start_pos={start_pos}, wanted={wanted})"
                )

            if len(rates) == 0:
                # Plus de données disponibles côté terminal
                break

            df_chunk = pd.DataFrame(rates)
            frames.append(df_chunk)

            got = len(rates)
            start_pos += got
            remaining -= got

            # Si MT5 renvoie moins que demandé, on est probablement au bout de l'historique dispo
            if got < used:
                break

        if not frames:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            )

        df = pd.concat(frames, ignore_index=True)

        # time = seconds epoch -> datetime UTC
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        df = df[cols].drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        return df

    def fetch_bars_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        """
        Méthode optionnelle (parfois instable selon les environnements MT5) :
        récupère les bougies sur un intervalle de dates via copy_rates_range.

        On laisse cette méthode car elle peut être utile, mais pour le projet,
        on utilisera surtout fetch_last_n_bars().
        """
        self._require_ready()
        self.ensure_symbol(symbol)

        # MT5 aime souvent les datetime "naïfs" (sans tzinfo)
        if date_from.tzinfo is not None:
            date_from = date_from.replace(tzinfo=None)
        if date_to.tzinfo is not None:
            date_to = date_to.replace(tzinfo=None)

        tf = self.get_timeframe(timeframe)
        rates = self._mt5.copy_rates_range(symbol, tf, date_from, date_to)

        if rates is None:
            raise MT5ConnectionError(
                f"copy_rates_range() a renvoyé None: {self._mt5.last_error()}"
            )

        if len(rates) == 0:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        return df[cols].sort_values("time").reset_index(drop=True)
