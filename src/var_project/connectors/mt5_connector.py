from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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

        initialize_kwargs: dict[str, Any] = {}
        if self.config.path:
            initialize_kwargs["path"] = str(self.config.path)
        if self.config.timeout_ms is not None:
            initialize_kwargs["timeout"] = int(self.config.timeout_ms)
        if self.config.portable:
            initialize_kwargs["portable"] = True

        ok = mt5.initialize(**initialize_kwargs) if initialize_kwargs else mt5.initialize()
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

    def _last_error(self) -> Any:
        if not self._mt5:
            return None
        try:
            return self._mt5.last_error()
        except Exception:
            return None

    def _should_retry_after_error(self, error: Any) -> bool:
        if not isinstance(error, tuple) or not error:
            return False
        try:
            return int(error[0]) < 0
        except (TypeError, ValueError):
            return False

    def _call_with_reconnect(self, func, *args, **kwargs):
        self._require_ready()
        result = func(*args, **kwargs)
        if result is not None:
            return result
        if not self._should_retry_after_error(self._last_error()):
            return None
        self.shutdown()
        self.init()
        return func(*args, **kwargs)

    def _coerce_namedtuple(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "_asdict"):
            return {str(key): self._coerce_namedtuple(item) for key, item in value._asdict().items()}
        if isinstance(value, dict):
            return {str(key): self._coerce_namedtuple(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._coerce_namedtuple(item) for item in value]
        return value

    def terminal_info(self) -> dict[str, Any]:
        info = self._call_with_reconnect(self._mt5.terminal_info)
        if info is None:
            raise MT5ConnectionError(f"terminal_info() a echoue: {self._mt5.last_error()}")
        return dict(self._coerce_namedtuple(info) or {})

    def account_info(self) -> dict[str, Any]:
        info = self._call_with_reconnect(self._mt5.account_info)
        if info is None:
            raise MT5ConnectionError(f"account_info() a echoue: {self._mt5.last_error()}")
        return dict(self._coerce_namedtuple(info) or {})

    def symbol_info(self, symbol: str) -> dict[str, Any]:
        self.ensure_symbol(symbol)
        info = self._call_with_reconnect(self._mt5.symbol_info, symbol)
        if info is None:
            raise MT5ConnectionError(f"symbol_info() a echoue pour {symbol}: {self._mt5.last_error()}")
        return dict(self._coerce_namedtuple(info) or {})

    def symbol_info_tick(self, symbol: str) -> dict[str, Any]:
        self.ensure_symbol(symbol)
        tick = self._call_with_reconnect(self._mt5.symbol_info_tick, symbol)
        if tick is None:
            raise MT5ConnectionError(f"symbol_info_tick() a echoue pour {symbol}: {self._mt5.last_error()}")
        payload = dict(self._coerce_namedtuple(tick) or {})
        time_msc = payload.get("time_msc")
        epoch = None
        if time_msc is not None:
            epoch = float(time_msc) / 1000.0
        elif payload.get("time") is not None:
            epoch = float(payload["time"])
        payload["time_utc"] = None if epoch is None else datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        return payload

    def positions_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            self.ensure_symbol(symbol)
            rows = self._call_with_reconnect(self._mt5.positions_get, symbol=symbol)
        else:
            rows = self._call_with_reconnect(self._mt5.positions_get)
        if rows is None:
            raise MT5ConnectionError(f"positions_get() a echoue: {self._mt5.last_error()}")
        return [dict(self._coerce_namedtuple(row) or {}) for row in rows]

    def orders_get(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            self.ensure_symbol(symbol)
            rows = self._call_with_reconnect(self._mt5.orders_get, symbol=symbol)
        else:
            rows = self._call_with_reconnect(self._mt5.orders_get)
        if rows is None:
            raise MT5ConnectionError(f"orders_get() a echoue: {self._mt5.last_error()}")
        return [dict(self._coerce_namedtuple(row) or {}) for row in rows]

    def history_orders_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]]:
        if ticket is not None:
            rows = self._call_with_reconnect(self._mt5.history_orders_get, ticket=int(ticket))
        elif position is not None:
            rows = self._call_with_reconnect(self._mt5.history_orders_get, position=int(position))
        else:
            start = self._history_datetime(date_from)
            end = self._history_datetime(date_to)
            rows = self._call_with_reconnect(self._mt5.history_orders_get, start, end)
        if rows is None:
            raise MT5ConnectionError(f"history_orders_get() a echoue: {self._mt5.last_error()}")
        payload = [dict(self._coerce_namedtuple(row) or {}) for row in rows]
        if symbol is None:
            return payload
        normalized = str(symbol).upper()
        return [item for item in payload if str(item.get("symbol") or "").upper() == normalized]

    def history_deals_get(
        self,
        date_from: datetime,
        date_to: datetime,
        *,
        symbol: str | None = None,
        ticket: int | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]]:
        if ticket is not None:
            rows = self._call_with_reconnect(self._mt5.history_deals_get, ticket=int(ticket))
        elif position is not None:
            rows = self._call_with_reconnect(self._mt5.history_deals_get, position=int(position))
        else:
            start = self._history_datetime(date_from)
            end = self._history_datetime(date_to)
            rows = self._call_with_reconnect(self._mt5.history_deals_get, start, end)
        if rows is None:
            raise MT5ConnectionError(f"history_deals_get() a echoue: {self._mt5.last_error()}")
        payload = [dict(self._coerce_namedtuple(row) or {}) for row in rows]
        if symbol is None:
            return payload
        normalized = str(symbol).upper()
        return [item for item in payload if str(item.get("symbol") or "").upper() == normalized]

    def order_check(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        payload = dict(request)
        result = self._mt5.order_check(payload)
        if result is None and self._should_retry_after_error(self._last_error()):
            self.shutdown()
            self.init()
            result = self._mt5.order_check(dict(request))
        if result is None:
            raise MT5ConnectionError(f"order_check() a echoue: {self._mt5.last_error()}")
        return dict(self._coerce_namedtuple(result) or {})

    def order_send(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_ready()
        payload = dict(request)
        result = self._mt5.order_send(payload)
        if result is None and self._should_retry_after_error(self._last_error()):
            self.shutdown()
            self.init()
            result = self._mt5.order_send(dict(request))
        if result is None:
            raise MT5ConnectionError(f"order_send() a echoue: {self._mt5.last_error()}")
        return dict(self._coerce_namedtuple(result) or {})

    # -----------------------------
    # Symbol utilities
    # -----------------------------
    def ensure_symbol(self, symbol: str) -> None:
        """
        Vérifie que le symbole existe et est visible dans MarketWatch.
        """
        self._require_ready()

        info = self._call_with_reconnect(self._mt5.symbol_info, symbol)
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

            # Si MT5 renvoie moins que demandé, on est probablement au bout de l'historique disponible
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

    def _history_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def fetch_tick(self, symbol: str) -> dict:
        """
        Retourne le tick courant (bid/ask) + time UTC.
        """
        self._require_ready()
        self.ensure_symbol(symbol)

        t = self._mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5ConnectionError(f"symbol_info_tick(None) pour {symbol}: {self._mt5.last_error()}")

        # time_msc est plus précis si dispo, sinon time (secondes)
        ts = getattr(t, "time_msc", None)
        if ts is not None:
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(t.time, tz=timezone.utc)

        return {
            "time": dt,
            "bid": float(t.bid),
            "ask": float(t.ask),
            "last": float(getattr(t, "last", 0.0) or 0.0),
        }

    def fetch_ticks_range(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        *,
        flags: int | None = None,
    ) -> pd.DataFrame:
        """
        Récupère l'historique de ticks MT5 sur un intervalle UTC.
        """
        self._require_ready()
        self.ensure_symbol(symbol)

        start = self._history_datetime(date_from)
        end = self._history_datetime(date_to)
        if flags is None:
            flags = getattr(self._mt5, "COPY_TICKS_ALL", 0)

        ticks = self._call_with_reconnect(self._mt5.copy_ticks_range, symbol, start, end, flags)
        if ticks is None:
            raise MT5ConnectionError(
                f"copy_ticks_range() a renvoyé None: {self._mt5.last_error()} (symbol={symbol}, start={start}, end={end})"
            )
        if len(ticks) == 0:
            return pd.DataFrame(columns=["time_utc", "bid", "ask", "last", "volume", "time_msc", "flags"])

        frame = pd.DataFrame(ticks)
        if "time_msc" in frame.columns:
            frame["time_utc"] = pd.to_datetime(frame["time_msc"], unit="ms", utc=True, errors="coerce")
        elif "time" in frame.columns:
            frame["time_utc"] = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
        else:
            frame["time_utc"] = pd.NaT
        columns = [column for column in ["time_utc", "bid", "ask", "last", "volume", "time_msc", "flags"] if column in frame.columns]
        frame = frame[columns].dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
        return frame

    def mid_price(self, symbol: str) -> float:
        tick = self.fetch_tick(symbol)
        bid, ask = tick["bid"], tick["ask"]
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        # fallback
        return bid if bid > 0 else ask
