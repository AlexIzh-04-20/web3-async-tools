"""
Async Portfolio Parser — Configuration & Initialization Module
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR: Path = Path(os.environ.get("HOME", "~")).expanduser() / "Brain_Base"
OUTPUT_DIR: Path = BASE_DIR / "portfolio_output"
CACHE_DIR: Path = BASE_DIR / "portfolio_cache"
LOG_DIR: Path = BASE_DIR / "portfolio_logs"

for _d in (OUTPUT_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger.remove()

LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
)

logger.add(
    LOG_DIR / "parser_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="gz",
    encoding="utf-8",
)

# ─────────────────────────────────────────────
# Parser defaults
# ─────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS: int = 5
REQUEST_TIMEOUT: int = 30
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF: float = 1.5
USER_AGENT: str = "Mozilla/5.0 (compatible; PortfolioBot/1.0)"


def get_base_dir() -> Path:
    return BASE_DIR


def get_output_dir() -> Path:
    return OUTPUT_DIR


# ─────────────────────────────────────────────
# HTTP Transport Module
# ─────────────────────────────────────────────

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, TypeVar

import aiohttp
from aiohttp import (
    ClientError,
    ClientResponse,
    ClientSession,
    ServerDisconnectedError,
    TCPConnector,
)

T = TypeVar("T")

# ── Exceptions ────────────────────────────────


class TransportError(Exception):
    """Base transport error."""


class RateLimitError(TransportError):
    """Raised when rate limit is exceeded."""


class MaxRetriesExceeded(TransportError):
    """Raised after all retry attempts are exhausted."""


# ── Configuration ─────────────────────────────


@dataclass(frozen=True)
class TransportConfig:
    """Immutable transport configuration."""

    max_concurrent: int = MAX_CONCURRENT_REQUESTS
    timeout: int = REQUEST_TIMEOUT
    retry_attempts: int = RETRY_ATTEMPTS
    retry_backoff: float = RETRY_BACKOFF
    user_agent: str = USER_AGENT
    default_headers: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        base = {"User-Agent": self.user_agent}
        base.update(self.default_headers)
        return base


# ── Retry decorator ───────────────────────────


def with_retry(
    attempts: int = RETRY_ATTEMPTS,
    backoff: float = RETRY_BACKOFF,
    exceptions: tuple[type[Exception], ...] = (
        ServerDisconnectedError,
        ClientError,
        asyncio.TimeoutError,
    ),
) -> Callable:
    """Decorator: retries an async function with exponential backoff."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    delay = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Retry {attempt}/{attempts} after {delay:.1f}s — {exc}",
                        attempt=attempt,
                        attempts=attempts,
                        delay=delay,
                        exc=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
            raise MaxRetriesExceeded(
                f"All {attempts} retries exhausted"
            ) from last_exc

        return wrapper

    return decorator


# ── HTTP Transport ────────────────────────────


class HttpTransport:
    """Async HTTP transport with rate limiting and retry support."""

    def __init__(self, config: Optional[TransportConfig] = None) -> None:
        self._config = config or TransportConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._session: Optional[ClientSession] = None
        self._request_count: int = 0

    # ── Lifecycle ──────────────────────────────

    async def __aenter__(self) -> "HttpTransport":
        await self._ensure_session()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            connector = TCPConnector(
                limit=self._config.max_concurrent,
                force_close=False,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=self._config.timeout)
            self._session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._config.headers(),
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            # Allow connector cleanup
            await asyncio.sleep(0.25)

    # ── Core request method ────────────────────

    @with_retry()
    async def _raw_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> ClientResponse:
        """Send request through the semaphore gate."""
        await self._ensure_session()
        async with self._semaphore:
            self._request_count += 1
            logger.debug(
                "HTTP {method} {url} (#{count})",
                method=method.upper(),
                url=url,
                count=self._request_count,
            )
            response = await self._session.request(method, url, **kwargs)
            response.raise_for_status()
            return response

    # ── Public API ─────────────────────────────

    async def get(
        self,
        url: str,
        *,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        """GET request returning response text."""
        async with await self._raw_request(
            "GET", url, params=params, headers=headers
        ) as resp:
            return await resp.text()

    async def get_json(
        self,
        url: str,
        *,
        params: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """GET request returning parsed JSON."""
        async with await self._raw_request(
            "GET", url, params=params, headers=headers
        ) as resp:
            return await resp.json(content_type=None)

    async def post(
        self,
        url: str,
        *,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        """POST request returning response text."""
        async with await self._raw_request(
            "POST", url, data=data, json=json, headers=headers
        ) as resp:
            return await resp.text()

    async def post_json(
        self,
        url: str,
        *,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """POST request returning parsed JSON."""
        async with await self._raw_request(
            "POST", url, data=data, json=json, headers=headers
        ) as resp:
            return await resp.json(content_type=None)

    # ── Utilities ──────────────────────────────

    @property
    def request_count(self) -> int:
        return self._request_count

    async def check_url(self, url: str) -> bool:
        """Return True if URL is reachable (HEAD with 2xx/3xx)."""
        try:
            await self._ensure_session()
            async with self._semaphore:
                async with self._session.head(
                    url,
                    allow_redirects=True,
                ) as resp:
                    return resp.status < 400
        except (ServerDisconnectedError, ClientError, asyncio.TimeoutError):
            return False


# ─────────────────────────────────────────────
# Data Normalization & Validation Module
# ─────────────────────────────────────────────

import enum
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    get_type_hints,
)
from dataclasses import dataclass, field, fields, asdict
from functools import wraps

# ── Exceptions ────────────────────────────────


class ValidationError(Exception):
    """Raised when data fails validation."""

    def __init__(self, field_name: str, message: str, value: Any = None) -> None:
        self.field_name = field_name
        self.value = value
        super().__init__(f"Validation failed for '{field_name}': {message}")


class NormalizationError(Exception):
    """Raised when normalization fails."""


# ── Type Guards ───────────────────────────────


def safe_get(
    data: dict[str, Any],
    key: str,
    default: Any = None,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    """Safely extract value from dict with optional type check."""
    value = data.get(key, default)
    if value is None:
        return default
    if expected_type is not None and not isinstance(value, expected_type):
        raise ValidationError(
            field_name=key,
            message=f"Expected {expected_type}, got {type(value).__name__}",
            value=value,
        )
    return value


def safe_list(data: dict[str, Any], key: str) -> list[Any]:
    """Extract list from dict; return empty list if missing or not a list."""
    value = data.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def safe_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Traverse nested dicts safely. Returns default if any key is missing."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def coerce_decimal(value: Any, field_name: str = "unknown") -> Decimal:
    """Convert value to Decimal; raises ValidationError on failure."""
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(
            field_name=field_name,
            message=f"Cannot convert '{value}' to Decimal",
            value=value,
        ) from exc


def coerce_timestamp(value: Any, field_name: str = "timestamp") -> datetime:
    """Parse timestamp from int (unix) or ISO string into timezone-aware datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(
                field_name=field_name,
                message=f"Cannot parse timestamp: '{value}'",
                value=value,
            ) from exc
    raise ValidationError(
        field_name=field_name,
        message=f"Expected int or str, got {type(value).__name__}",
        value=value,
    )


# ── Normalizer Protocol ───────────────────────

T_Model = TypeVar("T_Model", bound="BaseModel")


class BaseModel:
    """Minimal dataclass-like base with runtime validation and safe API parsing."""

    # Subclasses may define validators as class-level callables
    _validators: ClassVar[dict[str, list[callable]]] = {}

    @classmethod
    def from_dict(cls: type[T_Model], raw: dict[str, Any]) -> T_Model:
        """Construct instance from an API response dict.

        Missing keys → field defaults. Extra keys → ignored.
        Coerces values to declared field types (Decimal, datetime, enum).
        """
        if not isinstance(raw, dict):
            raise ValidationError(
                field_name=cls.__name__,
                message=f"Root payload must be dict, got {type(raw).__name__}",
                value=raw
            )
        hints = get_type_hints(cls)
        init_kwargs: dict[str, Any] = {}

        for field_name, field_type in hints.items():
            if field_name.startswith("_"):
                continue
            value = raw.get(field_name)

            # Extract nested model instances
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                if isinstance(value, dict):
                    init_kwargs[field_name] = field_type.from_dict(value)
                else:
                    init_kwargs[field_name] = None
            elif isinstance(value, list) and getattr(field_type, "__origin__", None) is list:
                # Handle list[Model] types
                args = getattr(field_type, "__args__", ())
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    inner = args[0]
                    init_kwargs[field_name] = [
                        inner.from_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    init_kwargs[field_name] = value if value is not None else []
            else:
                # Type coercion for common types
                if value is not None:
                    if field_type is Decimal:
                        value = coerce_decimal(value, field_name)
                    elif field_type is datetime:
                        value = coerce_timestamp(value, field_name)
                    elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
                        try:
                            value = field_type(str(value).lower())
                        except (ValueError, KeyError):
                            pass  # keep raw value; validator will catch
                    elif field_type is str and not isinstance(value, str):
                        value = str(value)
                init_kwargs[field_name] = value

        instance = cls(**init_kwargs)
        return instance

    @classmethod
    def from_list(cls: type[T_Model], items: list[dict[str, Any]]) -> list[T_Model]:
        """Parse a list of API dicts into model instances, skipping invalid entries."""
        result: list[T_Model] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict item at index {idx}: {type}",
                    idx=idx,
                    type=type(item).__name__,
                )
                continue
            try:
                result.append(cls.from_dict(item))
            except ValidationError as exc:
                logger.warning("Skipping invalid item at index {idx}: {exc}", idx=idx, exc=exc)
        return result

    def _run_validators(self) -> None:
        """Invoke all registered validators for this instance."""
        validators = getattr(self, "_validators", {})
        for field_name, checks in validators.items():
            value = getattr(self, field_name, None)
            for check in checks:
                check(field_name, value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict (handles nested BaseModel and Decimal)."""
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseModel):
                result[f.name] = value.to_dict()
            elif isinstance(value, list):
                result[f.name] = [
                    item.to_dict() if isinstance(item, BaseModel) else item
                    for item in value
                ]
            elif isinstance(value, Decimal):
                result[f.name] = str(value)
            elif isinstance(value, datetime):
                result[f.name] = value.isoformat()
            else:
                result[f.name] = value
        return result

    def validate(self) -> list[ValidationError]:
        """Run all validators; return list of errors (empty = valid)."""
        errors: list[ValidationError] = []
        validators = getattr(self, "_validators", {})
        for field_name, checks in validators.items():
            value = getattr(self, field_name, None)
            for check in checks:
                try:
                    check(field_name, value)
                except ValidationError as exc:
                    errors.append(exc)
        return errors


# ── Validation Helpers ────────────────────────


def require_non_empty(field_name: str, value: Any) -> None:
    """Raise if value is None, empty string, or empty list."""
    if value is None:
        raise ValidationError(field_name, "Required field is missing")
    if isinstance(value, str) and not value.strip():
        raise ValidationError(field_name, "String must not be empty")
    if isinstance(value, (list, dict)) and len(value) == 0:
        raise ValidationError(field_name, "Collection must not be empty")


def require_positive(field_name: str, value: Any) -> None:
    """Raise if value is not a positive number."""
    if value is None:
        return  # Allow None — use require_non_empty to enforce
    try:
        num = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValidationError(field_name, f"Expected numeric, got '{value}'")
    if num < 0:
        raise ValidationError(field_name, f"Expected positive, got {num}")


def require_in_range(
    min_val: float | None = None,
    max_val: float | None = None,
) -> callable:
    """Factory: returns a validator that checks value is within [min, max]."""

    def validator(field_name: str, value: Any) -> None:
        if value is None:
            return
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValidationError(field_name, f"Expected numeric, got '{value}'")
        if min_val is not None and num < min_val:
            raise ValidationError(field_name, f"Value {num} below minimum {min_val}")
        if max_val is not None and num > max_val:
            raise ValidationError(field_name, f"Value {num} above maximum {max_val}")

    return validator


def require_pattern(pattern: str) -> callable:
    """Factory: returns a validator that checks value matches a regex."""

    def validator(field_name: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ValidationError(field_name, f"Expected str, got {type(value).__name__}")
        if not re.match(pattern, value):
            raise ValidationError(field_name, f"Value '{value}' does not match pattern '{pattern}'")

    return validator


# ── Domain Models ─────────────────────────────


class Side(enum.Enum):
    """Trade side."""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(enum.Enum):
    """Position lifecycle status."""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position(BaseModel):
    """Normalized trading position."""

    symbol: str = ""
    quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_price: Decimal = field(default_factory=lambda: Decimal("0"))
    current_price: Decimal = field(default_factory=lambda: Decimal("0"))
    side: Side = Side.BUY
    status: PositionStatus = PositionStatus.OPEN
    pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    pnl_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _validators: ClassVar[dict] = {
        "symbol": [require_non_empty],
        "quantity": [require_positive],
        "entry_price": [require_positive],
    }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Position":
        """Handle common API field name variants."""
        if not isinstance(raw, dict):
            raise ValidationError(
                field_name="raw_data",
                message=f"Expected dict, got {type(raw).__name__}",
                value=raw
            )
        normalized: dict[str, Any] = {}

        # Symbol: support multiple keys
        normalized["symbol"] = (
            raw.get("symbol")
            or raw.get("ticker")
            or raw.get("asset")
            or ""
        )

        # Quantity
        qty = raw.get("qty") or raw.get("quantity") or raw.get("size") or "0"
        normalized["quantity"] = qty

        # Entry price
        normalized["entry_price"] = (
            raw.get("entry_price")
            or raw.get("avg_price")
            or raw.get("average_price")
            or "0"
        )

        # Current price
        normalized["current_price"] = (
            raw.get("current_price")
            or raw.get("last_price")
            or raw.get("mark_price")
            or "0"
        )

        # Side
        raw_side = str(raw.get("side", "buy")).lower()
        normalized["side"] = Side.BUY if raw_side == "buy" else Side.SELL

        # Status
        raw_status = str(raw.get("status", "open")).lower()
        try:
            normalized["status"] = PositionStatus(raw_status)
        except ValueError:
            normalized["status"] = PositionStatus.OPEN

        # PnL
        normalized["pnl"] = raw.get("pnl") or raw.get("unrealized_pnl") or "0"
        normalized["pnl_pct"] = raw.get("pnl_pct") or raw.get("pnl_percent") or "0"

        # Timestamp
        normalized["timestamp"] = (
            raw.get("timestamp")
            or raw.get("created_at")
            or raw.get("opened_at")
        )

        return super().from_dict(normalized)


@dataclass
class Balance(BaseModel):
    """Normalized account balance."""

    asset: str = "USD"
    total: Decimal = field(default_factory=lambda: Decimal("0"))
    available: Decimal = field(default_factory=lambda: Decimal("0"))
    locked: Decimal = field(default_factory=lambda: Decimal("0"))

    _validators: ClassVar[dict] = {
        "asset": [require_non_empty],
        "total": [require_positive],
    }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Balance":
        if not isinstance(raw, dict):
            raise ValidationError(
                field_name="raw_data",
                message=f"Expected dict, got {type(raw).__name__}",
                value=raw
            )
        normalized: dict[str, Any] = {}
        normalized["asset"] = (
            raw.get("asset")
            or raw.get("currency")
            or raw.get("coin")
            or "USD"
        )
        normalized["total"] = (
            raw.get("total")
            or raw.get("balance")
            or raw.get("equity")
            or "0"
        )
        normalized["available"] = (
            raw.get("available")
            or raw.get("free")
            or raw.get("available_balance")
            or "0"
        )
        normalized["locked"] = (
            raw.get("locked")
            or raw.get("frozen")
            or raw.get("reserved")
            or "0"
        )
        return super().from_dict(normalized)


@dataclass
class Transaction(BaseModel):
    """Normalized transaction / trade record."""

    tx_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    price: Decimal = field(default_factory=lambda: Decimal("0"))
    fee: Decimal = field(default_factory=lambda: Decimal("0"))
    total: Decimal = field(default_factory=lambda: Decimal("0"))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _validators: ClassVar[dict] = {
        "symbol": [require_non_empty],
        "quantity": [require_positive],
        "price": [require_positive],
    }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Transaction":
        if not isinstance(raw, dict):
            raise ValidationError(
                field_name="raw_data",
                message=f"Expected dict, got {type(raw).__name__}",
                value=raw
            )
        normalized: dict[str, Any] = {}
        normalized["tx_id"] = (
            raw.get("tx_id")
            or raw.get("id")
            or raw.get("order_id")
            or raw.get("trade_id")
            or ""
        )
        normalized["symbol"] = (
            raw.get("symbol")
            or raw.get("ticker")
            or raw.get("pair")
            or ""
        )
        raw_side = str(raw.get("side", "buy")).lower()
        normalized["side"] = Side.BUY if raw_side == "buy" else Side.SELL

        normalized["quantity"] = (
            raw.get("qty")
            or raw.get("quantity")
            or raw.get("amount")
            or raw.get("size")
            or "0"
        )
        normalized["price"] = (
            raw.get("price")
            or raw.get("avg_price")
            or raw.get("execution_price")
            or "0"
        )
        normalized["fee"] = raw.get("fee") or raw.get("commission") or "0"
        normalized["total"] = raw.get("total") or raw.get("cost") or "0"
        normalized["timestamp"] = (
            raw.get("timestamp")
            or raw.get("executed_at")
            or raw.get("created_at")
        )
        return super().from_dict(normalized)


@dataclass
class PortfolioSnapshot(BaseModel):
    """Complete portfolio snapshot aggregating positions, balances, transactions."""

    positions: list[Position] = field(default_factory=list)
    balances: list[Balance] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    source: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_api_response(
        cls,
        raw: dict[str, Any],
        source: str = "unknown",
    ) -> "PortfolioSnapshot":
        """Parse a generic API response into a PortfolioSnapshot.

        Handles common response envelope patterns:
        - {"data": {"positions": [...], "balances": [...]}}
        - {"positions": [...], "balances": [...]}
        - Flat list → treated as positions
        """
        if not isinstance(raw, dict):
            logger.warning("Expected dict response, got {type}", type=type(raw).__name__)
            return cls(source=source)

        # Unwrap common envelope keys
        data = raw.get("data") or raw.get("result") or raw

        if isinstance(data, list):
            # Flat list of positions
            positions_raw = data
            balances_raw: list[dict[str, Any]] = []
            transactions_raw: list[dict[str, Any]] = []
        elif isinstance(data, dict):
            positions_raw = safe_list(data, "positions") or safe_list(data, "orders")
            balances_raw = safe_list(data, "balances") or safe_list(data, "accounts")
            transactions_raw = (
                safe_list(data, "transactions")
                or safe_list(data, "trades")
                or safe_list(data, "fills")
            )
        else:
            positions_raw = []
            balances_raw = []
            transactions_raw = []

        return cls(
            positions=Position.from_list(positions_raw),
            balances=Balance.from_list(balances_raw),
            transactions=Transaction.from_list(transactions_raw),
            source=source,
            fetched_at=coerce_timestamp(raw.get("fetched_at")),
        )


# ── Validation Decorator ──────────────────────


def validate_output(cls: type[T_Model]) -> type[T_Model]:
    """Class decorator: auto-validates instance after __post_init__."""
    original_init = cls.__init__

    @wraps(original_init)
    def validated_init(self: T_Model, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        errors = self.validate() if hasattr(self, "validate") else []
        if errors:
            error_summary = "; ".join(str(e) for e in errors)
            logger.warning(
                "Model {cls} has validation warnings: {errors}",
                cls=cls.__name__,
                errors=error_summary,
            )

    cls.__init__ = validated_init  # type: ignore[assignment]
    return cls


# ── Batch Normalization Pipeline ──────────────


async def normalize_portfolio_data(
    raw_responses: dict[str, dict[str, Any]],
) -> PortfolioSnapshot:
    """Merge multiple API responses into one PortfolioSnapshot.

    Args:
        raw_responses: Mapping of source_name → raw API response dict.

    Returns:
        Merged PortfolioSnapshot with all data normalized.
    """
    all_positions: list[Position] = []
    all_balances: list[Balance] = []
    all_transactions: list[Transaction] = []

    for source, response in raw_responses.items():
        try:
            snapshot = PortfolioSnapshot.from_api_response(response, source=source)
            all_positions.extend(snapshot.positions)
            all_balances.extend(snapshot.balances)
            all_transactions.extend(snapshot.transactions)
            logger.info(
                "Normalized {source}: {p} positions, {b} balances, {t} transactions",
                source=source,
                p=len(snapshot.positions),
                b=len(snapshot.balances),
                t=len(snapshot.transactions),
            )
        except Exception as exc:
            logger.error(
                "Failed to normalize data from {source}: {exc}",
                source=source,
                exc=exc,
            )

    return PortfolioSnapshot(
        positions=all_positions,
        balances=all_balances,
        transactions=all_transactions,
        source="merged",
        fetched_at=datetime.now(timezone.utc),
    )


def validate_snapshot(snapshot: PortfolioSnapshot) -> dict[str, Any]:
    """Run full validation on a PortfolioSnapshot; return validation report."""
    report: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "counts": {
            "positions": len(snapshot.positions),
            "balances": len(snapshot.balances),
            "transactions": len(snapshot.transactions),
        },
    }

    # Validate each model instance
    for pos in snapshot.positions:
        errors = pos.validate()
        for err in errors:
            report["errors"].append({"model": "Position", "error": str(err)})
            report["valid"] = False

    for bal in snapshot.balances:
        errors = bal.validate()
        for err in errors:
            report["errors"].append({"model": "Balance", "error": str(err)})
            report["valid"] = False

    for tx in snapshot.transactions:
        errors = tx.validate()
        for err in errors:
            report["errors"].append({"model": "Transaction", "error": str(err)})
            report["valid"] = False

    # Cross-model consistency checks
    if snapshot.balances:
        total_equity = sum(b.total for b in snapshot.balances)
        total_position_value = sum(
            p.quantity * p.current_price for p in snapshot.positions
        )
        report["total_equity"] = str(total_equity)
        report["total_position_value"] = str(total_position_value)

        if total_equity > 0 and total_position_value > 0:
            exposure_pct = (total_position_value / total_equity) * 100
            report["exposure_pct"] = str(exposure_pct)
            if exposure_pct > 100:
                report["warnings"].append(
                    f"Position value exceeds equity by {exposure_pct - 100:.1f}%"
                )

    if not snapshot.positions and not snapshot.balances:
        report["warnings"].append("Snapshot is empty — no positions or balances found")

    return report


# ─────────────────────────────────────────────
# Storage Module (Async File Export)
# ─────────────────────────────────────────────

import csv
import io
from enum import Enum
from dataclasses import fields as dataclass_fields

import aiofiles
import aiofiles.os


class ExportFormat(Enum):
    """Supported export file formats."""

    JSON = "json"
    CSV = "csv"


class StorageError(Exception):
    """Raised when a storage operation fails."""


class PortfolioStorage:
    """Async storage backend for portfolio snapshots.

    Supports JSON and CSV export with automatic directory creation,
    atomic writes, and configurable output paths.
    """

    def __init__(
        self,
        output_dir: Path | str = OUTPUT_DIR,
        *,
        ensure_dir: bool = True,
    ) -> None:
        self._output_dir = Path(output_dir)
        if ensure_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # ── Directory management ───────────────────

    async def ensure_directory(self, path: Path | str) -> Path:
        """Create directory tree if it doesn't exist; return resolved Path."""
        target = Path(path)
        try:
            await aiofiles.os.makedirs(target, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"Failed to create directory '{target}': {exc}"
            ) from exc
        return target

    # ── JSON export ────────────────────────────

    async def save_json(
        self,
        snapshot: PortfolioSnapshot,
        filename: str | None = None,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        """Serialize a PortfolioSnapshot to JSON and write asynchronously.

        Args:
            snapshot: The portfolio snapshot to persist.
            filename: Output filename (without extension). Auto-generated if None.
            indent: JSON indentation level.
            ensure_ascii: Escape non-ASCII characters when True.

        Returns:
            Absolute path to the written file.

        Raises:
            StorageError: On write failure or serialization error.
        """
        if filename is None:
            ts = snapshot.fetched_at.strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_{ts}"

        path = self._output_dir / f"{filename}.json"

        try:
            data = snapshot.to_dict()
        except Exception as exc:
            raise StorageError(
                f"Serialization to dict failed: {exc}"
            ) from exc

        try:
            payload = json.dumps(
                data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=str,
            )
            async with aiofiles.open(path, "w", encoding="utf-8") as fh:
                await fh.write(payload)
            logger.info("JSON saved → {path}", path=path)
            return path
        except OSError as exc:
            raise StorageError(
                f"Failed to write JSON to '{path}': {exc}"
            ) from exc

    async def load_json(self, filename: str) -> dict[str, Any]:
        """Read a JSON file and return parsed dict.

        Raises:
            StorageError: On read or parse failure.
        """
        path = self._output_dir / filename
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as fh:
                content = await fh.read()
            return json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Failed to read JSON from '{path}': {exc}"
            ) from exc

    # ── CSV export ─────────────────────────────

    async def save_csv(
        self,
        snapshot: PortfolioSnapshot,
        filename: str | None = None,
        *,
        include_balances: bool = True,
        include_transactions: bool = True,
    ) -> Path:
        """Export positions (and optionally balances/transactions) to CSV.

        Writes a multi-section CSV with headers separating each entity type.

        Args:
            snapshot: The portfolio snapshot to persist.
            filename: Output filename (without extension). Auto-generated if None.
            include_balances: Append balances section when True.
            include_transactions: Append transactions section when True.

        Returns:
            Absolute path to the written file.

        Raises:
            StorageError: On write failure.
        """
        if filename is None:
            ts = snapshot.fetched_at.strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_{ts}"

        path = self._output_dir / f"{filename}.csv"

        try:
            buf = io.StringIO()
            writer = csv.writer(buf)

            # Section: Positions
            if snapshot.positions:
                writer.writerow(["=== POSITIONS ==="])
                pos_fields = [
                    f.name for f in dataclass_fields(snapshot.positions[0])
                    if not f.name.startswith("_")
                ]
                writer.writerow(pos_fields)
                for pos in snapshot.positions:
                    row = self._model_row(pos, pos_fields)
                    writer.writerow(row)
                writer.writerow([])

            # Section: Balances
            if include_balances and snapshot.balances:
                writer.writerow(["=== BALANCES ==="])
                bal_fields = [
                    f.name for f in dataclass_fields(snapshot.balances[0])
                    if not f.name.startswith("_")
                ]
                writer.writerow(bal_fields)
                for bal in snapshot.balances:
                    row = self._model_row(bal, bal_fields)
                    writer.writerow(row)
                writer.writerow([])

            # Section: Transactions
            if include_transactions and snapshot.transactions:
                writer.writerow(["=== TRANSACTIONS ==="])
                tx_fields = [
                    f.name for f in dataclass_fields(snapshot.transactions[0])
                    if not f.name.startswith("_")
                ]
                writer.writerow(tx_fields)
                for tx in snapshot.transactions:
                    row = self._model_row(tx, tx_fields)
                    writer.writerow(row)

            async with aiofiles.open(path, "w", encoding="utf-8", newline="") as fh:
                await fh.write(buf.getvalue())
            logger.info("CSV saved → {path}", path=path)
            return path
        except OSError as exc:
            raise StorageError(
                f"Failed to write CSV to '{path}': {exc}"
            ) from exc

    @staticmethod
    def _model_row(model: BaseModel, field_names: list[str]) -> list[str]:
        """Convert a model instance to a flat list of strings for CSV."""
        row: list[str] = []
        for name in field_names:
            value = getattr(model, name, "")
            if isinstance(value, Decimal):
                row.append(str(value))
            elif isinstance(value, datetime):
                row.append(value.isoformat())
            elif isinstance(value, enum.Enum):
                row.append(value.value)
            elif isinstance(value, BaseModel):
                row.append(json.dumps(value.to_dict(), default=str))
            elif isinstance(value, list):
                row.append(json.dumps(value, default=str))
            else:
                row.append(str(value) if value is not None else "")
        return row

    # ── Unified save interface ─────────────────

    async def save(
        self,
        snapshot: PortfolioSnapshot,
        fmt: ExportFormat = ExportFormat.JSON,
        filename: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save snapshot in the specified format.

        Delegates to save_json or save_csv based on *fmt*.
        Extra kwargs are forwarded to the format-specific method.
        """
        if fmt is ExportFormat.JSON:
            return await self.save_json(snapshot, filename, **kwargs)
        elif fmt is ExportFormat.CSV:
            return await self.save_csv(snapshot, filename, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

    async def save_all(
        self,
        snapshot: PortfolioSnapshot,
        filename: str | None = None,
    ) -> dict[str, Path]:
        """Export snapshot to both JSON and CSV simultaneously.

        Returns:
            Mapping of format name → written file path.
        """
        results: dict[str, Path] = {}
        json_path, csv_path = await asyncio.gather(
            self.save_json(snapshot, filename),
            self.save_csv(snapshot, filename),
        )
        results[ExportFormat.JSON.value] = json_path
        results[ExportFormat.CSV.value] = csv_path
        logger.info(
            "Dual export complete: {json} | {csv}",
            json=json_path,
            csv=csv_path,
        )
        return results

    # ── File utilities ─────────────────────────

    async def file_exists(self, filename: str) -> bool:
        """Check whether a file exists in the output directory."""
        path = self._output_dir / filename
        try:
            await aiofiles.os.stat(path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    async def list_files(self, extension: str | None = None) -> list[Path]:
        """List files in the output directory, optionally filtered by extension."""
        try:
            entries = await aiofiles.os.listdir(self._output_dir)
        except OSError as exc:
            raise StorageError(
                f"Failed to list directory '{self._output_dir}': {exc}"
            ) from exc

        paths = [self._output_dir / e for e in entries if (self._output_dir / e).is_file()]
        if extension is not None:
            ext = extension if extension.startswith(".") else f".{extension}"
            paths = [p for p in paths if p.suffix == ext]
        return sorted(paths)

    async def remove_file(self, filename: str) -> bool:
        """Delete a file from the output directory.

        Returns True if the file was removed, False if it didn't exist.
        """
        path = self._output_dir / filename
        try:
            await aiofiles.os.remove(path)
            logger.debug("Removed {path}", path=path)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StorageError(
                f"Failed to remove '{path}': {exc}"
            ) from exc
