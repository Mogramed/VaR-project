from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PortfolioSummary(BaseModel):
    id: int
    slug: str
    name: str
    base_currency: str
    mode: str | None = None
    symbols: list[str]
    positions: dict[str, float]
    created_at: str | None = None
    updated_at: str | None = None


class ArtifactSummary(BaseModel):
    id: int
    artifact_type: str
    format: str
    path: str
    size_bytes: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class BacktestRunSummary(BaseModel):
    id: int
    portfolio_id: int | None = None
    artifact_id: int | None = None
    timeframe: str | None = None
    days: int | None = None
    alpha: float
    window: int
    n_rows: int
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class ValidationRunSummary(BaseModel):
    id: int
    portfolio_id: int | None = None
    source_artifact_id: int | None = None
    alpha: float
    expected_rate: float
    best_model: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class SnapshotSummary(BaseModel):
    id: int
    portfolio_id: int | None = None
    artifact_id: int | None = None
    source: str
    alpha: float | None = None
    timeframe: str | None = None
    days: int | None = None
    window: int | None = None
    live_pnl: float | None = None
    live_loss: float | None = None
    breach_hist: bool | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class AlertSummary(BaseModel):
    id: int
    portfolio_id: int | None = None
    snapshot_id: int | None = None
    validation_run_id: int | None = None
    source: str
    severity: str
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    repo_root: str
    database_url: str
    portfolio_slug: str
    portfolio_mode: str | None = None
    portfolio_count: int
    desk_slug: str | None = None
    latest_artifacts: dict[str, str | None]
    defaults: dict[str, Any]
    dependencies: dict[str, Any] = Field(default_factory=dict)


class WorkerJobStatusResponse(BaseModel):
    enabled: bool
    interval_seconds: int
    state: str
    due: bool
    healthy: bool
    last_run_at: str | None = None
    last_run_age_seconds: float | None = None
    artifact_path: str | None = None


class WorkerStatusResponse(BaseModel):
    generated_at: str
    loop_sleep_seconds: int
    database_ready: bool
    jobs: dict[str, WorkerJobStatusResponse]


class RunSnapshotRequest(BaseModel):
    portfolio_slug: str | None = None
    timeframe: str | None = None
    days: int | None = None
    min_coverage: float | None = None
    alpha: float | None = None
    window: int | None = None
    n_sims: int | None = None
    dist: str | None = None
    df_t: int | None = None
    seed: int | None = None


class RunBacktestRequest(BaseModel):
    portfolio_slug: str | None = None
    timeframe: str | None = None
    days: int | None = None
    min_coverage: float | None = None
    alpha: float | None = None
    window: int | None = None
    n_sims: int | None = None
    dist: str | None = None
    df_t: int | None = None
    seed: int | None = None


class RunReportRequest(BaseModel):
    compare_path: str | None = None
    portfolio_slug: str | None = None


class TradeProposalRequest(BaseModel):
    portfolio_slug: str | None = None
    symbol: str
    delta_position_eur: float
    note: str | None = None


class CapitalRebalanceRequest(BaseModel):
    portfolio_slug: str | None = None
    total_budget_eur: float | None = None
    reserve_ratio: float | None = None
    reference_model: str | None = None
    symbol_weights: dict[str, float] | None = None


class ExecutionRequest(BaseModel):
    portfolio_slug: str | None = None
    symbol: str
    delta_position_eur: float
    note: str | None = None


class MarketDataSyncRequest(BaseModel):
    portfolio_slug: str | None = None
    days: int | None = None
    timeframes: list[str] | None = None


class SnapshotRunResponse(BaseModel):
    snapshot_id: int
    artifact_id: int
    artifact_path: str
    portfolio_slug: str
    source: str
    snapshot: dict[str, Any]


class BacktestRunResponse(BaseModel):
    backtest_run_id: int
    validation_run_id: int
    compare_artifact_id: int
    validation_artifact_id: int
    compare_csv: str
    validation_json: str
    best_model: str | None = None
    alert_count: int
    exception_counts: dict[str, int]


class ReportRunResponse(BaseModel):
    report_markdown: str
    chart_paths: list[str]


class BacktestFrameResponse(BaseModel):
    compare_csv: str
    portfolio_slug: str | None = None
    rows: list[dict[str, Any]]


class ReportContentResponse(BaseModel):
    report_markdown: str
    portfolio_slug: str | None = None
    content: str
    chart_paths: list[str]


class ModelComparisonRow(BaseModel):
    rank: int
    model: str
    score: float
    actual_rate: float
    expected_rate: float
    exceptions: int
    p_uc: float
    p_ind: float
    p_cc: float
    traffic_light: str | None = None
    current_var: float | None = None
    current_es: float | None = None


class ModelComparisonResponse(BaseModel):
    alpha: float
    champion_model: str | None = None
    challenger_model: str | None = None
    score_gap: float | None = None
    rate_gap: float | None = None
    exception_gap: int | None = None
    current_var_gap: float | None = None
    current_es_gap: float | None = None
    snapshot_source: str | None = None
    snapshot_timestamp: str | None = None
    ranking: list[ModelComparisonRow]


class RiskAttributionPositionResponse(BaseModel):
    symbol: str
    position_eur: float
    standalone_var: float
    standalone_es: float
    incremental_var: float
    incremental_es: float
    marginal_var: float
    marginal_es: float
    component_var: float
    component_es: float
    contribution_pct_var: float | None = None
    contribution_pct_es: float | None = None


class RiskAttributionModelResponse(BaseModel):
    model: str
    total_var: float
    total_es: float
    positions: dict[str, RiskAttributionPositionResponse]


class RiskAttributionResponse(BaseModel):
    alpha: float
    sample_size: int
    snapshot_source: str | None = None
    snapshot_timestamp: str | None = None
    models: dict[str, RiskAttributionModelResponse]


class RiskBudgetPositionResponse(BaseModel):
    symbol: str
    position_eur: float
    weight: float
    target_var_budget: float
    target_es_budget: float
    component_var: float
    component_es: float
    utilized_var: float
    utilized_es: float
    utilization_var: float | None = None
    utilization_es: float | None = None
    headroom_var: float
    headroom_es: float
    max_position_eur: float | None = None
    recommended_position_eur: float | None = None
    action: str
    status: str


class RiskBudgetModelResponse(BaseModel):
    model: str
    total_var: float
    total_es: float
    total_var_budget: float
    total_es_budget: float
    utilization_var: float | None = None
    utilization_es: float | None = None
    headroom_var: float
    headroom_es: float
    scale_to_var_budget: float | None = None
    scale_to_es_budget: float | None = None
    recommended_scale: float | None = None
    current_gross_notional: float
    recommended_gross_notional: float | None = None
    status: str
    positions: dict[str, RiskBudgetPositionResponse]


class RiskBudgetResponse(BaseModel):
    alpha: float
    sample_size: int
    preferred_model: str
    snapshot_source: str | None = None
    snapshot_timestamp: str | None = None
    models: dict[str, RiskBudgetModelResponse]


class RiskDecisionStateResponse(BaseModel):
    var: float
    es: float
    budget_utilization_var: float | None = None
    budget_utilization_es: float | None = None
    headroom_var: float
    headroom_es: float
    gross_notional: float
    position_eur: float
    status: str


class RiskDecisionResponse(BaseModel):
    id: int | None = None
    portfolio_id: int | None = None
    time_utc: str | None = None
    created_at: str | None = None
    decision_mode: str | None = None
    portfolio_slug: str | None = None
    timeframe: str | None = None
    days: int | None = None
    window: int | None = None
    symbol: str
    decision: str
    requested_delta_position_eur: float
    approved_delta_position_eur: float
    suggested_delta_position_eur: float | None = None
    resulting_position_eur: float
    model_used: str
    reasons: list[str]
    note: str | None = None
    pre_trade: RiskDecisionStateResponse
    post_trade: RiskDecisionStateResponse


class CapitalBudgetSummaryResponse(BaseModel):
    reference_model: str
    total_budget_eur: float
    reserve_ratio: float
    reserved_capital_eur: float
    model_budgets: dict[str, float] = Field(default_factory=dict)
    symbol_budgets: dict[str, float] = Field(default_factory=dict)


class CapitalAllocationResponse(BaseModel):
    symbol: str
    weight: float
    target_capital_eur: float
    consumed_capital_eur: float
    reserved_capital_eur: float
    remaining_capital_eur: float
    utilization: float | None = None
    action: str
    status: str


class ModelCapitalBudgetResponse(BaseModel):
    model: str
    budget_eur: float
    consumed_eur: float
    remaining_eur: float
    utilization: float | None = None
    status: str


class ReallocationRecommendationResponse(BaseModel):
    symbol_from: str
    symbol_to: str
    amount_eur: float
    reason: str
    priority: int


class CapitalUsageSnapshotResponse(BaseModel):
    id: int | None = None
    portfolio_id: int | None = None
    portfolio_slug: str
    base_currency: str
    reference_model: str
    snapshot_source: str
    snapshot_timestamp: str | None = None
    source: str | None = None
    created_at: str | None = None
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_reserved_eur: float
    total_capital_remaining_eur: float
    headroom_ratio: float | None = None
    status: str
    budget: CapitalBudgetSummaryResponse
    models: dict[str, ModelCapitalBudgetResponse] = Field(default_factory=dict)
    allocations: dict[str, CapitalAllocationResponse] = Field(default_factory=dict)
    recommendations: list[ReallocationRecommendationResponse] = Field(default_factory=list)


class DeskDefinitionResponse(BaseModel):
    slug: str
    name: str
    base_currency: str
    portfolio_slugs: list[str]


class PortfolioCapitalSliceResponse(BaseModel):
    portfolio_slug: str
    portfolio_name: str
    reference_model: str
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_remaining_eur: float
    utilization: float | None = None
    status: str
    alert_count: int


class DeskSnapshotResponse(BaseModel):
    desk_slug: str
    desk_name: str
    base_currency: str
    generated_at: str | None = None
    total_capital_budget_eur: float
    total_capital_consumed_eur: float
    total_capital_reserved_eur: float
    total_capital_remaining_eur: float
    worst_status: str
    portfolios: list[PortfolioCapitalSliceResponse] = Field(default_factory=list)


class AuditEventResponse(BaseModel):
    id: int
    portfolio_id: int | None = None
    actor: str
    action_type: str
    object_type: str | None = None
    object_id: int | None = None
    created_at: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MT5TerminalStatusResponse(BaseModel):
    connected: bool
    ready: bool
    execution_enabled: bool
    trade_allowed: bool | None = None
    tradeapi_disabled: bool | None = None
    company: str | None = None
    terminal_path: str | None = None
    data_path: str | None = None
    commondata_path: str | None = None
    message: str
    timestamp_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MT5AccountSnapshotResponse(BaseModel):
    login: int | None = None
    name: str | None = None
    server: str | None = None
    currency: str | None = None
    leverage: int | None = None
    balance: float
    equity: float
    profit: float
    margin: float
    margin_free: float
    margin_level: float | None = None
    trade_allowed: bool | None = None
    timestamp_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MT5PositionResponse(BaseModel):
    ticket: int | None = None
    symbol: str
    side: str
    volume_lots: float
    signed_position_eur: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    comment: str | None = None
    time_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MT5PendingOrderResponse(BaseModel):
    ticket: int | None = None
    symbol: str
    side: str
    volume_initial: float
    volume_current: float
    price_open: float
    comment: str | None = None
    time_setup_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class InstrumentDefinitionResponse(BaseModel):
    id: int | None = None
    symbol: str
    asset_class: str
    contract_size: float | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    profit_currency: str | None = None
    margin_currency: str | None = None
    tick_size: float | None = None
    tick_value: float | None = None
    volume_min: float | None = None
    volume_max: float | None = None
    volume_step: float | None = None
    trading_mode: str | None = None
    source: str = "mt5"
    synced_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class HoldingSnapshotResponse(BaseModel):
    symbol: str
    asset_class: str
    side: str
    volume_lots: float
    signed_position_eur: float
    signed_units: float | None = None
    contract_size: float | None = None
    base_currency: str | None = None
    profit_currency: str | None = None
    margin_currency: str | None = None
    mark_price: float | None = None
    market_value_base_ccy: float | None = None
    exposure_base_ccy: float | None = None
    unrealized_pnl_base_ccy: float | None = None
    profit: float
    source: str | None = None
    comment: str | None = None
    time_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PortfolioExposureItemResponse(BaseModel):
    symbol: str
    asset_class: str | None = None
    exposure_base_ccy: float
    signed_position_eur: float
    gross_exposure_share: float | None = None


class PortfolioExposureResponse(BaseModel):
    generated_at: str
    portfolio_slug: str
    portfolio_mode: str | None = None
    base_currency: str
    gross_exposure_base_ccy: float
    items: list[PortfolioExposureItemResponse] = Field(default_factory=list)


class OrderHistoryEntryResponse(BaseModel):
    id: int | None = None
    sync_run_id: int | None = None
    portfolio_id: int | None = None
    ticket: int | None = None
    position_id: int | None = None
    symbol: str
    side: str | None = None
    order_type: str | None = None
    state: str | None = None
    volume_initial: float | None = None
    volume_current: float | None = None
    price_open: float | None = None
    price_current: float | None = None
    comment: str | None = None
    is_manual: bool = False
    time_setup_utc: str | None = None
    time_done_utc: str | None = None
    source: str | None = None
    synced_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class DealHistoryEntryResponse(BaseModel):
    id: int | None = None
    sync_run_id: int | None = None
    portfolio_id: int | None = None
    ticket: int | None = None
    order_ticket: int | None = None
    position_id: int | None = None
    symbol: str
    side: str | None = None
    entry: str | None = None
    volume: float | None = None
    price: float | None = None
    profit: float | None = None
    commission: float | None = None
    swap: float | None = None
    fee: float | None = None
    reason: str | None = None
    comment: str | None = None
    is_manual: bool = False
    time_utc: str | None = None
    source: str | None = None
    synced_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MarketDataSyncStatusResponse(BaseModel):
    portfolio_slug: str
    portfolio_mode: str | None = None
    status: str
    configured: bool
    timeframe: str
    symbols: list[str] = Field(default_factory=list)
    instrument_count: int = 0
    latest_sync_at: str | None = None
    latest_bar_times: dict[str, str | None] = Field(default_factory=dict)
    missing_symbols: list[str] = Field(default_factory=list)
    missing_bars: list[str] = Field(default_factory=list)
    open_positions: list[HoldingSnapshotResponse] = Field(default_factory=list)
    pending_orders: list[MT5PendingOrderResponse] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ReconciliationMismatchResponse(BaseModel):
    symbol: str
    asset_class: str | None = None
    desk_exposure_eur: float
    live_exposure_eur: float
    difference_eur: float
    desk_volume_lots: float | None = None
    live_volume_lots: float | None = None
    order_ticket: int | None = None
    deal_ticket: int | None = None
    position_id: int | None = None
    reason: str | None = None
    status: str


class ReconciliationSummaryResponse(BaseModel):
    generated_at: str
    portfolio_slug: str
    portfolio_mode: str | None = None
    market_data_status: str
    latest_sync_at: str | None = None
    open_positions_count: int
    pending_orders_count: int
    manual_event_count: int
    unmatched_execution_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    holdings: list[HoldingSnapshotResponse] = Field(default_factory=list)
    mismatches: list[ReconciliationMismatchResponse] = Field(default_factory=list)
    recent_execution_attempts: list[ExecutionResultResponse] = Field(default_factory=list)
    recent_fills: list[ExecutionFillResponse] = Field(default_factory=list)


class MT5TickResponse(BaseModel):
    symbol: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    time_utc: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class MT5LiveEventChangeSummaryResponse(BaseModel):
    symbols_added: list[str] = Field(default_factory=list)
    symbols_removed: list[str] = Field(default_factory=list)
    open_positions: int = 0
    pending_orders: int = 0
    order_history: int = 0
    deal_history: int = 0


class MT5LiveStateResponse(BaseModel):
    sequence: int
    source: str | None = None
    status: str
    connected: bool
    degraded: bool
    stale: bool
    generated_at: str
    last_success_at: str | None = None
    last_error: str | None = None
    poll_interval_seconds: float
    history_poll_interval_seconds: float
    history_lookback_minutes: int
    portfolio_slug: str | None = None
    portfolio_mode: str | None = None
    symbols: list[str] = Field(default_factory=list)
    terminal_status: MT5TerminalStatusResponse | None = None
    account: MT5AccountSnapshotResponse | None = None
    ticks: dict[str, MT5TickResponse] = Field(default_factory=dict)
    holdings: list[HoldingSnapshotResponse] = Field(default_factory=list)
    pending_orders: list[MT5PendingOrderResponse] = Field(default_factory=list)
    order_history: list[OrderHistoryEntryResponse] = Field(default_factory=list)
    deal_history: list[DealHistoryEntryResponse] = Field(default_factory=list)
    exposure: PortfolioExposureResponse | None = None
    reconciliation: ReconciliationSummaryResponse | None = None


class MT5LiveEventResponse(BaseModel):
    sequence: int
    kind: str
    timestamp_utc: str
    change_summary: MT5LiveEventChangeSummaryResponse = Field(default_factory=MT5LiveEventChangeSummaryResponse)
    state: MT5LiveStateResponse


class ExecutionGuardDecisionResponse(BaseModel):
    decision: str
    risk_decision: str
    requested_delta_position_eur: float
    approved_delta_position_eur: float
    executable_delta_position_eur: float
    suggested_delta_position_eur: float | None = None
    model_used: str
    side: str | None = None
    volume_lots: float
    price: float | None = None
    execution_enabled: bool
    submit_allowed: bool
    margin_ok: bool
    margin_required: float | None = None
    free_margin_after: float | None = None
    order_check_retcode: int | None = None
    order_check_comment: str | None = None
    reasons: list[str] = Field(default_factory=list)


class ExecutionPreviewResponse(BaseModel):
    time_utc: str
    portfolio_slug: str
    symbol: str
    terminal_status: MT5TerminalStatusResponse
    account: MT5AccountSnapshotResponse
    live_positions: list[MT5PositionResponse] = Field(default_factory=list)
    pending_orders: list[MT5PendingOrderResponse] = Field(default_factory=list)
    risk_decision: RiskDecisionResponse
    guard: ExecutionGuardDecisionResponse
    order_request: dict[str, Any] = Field(default_factory=dict)
    order_check: dict[str, Any] = Field(default_factory=dict)
    pre_capital: dict[str, Any] = Field(default_factory=dict)
    post_capital: dict[str, Any] = Field(default_factory=dict)


class ExecutionFillResponse(BaseModel):
    id: int | None = None
    execution_result_id: int | None = None
    portfolio_id: int | None = None
    symbol: str
    order_ticket: int | None = None
    deal_ticket: int | None = None
    position_id: int | None = None
    side: str | None = None
    entry: str | None = None
    volume_lots: float
    price: float | None = None
    profit: float | None = None
    commission: float | None = None
    swap: float | None = None
    fee: float | None = None
    reason: str | None = None
    comment: str | None = None
    is_manual: bool = False
    slippage_points: float | None = None
    time_utc: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ExecutionResultResponse(BaseModel):
    id: int | None = None
    portfolio_id: int | None = None
    decision_id: int | None = None
    created_at: str | None = None
    time_utc: str
    portfolio_slug: str
    symbol: str
    status: str
    requested_delta_position_eur: float
    approved_delta_position_eur: float
    executed_delta_position_eur: float
    requested_volume_lots: float | None = None
    approved_volume_lots: float | None = None
    submitted_volume_lots: float | None = None
    filled_volume_lots: float | None = None
    remaining_volume_lots: float | None = None
    fill_ratio: float | None = None
    broker_status: str | None = None
    position_id: int | None = None
    slippage_points: float | None = None
    reconciliation_status: str | None = None
    mt5_order_ticket: int | None = None
    mt5_deal_ticket: int | None = None
    terminal_status: MT5TerminalStatusResponse
    account_before: MT5AccountSnapshotResponse
    account_after: MT5AccountSnapshotResponse | None = None
    guard: ExecutionGuardDecisionResponse
    risk_decision: RiskDecisionResponse
    order_request: dict[str, Any] = Field(default_factory=dict)
    order_check: dict[str, Any] = Field(default_factory=dict)
    mt5_result: dict[str, Any] = Field(default_factory=dict)
    positions_after: list[MT5PositionResponse] = Field(default_factory=list)
    post_capital: dict[str, Any] = Field(default_factory=dict)
    fills: list[ExecutionFillResponse] = Field(default_factory=list)
