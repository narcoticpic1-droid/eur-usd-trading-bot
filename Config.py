# =================================================================
# FOREX TRADING BOT CONFIGURATION
# Pure Price Action Analysis with Multi-AI Evaluation
# EUR/USD Focus with Risk Management
# =================================================================

import datetime
import os

# ----------------- #
# --- API Keys --- #
# ----------------- #

# IMPORTANT: Replace these with your actual keys
# OANDA API Configuration
OANDA_API_KEY = "YOUR_OANDA_API_KEY_HERE"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID_HERE"  
OANDA_ENVIRONMENT = "practice"  # "practice" or "live"

# Alternative Forex Brokers (for future use)
MT5_LOGIN = ""
MT5_PASSWORD = ""
MT5_SERVER = ""

FXCM_API_KEY = ""
FXCM_USERNAME = ""
FXCM_PASSWORD = ""

# AI Services API Keys
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
CLAUDE_API_KEY = "YOUR_API_KEY_HERE"

# Telegram Bot Configuration
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"

# ----------------- #
# -- Forex Settings -- #
# ----------------- #

# Primary Broker
PRIMARY_BROKER = 'oanda'

# Currency Pairs to Monitor (OANDA format)
FOREX_PAIRS = [
    'EUR_USD',    # Primary pair - high liquidity
    'GBP_USD',    # Major pair
    'USD_JPY',    # Major pair
    'USD_CHF',    # Safe haven pair
    'AUD_USD'     # Commodity currency
]

# Primary focus pair for detailed analysis
PRIMARY_PAIR = 'EUR_USD'

# Timeframe settings
TIMEFRAME = 'H1'  # 1 hour candles
KLINE_LIMIT = 200  # Number of candles to fetch
MIN_CANDLES_REQUIRED = 150  # Minimum for analysis

# Analysis interval (in seconds)
LOOP_INTERVAL_SECONDS = 3600  # 1 hour (matches timeframe)

# Market hours (UTC)
MARKET_HOURS = {
    'start': 22,  # Sunday 22:00 UTC (Sydney open)
    'end': 22,    # Friday 22:00 UTC (New York close)
    'check_holidays': True
}

# ----------------- #
# -- AI Configuration -- #
# ----------------- #

# OpenAI Settings
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 1500

# Gemini Settings  
GEMINI_MODEL = "gemini-1.5-pro"
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_TOKENS = 1500

# Claude Settings
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
CLAUDE_MAX_TOKENS = 1500

# Multi-AI Configuration
AI_SETUP = {
    'primary_ai': 'gemini',
    'secondary_ai': 'openai', 
    'claude_ai': 'enabled',
    'require_consensus': True,
    'consensus_threshold': 0.7,
    'ai_timeout_seconds': 30,
    'enable_ai_learning': True
}

# ----------------- #
# -- Risk Management -- #
# ----------------- #

# Conservative Forex Leverage (CRITICAL FOR SAFETY)
MIN_LEVERAGE = 1
MAX_LEVERAGE = 5  # Much safer for Forex
DEFAULT_LEVERAGE = 2

# Pair-specific leverage limits
PAIR_MAX_LEVERAGE = {
    'EUR_USD': 5,   # Most liquid pair
    'GBP_USD': 4,   # Moderate volatility  
    'USD_JPY': 5,   # Stable pair
    'USD_CHF': 3,   # Lower volatility
    'AUD_USD': 3    # Higher volatility
}

# Portfolio Risk Management
PORTFOLIO_MANAGEMENT = {
    'max_daily_risk': 0.02,  # 2% daily risk (very conservative)
    'max_single_position_risk': 0.005,  # 0.5% per trade
    'max_concurrent_positions': 2,  # Maximum 2 positions
    'max_total_exposure': 0.03,  # 3% total exposure
    'drawdown_limit': 0.05,  # 5% portfolio drawdown limit
    'daily_loss_limit': 0.03,  # 3% daily loss limit
    'cooling_off_hours': 8,  # 8 hour cooling off after losses
}

# Position Sizing (in account percentage)
POSITION_SIZING = {
    'EUR_USD': {'base_size': 0.005, 'max_size': 0.008},  # 0.5-0.8%
    'GBP_USD': {'base_size': 0.004, 'max_size': 0.006},  # 0.4-0.6%
    'USD_JPY': {'base_size': 0.005, 'max_size': 0.008},  # 0.5-0.8%
    'USD_CHF': {'base_size': 0.004, 'max_size': 0.006},  # 0.4-0.6%
    'AUD_USD': {'base_size': 0.003, 'max_size': 0.005}   # 0.3-0.5%
}

# ----------------- #
# -- Signal Quality Control -- #
# ----------------- #

# Signal Filters (Strict for Forex)
SIGNAL_FILTERS = {
    'min_confidence': 0.75,  # Higher confidence for Forex
    'min_risk_reward': 2.5,  # Better R:R ratio
    'require_volume_confirmation': False,  # Volume less relevant in Forex
    'require_ai_consensus': True,
    'max_distance_from_key_level': 0.001,  # 10 pips for EUR/USD
    'min_adx_for_trend_signals': 25,
    'require_market_hours': True
}

# Multi-pair coordination
PAIR_COORDINATION = {
    'max_correlated_positions': 1,  # No correlated positions
    'usd_strength_analysis': True,  # Analyze USD strength
    'avoid_opposing_signals': True,
    'correlation_threshold': 0.7
}

# ----------------- #
# -- Price Action Settings -- #
# ----------------- #

# Forex-specific Price Action
PRICE_ACTION_SETTINGS = {
    'min_candles_required': 150,
    'swing_detection_period': 10,  # Shorter for Forex
    'trend_strength_period': 14,
    'key_level_tolerance': 0.0008,  # 8 pips for EUR/USD
    'pattern_confirmation_bars': 2,
    'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
    'support_resistance_strength': 3  # Minimum touches
}

# Pair-specific adjustments
PAIR_SPECIFIC_SETTINGS = {
    'EUR_USD': {
        'pip_value': 0.0001,
        'spread_threshold': 2,  # 2 pips
        'volatility_multiplier': 1.0,
        'trend_following_bias': 1.1
    },
    'GBP_USD': {
        'pip_value': 0.0001, 
        'spread_threshold': 3,  # 3 pips
        'volatility_multiplier': 1.3,
        'trend_following_bias': 1.0
    },
    'USD_JPY': {
        'pip_value': 0.01,
        'spread_threshold': 2,
        'volatility_multiplier': 0.9,
        'trend_following_bias': 1.2
    },
    'USD_CHF': {
        'pip_value': 0.0001,
        'spread_threshold': 3,
        'volatility_multiplier': 0.8,
        'trend_following_bias': 1.1
    },
    'AUD_USD': {
        'pip_value': 0.0001,
        'spread_threshold': 3,
        'volatility_multiplier': 1.2,
        'trend_following_bias': 0.9
    }
}

# ----------------- #
# -- Circuit Breakers -- #
# ----------------- #

# Enhanced Safety for Forex
CIRCUIT_BREAKERS = {
    'enable_emergency_stop': True,
    'max_daily_trades_per_pair': 3,
    'max_total_daily_trades': 5,
    'consecutive_loss_limit': 2,
    'unusual_volatility_threshold': 0.02,  # 2% unusual move
    'news_event_protection': True,
    'market_close_protection': True,
    'weekend_protection': True,
    'api_error_threshold': 3
}

# Emergency Conditions
EMERGENCY_STOPS = {
    'portfolio_loss_24h': 0.05,  # 5% loss in 24h
    'single_position_loss': 0.02,  # 2% loss on single position
    'news_spike_detection': True,
    'spread_widening_threshold': 10,  # 10x normal spread
    'connection_loss_protection': True
}

# News Events to Avoid (Hours before/after)
NEWS_AVOIDANCE = {
    'high_impact_events': 2,  # 2 hours before/after
    'medium_impact_events': 1,  # 1 hour before/after
    'central_bank_meetings': 4,  # 4 hours before/after
    'nfp_release': 2,  # Non-farm payrolls
    'fomc_meetings': 4
}

# ----------------- #
# -- Technical Analysis -- #
# ----------------- #

# Technical Indicators
TECHNICAL_SETTINGS = {
    'ema_periods': [8, 21, 50, 200],
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'adx_period': 14,
    'atr_period': 14,
    'stochastic_k': 14,
    'stochastic_d': 3
}

# Support/Resistance Settings
SUPPORT_RESISTANCE = {
    'lookback_period': 50,
    'min_touches': 2,
    'proximity_threshold': 0.0005,  # 5 pips
    'strength_multiplier': {
        'daily': 3,
        'weekly': 5,
        'monthly': 8
    }
}

# ----------------- #
# -- Notifications -- #
# ----------------- #

# Notification Settings
NOTIFICATION_SETTINGS = {
    'send_forex_analysis': True,
    'send_market_overview': True,
    'send_risk_alerts': True,
    'send_news_warnings': True,
    'include_charts': False,
    'priority_levels': {
        'EMERGENCY': '🚨',
        'HIGH': '🔴', 
        'MEDIUM': '🟡',
        'LOW': '🟢',
        'INFO': 'ℹ️'
    }
}

# Message Formatting
MESSAGE_SETTINGS = {
    'max_message_length': 4000,
    'include_pip_values': True,
    'include_time_analysis': True,
    'use_forex_terminology': True,
    'group_related_signals': True
}

# ----------------- #
# -- Broker Settings -- #
# ----------------- #

# Connection Settings
BROKER_TIMEOUT = 30
BROKER_RETRIES = 3
BROKER_DELAY = 5

# Data Validation
DATA_VALIDATION = {
    'check_spread': True,
    'max_spread_multiplier': 3,  # 3x normal spread
    'check_gaps': True,
    'max_gap_size': 50,  # 50 pips
    'validate_timestamps': True,
    'weekend_data_filter': True
}

# ----------------- #
# -- Performance Tracking -- #
# ----------------- #

# Comprehensive Tracking
PERFORMANCE_TRACKING = {
    'track_per_pair': True,
    'track_per_ai_model': True,
    'track_time_of_day': True,
    'track_news_impact': True,
    'calculate_sharpe_ratio': True,
    'calculate_max_drawdown': True,
    'benchmark_vs_buy_hold': False,  # Not applicable to Forex
    'analysis_period_days': 30
}

# Advanced Metrics
ADVANCED_METRICS = {
    'track_signal_latency': True,
    'measure_slippage': True,
    'calculate_win_streaks': True,
    'analyze_session_performance': True,  # Asian, London, New York
    'track_volatility_periods': True,
    'correlation_analysis': True
}

# ----------------- #
# -- Learning System -- #
# ----------------- #

# Adaptive Learning
LEARNING_SYSTEM = {
    'enable_learning': True,
    'learning_rate': 0.01,
    'memory_period_days': 90,
    'min_samples_for_learning': 50,
    'feature_importance_tracking': True,
    'model_retraining_frequency': 'weekly',
    'cross_validation_folds': 5
}

# Feature Engineering
FEATURE_ENGINEERING = {
    'price_features': True,
    'technical_features': True,
    'time_features': True,
    'volatility_features': True,
    'correlation_features': True,
    'news_sentiment_features': False,  # Future enhancement
    'economic_calendar_features': False  # Future enhancement
}

# ----------------- #
# -- Logging and Debug -- #
# ----------------- #

DEBUG_MODE = True
DETAILED_LOGGING = True

LOGGING_SETTINGS = {
    'log_level': 'INFO',
    'log_trades': True,
    'log_ai_decisions': True,
    'log_risk_calculations': True,
    'log_market_analysis': True,
    'separate_log_per_pair': True,
    'max_log_size_mb': 100,
    'backup_count': 7
}

# Log file paths
LOG_PATHS = {
    'main': 'logs/forex_bot.log',
    'trades': 'logs/trades.log',
    'analysis': 'logs/analysis.log',
    'errors': 'logs/errors.log'
}

# ----------------- #
# -- Database Settings -- #
# ----------------- #

# SQLite Database
DATABASE_SETTINGS = {
    'db_path': 'data/forex_trading.db',
    'backup_frequency': 'daily',
    'backup_retention_days': 30,
    'enable_encryption': False  # Future enhancement
}

# ----------------- #
# -- Startup Configuration -- #
# ----------------- #

# Create necessary directories
REQUIRED_DIRECTORIES = [
    'logs',
    'data', 
    'backups',
    'reports'
]

for directory in REQUIRED_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# ----------------- #
# -- Startup Banner -- #
# ----------------- #

print("=" * 60)
print("🏦 FOREX PURE PRICE ACTION TRADING BOT")
print("📊 Primary Pair: EUR/USD")
print("🧠 Multi-AI Analysis: Gemini + OpenAI + Claude")
print("⏰ Timeframe: 1 Hour")
print(f"📈 Max Leverage: {MAX_LEVERAGE}x (Conservative)")
print("🛡️ Advanced Risk Management")
print("🔄 Analysis Every Hour")
print("=" * 60)

# CRITICAL SAFETY WARNINGS FOR FOREX
print("⚠️ FOREX TRADING WARNINGS:")
print("• Forex trading involves significant risk")
print("• Leverage amplifies both profits and losses")
print("• Economic news can cause rapid price movements")
print("• Always trade during appropriate market hours")
print("• This system provides analysis, not financial advice")
print("• Start with demo/practice account")
print("• Never risk more than you can afford to lose")
print("=" * 60)

# API Status Check
api_warnings = []

if not OANDA_API_KEY or "YOUR_OANDA" in OANDA_API_KEY:
    api_warnings.append("OANDA API Key not configured!")

if not OPENAI_API_KEY or "YOUR_OPENAI" in OPENAI_API_KEY:
    api_warnings.append("OpenAI API Key not configured!")

if not GEMINI_API_KEY or "YOUR_GEMINI" in GEMINI_API_KEY:
    api_warnings.append("Gemini API Key not configured!")

if not TELEGRAM_TOKEN or "YOUR_TELEGRAM" in TELEGRAM_TOKEN:
    api_warnings.append("Telegram Token not configured!")

if api_warnings:
    print("⚠️ CONFIGURATION WARNINGS:")
    for warning in api_warnings:
        print(f"• {warning}")
    print("=" * 60)

print("✅ Forex Configuration loaded")
print(f"🎯 Primary Broker: {PRIMARY_BROKER.upper()}")
print(f"🎯 Environment: {OANDA_ENVIRONMENT.upper()}")
print(f"📍 Monitoring {len(FOREX_PAIRS)} currency pairs")
print("=" * 60)

# Configuration validation
def validate_config():
    """Validate critical configuration settings"""
    errors = []
    
    if MAX_LEVERAGE > 10:
        errors.append("Max leverage too high for safe Forex trading")
    
    if PORTFOLIO_MANAGEMENT['max_single_position_risk'] > 0.02:
        errors.append("Single position risk too high")
        
    if not SIGNAL_FILTERS['require_ai_consensus']:
        errors.append("AI consensus should be required for safety")
    
    if errors:
        print("🚨 CONFIGURATION ERRORS:")
        for error in errors:
            print(f"• {error}")
        print("Please fix these issues before starting the bot")
        return False
    
    return True

# Validate configuration on import
CONFIG_VALID = validate_config()
