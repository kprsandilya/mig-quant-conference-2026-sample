import numpy as np
import talib

def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Arguments:
        prices  -- np.ndarray of shape (num_stocks, num_days)
                   Contains the Open price for each stock on each day.

    Returns:
        actions -- np.ndarray of the same shape (num_stocks, num_days).
                   Each value is the number of shares TRADED on that day.
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)
    
    # Require enough data to calculate the 50-day EMA and MACD safely
    if num_days <= 50:
        return actions

    for i in range(num_stocks):
        # ta-lib requires float64 1D arrays
        stock_prices = prices[i, :].astype(np.float64)
        
        # Calculate Trend Indicators
        ema_fast = talib.EMA(stock_prices, timeperiod=10)
        ema_slow = talib.EMA(stock_prices, timeperiod=50)
        
        # Calculate Momentum Indicator
        macd, signal, _ = talib.MACD(stock_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        
        position = 0
        for t in range(50, num_days):
            current_price = stock_prices[t]
            
            # Using t-1 indicators to make a decision for day t's Open price trade
            bullish_trend = ema_fast[t-1] > ema_slow[t-1]
            bullish_momentum = macd[t-1] > signal[t-1]
            
            bearish_trend = ema_fast[t-1] < ema_slow[t-1]
            bearish_momentum = macd[t-1] < signal[t-1]
            
            target_allocation = 500  # Target $500 per position
            target_shares = max(1, int(target_allocation / current_price))
            
            # Go Long if trend and momentum are bullish
            if bullish_trend and bullish_momentum and position <= 0:
                # Buy enough to cover any existing short AND establish the new long
                actions[i, t] = target_shares + abs(position)
                position = target_shares
                
            # Go Short if trend and momentum are bearish
            elif bearish_trend and bearish_momentum and position >= 0:
                # Sell enough to close any existing long AND establish the new short
                actions[i, t] = -target_shares - position
                position = -target_shares

    return actions