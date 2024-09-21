import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import minimize


# Set page layout to wide
st.set_page_config(layout="wide")

# Black-Scholes Options Pricing Model
def black_scholes_price(S, K, T, r, sigma, option_type='Call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'Call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_scholes_delta(S, K, T, r, sigma, option_type='Call'):
    S = np.asarray(S)
    if T <= 0:
        return np.where(S >= K, 1 if option_type == 'Call' else 0, 0 if option_type == 'Call' else -1)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'Call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def black_scholes_theta(S, K, T, r, sigma, option_type='Call'):
    S = np.asarray(S)
    if T <= 0:
        return - (K - S) if option_type == 'Call' else - (S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta

# Function to calculate implied volatility
def implied_volatility(S, K, T, r, market_price, option_type='Call'):
    def objective_function(sigma):
        return (black_scholes_price(S, K, T, r, sigma, option_type) - market_price) ** 2
    
    result = minimize(objective_function, 0.2, bounds=[(0.001, 5)])
    return result.x[0] if result.success else np.nan

# Binomial Options Pricing Model
def binomial_option_pricing(S, K, T, r, sigma, N, option_type='Call'):
    dt = T / N  # single time step
    u = np.exp(sigma * np.sqrt(dt))  # up factor
    d = 1 / u  # down factor
    q = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    # Initialize asset prices at maturity
    ST = np.zeros(N + 1)
    ST[0] = S * d**N
    for j in range(1, N + 1):
        ST[j] = ST[j - 1] * u / d

    # Initialize option values at maturity
    C = np.zeros(N + 1)
    if option_type == 'Call':
        C = np.maximum(0, ST - K)
    else:
        C = np.maximum(0, K - ST)

    # Step back through the tree
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            C[j] = np.exp(-r * dt) * (q * C[j + 1] + (1 - q) * C[j])

    return C[0]

# Streamlit App
st.title('Multi-Leg Options P&L Analyzer')

# Sidebar Input
symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL', key='symbol_input')


# Manage state for stock price and calculated volatility
if 'stock_price' not in st.session_state:
    st.session_state.stock_price = None
if 'strike_price' not in st.session_state:
    st.session_state.strike_price = None
if 'option_price' not in st.session_state:
    st.session_state.option_price = None
if 'calculated_volatility' not in st.session_state:
    st.session_state.calculated_volatility = None
if 'op_cost' not in st.session_state:
    st.session_state.op_cost = None

# Fetch and update stock price when symbol changes
if symbol:
    stock_data = yf.Ticker(symbol)
    stock_price_history = stock_data.history(period='1d')
    
    if not stock_price_history.empty:
        new_stock_price = stock_price_history['Close'].iloc[0]
        st.sidebar.write(f'Last Price: {round(new_stock_price,2)}')
        if st.session_state.stock_price != new_stock_price:
            st.session_state.stock_price = new_stock_price
            st.session_state.strike_price = float(np.floor(new_stock_price))

    else:
        st.warning("No data available for the selected symbol or period. Please try another symbol or check the market status.")
        st.session_state.stock_price = None
        
strike_price_default = st.session_state.strike_price if st.session_state.strike_price is not None else 100.0

legs = 4

leg_params = []

for i in range(legs):
    st.subheader(f"Leg {i+1}")
    cols = st.columns(9)
    
    with cols[0]:
        quantity = st.number_input('Quantity: ', value=1, min_value=1, key=f'quantity_{i}')
    
    with cols[1]:
        buy_sell = st.selectbox('Buy/Sell', ['Buy', 'Sell'], key=f'buy_sell_{i}')
    
    with cols[2]:
        option_type = st.selectbox('Option Type', ['Call', 'Put'], key=f'option_type_{i}')
    
    with cols[3]:
        strike_price = st.number_input('Strike Price: ', value=float(st.session_state.strike_price), key=f'strike_price_{i}')
    
    with cols[4]:
        expiration_days = st.number_input('Days to Expiration: ', min_value=0, value=30, key=f'expiration_days_{i}')
    
    with cols[5]:
        risk_free_rate = st.number_input('Risk-Free-Rate: ', value=5.0, min_value=0.0, key=f'risk_free_rate_{i}') / 100
        
    with cols[6]:
        option_price = st.number_input('Option Price: ', value=st.session_state.option_price if st.session_state.option_price is not None else 10.0, key=f'option_price_input{i}')
    
    # # Calculate volatility based on option price
    # if st.session_state.stock_price is not None and strike_price is not None and expiration_days > 0:
    #     T = expiration_days / 365
    #     st.session_state.calculated_volatility = implied_volatility(
    #         st.session_state.stock_price, strike_price, T, risk_free_rate, option_price, option_type
    #     )
    
    
    with cols[7]:
        op_cost = st.number_input('Option Cost: ', value=st.session_state.op_cost if st.session_state.op_cost is not None else option_price, key = f'op_cost_{i}')
        
    with cols[8]:
        included = st.checkbox('Included', key = f'included_{i}')
        
    # # Use the calculated volatility if available
    # volatility = st.session_state.calculated_volatility if st.session_state.calculated_volatility is not None else 0.20
     
        
    leg_params.append((quantity, buy_sell, option_type, strike_price, expiration_days, risk_free_rate, option_price, op_cost, included))

if st.session_state.stock_price is not None:
    stock_price = st.session_state.stock_price
    N=200
    
# Calculate cumulative P&L, Delta, Theta across all legs
stock_prices = np.linspace(stock_price * 0.9, stock_price * 1.1, 100)
pnl_total = np.zeros_like(stock_prices)
pnl_expiration = np.zeros_like(stock_prices)
delta_total = np.zeros_like(stock_prices)
theta_total = np.zeros_like(stock_prices)



for quantity, buy_sell, option_type, strike_price, expiration_days, risk_free_rate, option_price, op_cost, included in leg_params:
    if included:
        multiplier = quantity * (1 if buy_sell == 'Buy' else -1)
        T = expiration_days / 365
        
        # Calculate implied volatility based on option price
        implied_vol = implied_volatility(
            stock_price, strike_price, T, risk_free_rate, option_price, option_type)
        
        print('Inputs: ',stock_price, strike_price, T, risk_free_rate, option_price, option_type,'\n\n\n')
        
        print(T,implied_vol, implied_volatility(
            stock_price, strike_price, T, risk_free_rate, option_price, option_type))
        
        pnl_leg = (np.array([black_scholes_price(S, strike_price, T, risk_free_rate, implied_vol, option_type)
                         if T <= 0.01 
                         else binomial_option_pricing(S, strike_price, T, risk_free_rate, implied_vol, N, option_type)
                         for S in stock_prices]) - op_cost) * (quantity if buy_sell == 'Buy' else -quantity) * 100
        
        print(implied_vol, pnl_leg)
        
        pnl_at_expiration = np.array([
        (max(S - strike_price, 0) if option_type == 'Call' else max(strike_price - S, 0)) - op_cost
        for S in stock_prices]) * multiplier * 100
        
        delta_leg = multiplier * np.array([black_scholes_delta(S, strike_price, T, risk_free_rate, implied_vol, option_type) for S in stock_prices])
        theta_leg = multiplier * np.array([black_scholes_theta(S, strike_price, T, risk_free_rate, implied_vol, option_type) for S in stock_prices])
        
        pnl_total += pnl_leg
        pnl_expiration += pnl_at_expiration
        delta_total += delta_leg
        theta_total += theta_leg
    
with cols[0]:
    # Calculate Delta and Theta if checkboxes are checked
    include_delta = st.checkbox('Include Delta', value=False)
    include_theta = st.checkbox('Include Theta', value=False)
    
with cols[1]:
    pl_expiration = st.checkbox('Show at Expiration', value=False)
    include_shading = st.checkbox('Shading', value=False)



# Create Plotly figure
fig = go.Figure()

# Plot P&L
fig.add_trace(go.Scatter(x=stock_prices, y=pnl_total, mode='lines', name='P&L',
                         hovertemplate='Stock Price: %{x}<br>P&L: $%{y:.2f}<br>' + (
    'Delta: %{customdata[0]:.2f}<br>Theta: %{customdata[1]:.2f}') +
    '<extra></extra>',
customdata=np.column_stack((delta_total, theta_total))
))

if pl_expiration:

    # Add P&L at expiration trace to the plot
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=pnl_expiration,
        mode='lines',
        name='P&L at Expiration',
        line=dict(color='green', width=1),
        opacity=0.5,
        hovertemplate='Stock Price: %{x}<br>P&L at Expiration: $%{y:.2f}<extra></extra>'
    ))

if include_shading:
    
    # Add shaded area for positive P&L
    fig.add_trace(go.Scatter(
        x=stock_prices[pnl_total >= 0],
        y=pnl_total[pnl_total >= 0],
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='rgba(0, 255, 0, 0.2)'),
        mode='none',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add shaded area for negative P&L
    fig.add_trace(go.Scatter(
        x=stock_prices[pnl_total < 0],
        y=pnl_total[pnl_total < 0],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0.2)'),
        mode='none',
        showlegend=False,
        hoverinfo='skip'
    ))


if include_delta:
    
    print(delta_total)
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=delta_total,
        mode='lines',
        name='Delta',
        yaxis='y2',
        line=dict(dash='dash', color='red', width=1),
        opacity=0.3,
        hovertemplate='Stock Price: %{x}<br>Delta: %{y:.2f}<br>' + (
    'P&L: $%{customdata[0]:.2f}<br>Theta: %{customdata[1]:.2f}') +
    '<extra></extra>',
customdata=np.column_stack((pnl_total, theta_total))
    ))

if include_theta:
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=theta_total,
        mode='lines',
        name='Theta',
        yaxis='y3',
        line=dict(dash='dash', color='blue', width=1),
        opacity=0.3,
        hovertemplate='Stock Price: %{x}<br>Theta: %{y:.2f}<br>' + (
    'P&L: $%{customdata[0]:.2f}<br>Delta: %{customdata[1]:.2f}') +
    '<extra></extra>',
customdata=np.column_stack((pnl_total, delta_total))
    ))

fig.update_layout(
title='Options P&L and Greeks vs Stock Price',
xaxis_title='Stock Price',
yaxis_title='P&L',
yaxis2=dict(
    title='',
    overlaying='y',
    side='right',
    showticksuffix='none',
    showticklabels=False,
    showline=False,  # Hide y-axis line
    showgrid=False,  # Hide y-axis grid lines
    zeroline=False   # Hide zero line
),
yaxis3=dict(
    title='',
    overlaying='y',
    side='right',
    showticksuffix='none',
    showticklabels=False,
    showline=False,  # Hide y-axis line
    showgrid=False,  # Hide y-axis grid lines
    zeroline=False   # Hide zero line
),
template='plotly_white',
height=600   # Adjust height as needed
)

st.plotly_chart(fig, use_container_width=True)