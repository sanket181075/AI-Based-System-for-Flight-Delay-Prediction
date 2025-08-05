
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flight_predictor import FlightDelayPredictor
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1e3c72;
        margin: 1rem 0;
    }
    
    .flight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .delay-high {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
    }
    
    .delay-medium {
        border-left-color: #fd7e14;
        background: linear-gradient(135deg, #fff8f0 0%, #ffe8d1 100%);
    }
    
    .delay-low {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #f0fff4 0%, #e6ffe6 100%);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_test_data():
    return pd.read_csv('test_data.csv')

@st.cache_resource
def load_model():
    return FlightDelayPredictor.load('flight_delay_predictor.pkl')

def create_features(df):
    # Create a proper copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df['HOUR_OF_DAY'] = df['CRS_DEP_TIME'] // 100

    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['SEASON'] = df['MONTH'].map(season_map)

    bins = [0, 600, 1200, 1800, 2400]
    labels = ['Early_Morning', 'Morning', 'Afternoon', 'Evening']
    df['TIME_CATEGORY'] = pd.cut(df['CRS_DEP_TIME'], bins=bins, labels=labels, right=False)

    severity_map = {'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Severe': 4, 'Unknown': 1}
    df['SEVERITY_SCORE'] = df['Severity'].map(severity_map)

    df['PRECIP_CAT'] = pd.cut(
        df['Precipitation(in)'],
        bins=[-1, 0.01, 0.1, 0.5, float('inf')],
        labels=['None', 'Light', 'Moderate', 'Heavy']
    )

    df['HEAVY_RAIN'] = ((df['Type'] == 'Rain') & df['Severity'].isin(['Heavy', 'Severe'])).astype(int)
    df['SNOW_STORM'] = ((df['Type'] == 'Snow') & df['Severity'].isin(['Moderate', 'Heavy', 'Severe'])).astype(int)

    return df

def get_delay_category(probability):
    """Categorize delay probability into low, medium, high"""
    if probability < 0.45:
        return "Low", "🟢", "delay-low"
    elif probability < 0.65:
        return "Medium", "🟡", "delay-medium"
    else:
        return "High", "🔴", "delay-high"

def format_time(time_int):
    """Convert integer time to readable format"""
    hours = time_int // 100
    minutes = time_int % 100
    return f"{hours:02d}:{minutes:02d}"

def format_date(date_str):
    """Format date string to readable format"""
    date_obj = pd.to_datetime(date_str)
    return date_obj.strftime("%B %d, %Y (%A)")

# Main app
st.markdown('<div class="main-header"><h1>✈️ Flight Delay Predictor</h1><p>Get real-time predictions for your flight delays</p></div>', unsafe_allow_html=True)

# Load data and model
model = load_model()
test_df = load_test_data()

# Sidebar for filters
with st.sidebar:
    st.markdown("### 🎯 Search Filters")
    
    # Origin and destination selection
    origin = st.selectbox("🛫 Origin Airport", options=sorted(test_df['ORIGIN'].unique()), index=0)
    destination = st.selectbox("🛬 Destination Airport", options=sorted(test_df['DEST'].unique()), index=0)
    
    # Date selection
    available_dates = sorted(test_df['FL_DATE'].unique())
    selected_date = st.selectbox("📅 Flight Date", options=available_dates, index=0)
    
    # Time range selection
    st.markdown("### ⏰ Time Range")
    time_filter = st.checkbox("Filter by time range")
    
    if time_filter:
        # Get available times for selected route and date
        route_flights = test_df[
            (test_df['ORIGIN'] == origin) &
            (test_df['DEST'] == destination) &
            (test_df['FL_DATE'] == selected_date)
        ]
        
        if not route_flights.empty:
            available_times = sorted(route_flights['CRS_DEP_TIME'].unique())
            time_options = [f"{format_time(t)} ({t})" for t in available_times]
            
            start_time = st.selectbox("From", options=time_options, index=0)
            end_time = st.selectbox("To", options=time_options, index=len(time_options)-1)
            
            # Extract time values
            start_time_val = int(start_time.split('(')[1].split(')')[0])
            end_time_val = int(end_time.split('(')[1].split(')')[0])
        else:
            st.warning("No flights found for selected route and date")
            start_time_val = 0
            end_time_val = 2359
    else:
        start_time_val = 0
        end_time_val = 2359

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Flight Search Results")
    
    if st.button("🚀 Search Flights", type="primary"):
        # Filter flights based on selection
        filtered = test_df[
            (test_df['ORIGIN'] == origin) &
            (test_df['DEST'] == destination) &
            (test_df['FL_DATE'] == selected_date) &
            (test_df['CRS_DEP_TIME'] >= start_time_val) &
            (test_df['CRS_DEP_TIME'] <= end_time_val)
        ]

        if filtered.empty:
            st.error("❌ No flights found for the specified criteria. Please try different options.")
        else:
            # Create features and get predictions
            filtered = create_features(filtered)
            preds_df = model.predict(filtered)
            
            # Adjust threshold for more realistic predictions (balanced threshold for mixed results)
            # Using 0.5 for a balanced approach - flights with 50%+ probability will be marked as delayed
            adjusted_threshold = 0.5  # Balanced threshold for realistic predictions
            
            # Recalculate predictions with adjusted threshold
            preds_df['ADJUSTED_DELAY'] = (preds_df['DELAY_PROBABILITY'] >= adjusted_threshold).astype(int)
            
            # Display summary metrics
            total_flights = len(preds_df)
            delayed_flights = preds_df['ADJUSTED_DELAY'].sum()
            on_time_flights = total_flights - delayed_flights
            
            # Summary cards
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Total Flights</h3>
                    <h2>{total_flights}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✅ On Time</h3>
                    <h2 style="color: #28a745;">{on_time_flights}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️ Delayed</h3>
                    <h2 style="color: #dc3545;">{delayed_flights}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Flight details
            st.markdown("### 🛫 Flight Details")
            
            for idx, flight in preds_df.iterrows():
                delay_prob = flight['DELAY_PROBABILITY']
                delay_category, emoji, css_class = get_delay_category(delay_prob)
                is_delayed = flight['ADJUSTED_DELAY']
                
                # Format flight time
                flight_time = format_time(flight['CRS_DEP_TIME'])
                
                # Create flight card
                st.markdown(f"""
                <div class="flight-card {css_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>{flight['ORIGIN']} → {flight['DEST']}</h4>
                            <p><strong>Departure:</strong> {flight_time} | <strong>Date:</strong> {format_date(flight['FL_DATE'])}</p>
                            <p><strong>Delay Probability:</strong> {delay_prob:.1%}</p>
                        </div>
                        <div style="text-align: center;">
                            <h2>{emoji}</h2>
                            <p><strong>{delay_category} Risk</strong></p>
                            <p style="font-size: 0.9em; color: {'#dc3545' if is_delayed else '#28a745'};">
                                <strong>{'DELAYED' if is_delayed else 'ON TIME'}</strong>
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Delay Statistics")
    
    # Show some statistics about the selected route
    route_stats = test_df[
        (test_df['ORIGIN'] == origin) &
        (test_df['DEST'] == destination)
    ]
    
    if not route_stats.empty:
        # Create a simple delay probability distribution chart
        if st.button("📊 Show Route Statistics"):
            route_stats = create_features(route_stats)
            route_preds = model.predict(route_stats)
            
            # Create histogram of delay probabilities
            fig = px.histogram(
                route_preds, 
                x='DELAY_PROBABILITY',
                nbins=20,
                title=f"Delay Probability Distribution<br>{origin} → {destination}",
                labels={'DELAY_PROBABILITY': 'Delay Probability', 'count': 'Number of Flights'},
                color_discrete_sequence=['#1e3c72']
            )
            
            fig.update_layout(
                xaxis_title="Delay Probability",
                yaxis_title="Number of Flights",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show average delay probability
            avg_delay_prob = route_preds['DELAY_PROBABILITY'].mean()
            st.metric("Average Delay Probability", f"{avg_delay_prob:.1%}")
    
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. **Select your route** - Choose origin and destination airports
    2. **Pick your date** - Select your travel date
    3. **Filter by time** (optional) - Choose your preferred time range
    4. **Click Search** - Get real-time delay predictions
    
    **Delay Categories:**
    - 🟢 **Low Risk** (< 45%): Likely on time
    - 🟡 **Medium Risk** (45-65%): Possible delays
    - 🔴 **High Risk** (> 65%): High chance of delays
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>✈️ Flight Delay Predictor | Powered by Machine Learning</p>
    <p>Predictions are based on historical data and weather conditions</p>
</div>
""", unsafe_allow_html=True)
