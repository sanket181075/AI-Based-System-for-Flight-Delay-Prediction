# ✈️ Flight Delay Predictor

A machine learning-powered web application that predicts flight delays based on historical data, weather conditions, and flight characteristics.

## 🚀 Features

- **Real-time Predictions**: Get instant delay predictions for your flights
- **Interactive UI**: Modern, responsive web interface with beautiful styling
- **Time-based Filtering**: Filter flights by departure time ranges
- **Visual Indicators**: Color-coded delay risk categories (Low/Medium/High)
- **Route Statistics**: View delay probability distributions for specific routes
- **Weather Integration**: Considers weather conditions in predictions

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.12+
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: Azure App Service

## 📋 Prerequisites

- Python 3.12 or higher
- Git
- Azure account (for deployment)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flight-delay-predictor.git
   cd flight-delay-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Azure Deployment

1. **Create Azure App Service**
   ```bash
   az group create --name flight-delay-predictor-rg --location eastus
   az appservice plan create --name flight-delay-plan --resource-group flight-delay-predictor-rg --sku B1
   az webapp create --name flight-delay-predictor --resource-group flight-delay-predictor-rg --plan flight-delay-plan --runtime "PYTHON:3.12"
   ```

2. **Configure deployment settings**
   ```bash
   az webapp config set --name flight-delay-predictor --resource-group flight-delay-predictor-rg --startup-file "streamlit run app.py --server.port 8000 --server.address 0.0.0.0"
   ```

3. **Deploy from Git**
   ```bash
   az webapp deployment source config --name flight-delay-predictor --resource-group flight-delay-predictor-rg --repo-url https://github.com/yourusername/flight-delay-predictor.git --branch main
   ```

## 📁 Project Structure

```
flight-delay-predictor/
├── app.py                      # Main Streamlit application
├── flight_predictor.py         # ML model prediction class
├── flight_delay_predictor.pkl  # Trained model file
├── test_data.csv              # Test dataset
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
└── azure-deploy.yml         # Azure deployment configuration
```

## 🎯 How to Use

1. **Select Route**: Choose origin and destination airports
2. **Pick Date**: Select your travel date
3. **Filter Time** (optional): Choose preferred departure time range
4. **Get Predictions**: View delay predictions with risk categories

### Delay Categories

- 🟢 **Low Risk** (< 45%): Likely on time
- 🟡 **Medium Risk** (45-65%): Possible delays  
- 🔴 **High Risk** (> 65%): High chance of delays

## 🔧 Configuration

### Model Thresholds

The application uses configurable thresholds for delay predictions:

- **Main Threshold**: 0.5 (50% probability for delay classification)
- **Low Risk**: < 45%
- **Medium Risk**: 45-65%
- **High Risk**: > 65%

### Customization

You can adjust thresholds in:
- `app.py`: Line 180 (main threshold)
- `flight_predictor.py`: Line 42 (model threshold)

## 📊 Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: Weather conditions, flight times, seasonal patterns
- **Training Data**: Historical flight data with weather information
- **Performance**: Optimized for realistic delay predictions

## 🌐 Deployment

### Azure App Service

The application is configured for Azure App Service deployment with:

- **Runtime**: Python 3.12
- **Framework**: Streamlit
- **Port**: 8000 (configurable)
- **Scaling**: Automatic (configurable)

### Environment Variables

Set these in Azure App Service Configuration:

```bash
PYTHON_VERSION=3.12
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flight data sources
- Weather data providers
- Streamlit community
- Azure documentation

**Made with ❤️ for better travel planning** 
