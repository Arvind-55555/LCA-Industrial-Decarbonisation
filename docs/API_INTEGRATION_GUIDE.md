# API Integration Guide

This guide explains how to set up and use real API integrations with the LCA Optimizer.

## Quick Start

1. **Set up API keys**:
   ```bash
   python examples/setup_api_keys.py
   ```

2. **Test API connections**:
   ```bash
   python examples/real_api_usage.py
   ```

## Supported APIs

### 1. Electricity Maps

**Purpose**: Real-time grid carbon intensity data

**Setup**:
1. Sign up at https://www.electricitymaps.com/
2. Get your API key from the dashboard
3. Add to `.env`: `ELECTRICITY_MAPS_API_KEY=your_key_here`

**Usage**:
```python
from lca_optimizer.data.grid_data_enhanced import ElectricityMapsLoader

loader = ElectricityMapsLoader(api_key="your_key")
ci = loader.get_current_carbon_intensity("DE")  # Germany
print(f"Carbon Intensity: {ci.carbon_intensity} g CO2eq/kWh")
```

**Features**:
- Real-time carbon intensity
- Historical data
- Renewable energy share
- Caching for rate limit compliance

### 2. WattTime

**Purpose**: Real-time marginal operating emission rates (MOER)

**Setup**:
1. Sign up at https://www.watttime.org/
2. Get username and password
3. Add to `.env`:
   ```
   WATTTIME_USERNAME=your_username
   WATTTIME_PASSWORD=your_password
   ```

**Usage**:
```python
from lca_optimizer.data.grid_data_enhanced import WattTimeLoader

loader = WattTimeLoader(username="user", password="pass")
ci = loader.get_current_carbon_intensity("CAISO")  # California
```

**Features**:
- Real-time MOER data
- Balancing authority coverage
- Automatic authentication

### 3. ENTSO-E Transparency Platform

**Purpose**: European grid data

**Setup**:
1. Register at https://transparency.entsoe.eu/
2. Get security token
3. Add to `.env`: `ENTSOE_SECURITY_TOKEN=your_token`

**Usage**:
```python
# Coming soon - ENTSO-E integration
```

### 4. GREET Database

**Purpose**: Well-to-wheel emission factors for fuel pathways

**Setup**:
- No API key required
- Can use default factors or load custom data files

**Usage**:
```python
from lca_optimizer.data.greet_integration import GREETIntegration

greet = GREETIntegration()
wtw_data = greet.get_wtw_emissions("hydrogen_electrolysis_wind")
print(f"WTW Emissions: {wtw_data['well_to_wheel']} g CO2eq/MJ")
```

**Features**:
- Pre-loaded default factors
- Custom data file support
- Pathway comparison

### 5. Brightway2 / Ecoinvent

**Purpose**: Life Cycle Inventory database

**Setup**:
1. Install Brightway2: `pip install brightway2`
2. Import Ecoinvent database (requires license)
3. Set up project

**Usage**:
```python
from lca_optimizer.data.brightway_integration import BrightwayIntegration

bw = BrightwayIntegration(project_name="lca_optimizer")
factors = bw.get_emission_factor("steel production")
```

**Requirements**:
- Ecoinvent database license
- Brightway2 installation
- Database import

## API Client Features

All API integrations use a unified `APIClient` with:

- **Automatic Retry**: Retries failed requests with exponential backoff
- **Rate Limiting**: Enforces API rate limits
- **Caching**: Caches responses to reduce API calls
- **Error Handling**: Graceful fallback to defaults
- **Timeout Management**: Prevents hanging requests

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Electricity Maps
ELECTRICITY_MAPS_API_KEY=your_key_here

# WattTime
WATTTIME_USERNAME=your_username
WATTTIME_PASSWORD=your_password

# ENTSO-E
ENTSOE_SECURITY_TOKEN=your_token

# Cache Settings
CACHE_DIR=data/cache
CACHE_ENABLED=true
CACHE_MAX_AGE_HOURS=1
```

### Programmatic Configuration

```python
from lca_optimizer.config.settings import Settings

settings = Settings(
    electricity_maps_api_key="your_key",
    watttime_username="user",
    watttime_password="pass"
)
```

## Caching

API responses are cached to:
- Reduce API calls
- Improve performance
- Handle rate limits

Cache location: `data/cache/` (configurable)

Cache duration: 1 hour by default (configurable)

## Error Handling

All API integrations include:
- Automatic fallback to default values
- Error logging
- Graceful degradation

If an API fails, the system will:
1. Log the error
2. Use cached data if available
3. Fall back to default values
4. Continue operation

## Rate Limits

### Electricity Maps
- Free tier: 10 requests/second
- Paid tiers: Higher limits

### WattTime
- Free tier: Limited requests
- Paid tiers: Higher limits

The API client automatically enforces rate limits.

## Testing API Connections

Run the test script:
```bash
python examples/real_api_usage.py
```

This will:
- Test all configured APIs
- Display current data
- Show any errors

## Troubleshooting

### API Key Not Working
1. Verify key is correct in `.env`
2. Check API key hasn't expired
3. Verify account is active

### Rate Limit Errors
1. Enable caching (default: enabled)
2. Increase cache duration
3. Reduce request frequency

### Connection Errors
1. Check internet connection
2. Verify API endpoint is accessible
3. Check firewall settings

## Best Practices

1. **Always use caching**: Reduces API calls and improves performance
2. **Handle errors gracefully**: Use try-except blocks
3. **Monitor API usage**: Track your API quota
4. **Use environment variables**: Never commit API keys
5. **Test with defaults first**: Verify system works without APIs

## Next Steps

1. Set up API keys using `examples/setup_api_keys.py`
2. Test connections with `examples/real_api_usage.py`
3. Integrate into your LCA calculations
4. Monitor API usage and costs

