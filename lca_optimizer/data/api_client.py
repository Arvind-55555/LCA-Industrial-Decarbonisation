"""
Unified API Client with retry logic, error handling, and caching
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Optional, Any, Callable
import time
import logging
from functools import wraps
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class APIClient:
    """
    Unified API client with retry logic, rate limiting, and caching.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        rate_limit: Optional[float] = None,  # requests per second
        cache_dir: Optional[str] = None
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API
            api_key: API key (optional)
            headers: Additional headers
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_backoff: Backoff multiplier for retries
            rate_limit: Rate limit (requests per second)
            cache_dir: Directory for caching responses
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup headers
        self.headers = headers or {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _rate_limit_check(self):
        """Enforce rate limiting"""
        if self.rate_limit:
            elapsed = time.time() - self.last_request_time
            min_interval = 1.0 / self.rate_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self.last_request_time = time.time()
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and params"""
        key_data = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        return key_data.replace("/", "_").replace(":", "_")
    
    def _get_cached_response(self, cache_key: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Get cached response if available and not expired"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age > max_age_hours * 3600:
            cache_file.unlink()
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def _cache_response(self, cache_key: str, data: Dict):
        """Cache response"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Make GET request with retry logic and caching.
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            use_cache: Whether to use cached responses
            cache_max_age_hours: Maximum age of cached response in hours
        
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached = self._get_cached_response(cache_key, cache_max_age_hours)
            if cached:
                logger.debug(f"Using cached response for {endpoint}")
                return cached
        
        # Rate limiting
        self._rate_limit_check()
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Cache response
            if use_cache:
                self._cache_response(cache_key, data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json_data: JSON data
        
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Rate limiting
        self._rate_limit_check()
        
        try:
            response = self.session.post(
                url,
                data=data,
                json=json_data,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST request failed: {e}")
            raise

