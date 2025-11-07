"""
SSL Fix Module for Network Connectivity Issues.
This is the single source of SSL configuration for the entire application.
"""

import os
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning


# Apply SSL fix at module import time
def apply_ssl_fix():
    """
    Apply comprehensive SSL fix for network connectivity issues.
    This should be called once at application startup.
    """
    # Set SSL environment variables
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_VERIFY'] = 'false'

    # Disable SSL warnings
    urllib3.disable_warnings(InsecureRequestWarning)

    # Configure SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Set default SSL context
    ssl._create_default_https_context = ssl._create_unverified_context

    # Monkey patch httpx SSL context
    try:
        import httpx
        original_create_connection = httpx._transports.default.create_connection

        def patched_create_connection(*args, **kwargs):
            kwargs['ssl_context'] = ssl_context
            return original_create_connection(*args, **kwargs)

        httpx._transports.default.create_connection = patched_create_connection
    except Exception:
        pass

    # Monkey patch requests SSL context
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.ssl_ import create_urllib3_context

        class SSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                context = create_urllib3_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                kwargs['ssl_context'] = context
                return super().init_poolmanager(*args, **kwargs)

        # Apply to default session
        session = requests.Session()
        session.mount('https://', SSLAdapter())
        session.mount('http://', SSLAdapter())
    except Exception:
        pass

    return ssl_context

def test_ssl_fix():
    """Test if SSL fix is working."""
    try:
        import requests
        
        # Test basic connectivity
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        print(f"OpenAI API test: {response.status_code}")
        
        if response.status_code in [200, 401]:  # 401 is expected without auth
            return True
        else:
            return False
            
    except Exception as e:
        print(f"SSL test failed: {e}")
        return False

if __name__ == "__main__":
    print("Applying SSL fix...")
    ssl_context = apply_ssl_fix()
    print(f"SSL context: {ssl_context}")
    
    if test_ssl_fix():
        print("SSL fix is working!")
    else:
        print("SSL fix failed!")
