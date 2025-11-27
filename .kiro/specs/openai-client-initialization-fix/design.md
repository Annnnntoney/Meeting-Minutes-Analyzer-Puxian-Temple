# Design Document

## Overview

本設計文件針對 Streamlit 應用程式中 OpenAI 客戶端初始化失敗問題提供解決方案。根據錯誤堆疊追蹤分析，問題出現在 `SyncHttpxClientWrapper` 初始化過程中，這表明缺少必要的 HTTP 客戶端依賴（httpx）或版本不相容。

主要解決策略：
1. 在 requirements.txt 中明確添加 httpx 依賴
2. 更新 OpenAI SDK 到更穩定的版本
3. 增強錯誤處理機制，提供更清晰的錯誤訊息
4. 添加 API Key 驗證邏輯

## Architecture

應用程式採用分層架構：

```
使用者介面層 (Streamlit UI)
    ↓
配置管理層 (API Key 管理)
    ↓
客戶端初始化層 (OpenAI Client)
    ↓
HTTP 傳輸層 (httpx)
    ↓
OpenAI API
```

關鍵改進點：
- 在客戶端初始化層添加完整的錯誤處理
- 確保 HTTP 傳輸層依賴正確安裝
- 在配置管理層添加 API Key 格式驗證

## Components and Interfaces

### 1. Dependency Management Component

**職責**: 管理所有 Python 套件依賴

**介面**:
- `requirements.txt`: 定義所有必要的套件及其版本

**關鍵變更**:
```python
# 添加明確的 httpx 依賴
httpx>=0.24.0
# 更新 OpenAI SDK 到更穩定版本
openai>=1.50.0
```

### 2. OpenAI Client Initialization Component

**職責**: 安全地初始化 OpenAI 客戶端並處理各種錯誤情況

**現有介面**:
```python
def _load_openai_client(explicit_key: Optional[str] = None) -> OpenAI
```

**改進後介面**:
```python
def _load_openai_client(explicit_key: Optional[str] = None) -> OpenAI:
    """
    初始化 OpenAI 客戶端，包含完整的錯誤處理
    
    Args:
        explicit_key: 明確提供的 API Key
        
    Returns:
        OpenAI: 已初始化的客戶端實例
        
    Raises:
        SystemExit: 當無法初始化客戶端時停止應用程式
    """
```

### 3. API Key Validation Component

**職責**: 驗證 API Key 格式的正確性

**新增介面**:
```python
def _validate_api_key(api_key: str) -> bool:
    """
    驗證 API Key 格式
    
    Args:
        api_key: 待驗證的 API Key
        
    Returns:
        bool: Key 格式是否有效
    """
```

## Data Models

### API Key Configuration

```python
@dataclass
class APIKeyConfig:
    key: str
    source: str  # "explicit", "secrets", "env"
    is_valid: bool
```

## Correctn
ess Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Valid API key initialization succeeds

*For any* valid API key format, initializing the OpenAI client should complete without raising exceptions and return a valid client instance.

**Validates: Requirements 1.1**

### Property 2: Invalid API key produces error message

*For any* invalid API key format (empty, malformed, or None), the system should display a clear error message and prevent further execution.

**Validates: Requirements 1.2**

### Property 3: Error logging completeness

*For any* error that occurs during client initialization, the system should log detailed error information including the error type and message.

**Validates: Requirements 2.4**

## Error Handling

### Error Categories

1. **Missing API Key**
   - Detection: API key is None, empty string, or only whitespace
   - Response: Display error message "請在側邊欄輸入 OpenAI API Key，或設定環境變數 OPENAI_API_KEY。"
   - Action: Stop application execution with `st.stop()`

2. **Invalid API Key Format**
   - Detection: API key doesn't match expected format (e.g., doesn't start with "sk-")
   - Response: Display warning message about potential invalid key format
   - Action: Attempt initialization but catch authentication errors

3. **Network/Connection Errors**
   - Detection: Catch `httpx` connection exceptions during initialization
   - Response: Display error message "無法連接到 OpenAI 服務，請檢查網路連線。"
   - Action: Stop application execution

4. **Authentication Errors**
   - Detection: Catch OpenAI authentication exceptions
   - Response: Display error message "API Key 無效，請檢查您的金鑰是否正確。"
   - Action: Stop application execution

5. **Missing Dependencies**
   - Detection: Catch ImportError for httpx or other required packages
   - Response: Display error message with installation instructions
   - Action: Stop application execution

### Error Handling Flow

```python
try:
    # Validate API key format
    if not api_key or not api_key.strip():
        raise ValueError("API key is empty")
    
    # Initialize client
    client = OpenAI(api_key=api_key)
    
    # Optional: Test connection with a simple API call
    # This helps catch authentication errors early
    
except ValueError as e:
    st.error("請在側邊欄輸入 OpenAI API Key，或設定環境變數 OPENAI_API_KEY。")
    st.stop()
except ImportError as e:
    st.error(f"缺少必要的依賴套件：{e}。請執行 pip install -r requirements.txt")
    st.stop()
except Exception as e:
    st.error(f"初始化 OpenAI 客戶端時發生錯誤：{e}")
    st.stop()
```

## Testing Strategy

### Unit Testing

本專案將使用 pytest 作為測試框架。單元測試將涵蓋：

1. **API Key Validation Tests**
   - Test valid API key formats
   - Test invalid API key formats (empty, None, whitespace-only)
   - Test API key retrieval from different sources

2. **Error Handling Tests**
   - Test error messages for missing API key
   - Test error messages for invalid credentials
   - Test error messages for network issues

3. **Integration Tests**
   - Test client initialization with mocked OpenAI API
   - Test full flow from API key input to client creation

### Property-Based Testing

本專案將使用 **Hypothesis** 作為 property-based testing 框架。

**Configuration**:
- Each property test will run a minimum of 100 iterations
- Tests will use Hypothesis strategies to generate diverse inputs

**Property Test Requirements**:
- Each property-based test MUST be tagged with a comment referencing the correctness property
- Tag format: `# Feature: openai-client-initialization-fix, Property {number}: {property_text}`
- Each correctness property MUST be implemented by a SINGLE property-based test

**Property Tests to Implement**:

1. **Property 1 Test: Valid API key initialization**
   ```python
   # Feature: openai-client-initialization-fix, Property 1: Valid API key initialization succeeds
   @given(valid_api_key=st.text(min_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
   @settings(max_examples=100)
   def test_valid_api_key_initialization(valid_api_key):
       # Test that valid-looking API keys don't cause initialization to crash
       # Note: We can't test actual API calls without real keys, but we can test
       # that the initialization code handles the input correctly
   ```

2. **Property 2 Test: Invalid API key error handling**
   ```python
   # Feature: openai-client-initialization-fix, Property 2: Invalid API key produces error message
   @given(invalid_key=st.one_of(st.none(), st.just(""), st.text(max_size=5)))
   @settings(max_examples=100)
   def test_invalid_api_key_error_handling(invalid_key):
       # Test that invalid keys produce appropriate error messages
   ```

3. **Property 3 Test: Error logging**
   ```python
   # Feature: openai-client-initialization-fix, Property 3: Error logging completeness
   @given(error_scenario=st.sampled_from(['missing_key', 'invalid_key', 'network_error']))
   @settings(max_examples=100)
   def test_error_logging(error_scenario):
       # Test that all error scenarios result in appropriate logging
   ```

### Testing Dependencies

Add to requirements.txt:
```
pytest>=7.4.0
pytest-cov>=4.1.0
hypothesis>=6.92.0
```

## Implementation Notes

### Dependency Updates

The primary fix requires updating `requirements.txt`:

```txt
# Core dependencies
openai>=1.50.0  # Updated from 1.42.0 for better Python 3.13 support
streamlit==1.38.0
python-dotenv==1.0.1
pydantic==2.8.2
requests==2.32.3
python-docx==1.1.2

# Explicitly add httpx (required by openai SDK)
httpx>=0.24.0

# Testing dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
hypothesis>=6.92.0
```

### Code Changes

The `_load_openai_client` function needs enhanced error handling:

1. Add API key format validation
2. Add try-except blocks for different error types
3. Provide user-friendly error messages in Traditional Chinese
4. Log detailed error information for debugging

### Deployment Considerations

- Ensure Streamlit Cloud environment has all dependencies installed
- Verify Python 3.13 compatibility of all packages
- Test with actual OpenAI API keys in staging environment
- Monitor error logs for any initialization failures in production
