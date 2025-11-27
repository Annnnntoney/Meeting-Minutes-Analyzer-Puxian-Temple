# Implementation Plan

- [x] 1. Update dependency configuration
  - Update requirements.txt to include httpx and updated OpenAI SDK version
  - Ensure all dependencies are compatible with Python 3.13
  - _Requirements: 3.2, 3.3_

- [ ] 2. Implement API key validation
- [x] 2.1 Create API key validation function
  - Write `_validate_api_key()` function to check API key format
  - Validate that key is not None, empty, or whitespace-only
  - Optionally check for "sk-" prefix pattern
  - _Requirements: 1.2_

- [ ]* 2.2 Write property test for API key validation
  - **Property 2: Invalid API key produces error message**
  - **Validates: Requirements 1.2**

- [ ] 3. Enhance OpenAI client initialization with error handling
- [x] 3.1 Add comprehensive error handling to `_load_openai_client()`
  - Add try-except blocks for ValueError, ImportError, and general exceptions
  - Implement user-friendly error messages in Traditional Chinese
  - Add error logging for debugging purposes
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4_

- [ ]* 3.2 Write property test for client initialization
  - **Property 1: Valid API key initialization succeeds**
  - **Validates: Requirements 1.1**

- [ ]* 3.3 Write property test for error logging
  - **Property 3: Error logging completeness**
  - **Validates: Requirements 2.4**

- [ ]* 3.4 Write unit tests for error handling scenarios
  - Test missing API key error message
  - Test invalid credentials error message
  - Test network error handling
  - Test missing dependencies error message
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 4. Test API key retrieval from multiple sources
- [x] 4.1 Verify API key retrieval from Streamlit secrets
  - Ensure `st.secrets["OPENAI_API_KEY"]` is properly handled
  - Handle FileNotFoundError and RuntimeError gracefully
  - _Requirements: 1.3_

- [x] 4.2 Verify API key retrieval from environment variables
  - Ensure `os.getenv("OPENAI_API_KEY")` works correctly
  - Test precedence: explicit key > secrets > environment
  - _Requirements: 1.4_

- [x] 4.3 Verify API key storage from sidebar input
  - Ensure sidebar input is stored in `st.session_state`
  - Verify key is also set in `os.environ` for consistency
  - _Requirements: 1.5_

- [ ]* 4.4 Write integration tests for API key sources
  - Test retrieval from Streamlit secrets
  - Test retrieval from environment variables
  - Test retrieval from sidebar input
  - Test precedence order
  - _Requirements: 1.3, 1.4, 1.5_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
