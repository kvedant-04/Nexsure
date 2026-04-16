# API Integration Guide

## Overview

The frontend and backend are now fully integrated with proper error handling, loading states, and real-time data fetching using a centralized API client.

## Architecture

### API Client (`lib/api.ts`)

A reusable API utility module that handles:
- **Request/Response Management**: Centralized API calls with consistent error handling
- **Type Safety**: TypeScript interfaces for all API responses
- **Error Handling**: Custom `APIError` class with detailed error messages
- **Configuration**: Environment-based API URL configuration

### Key Features

1. **Error Classes**
   ```typescript
   // Custom error with status code and details
   throw new APIError(statusCode, details, message)
   ```

2. **API Methods**
   ```typescript
   api.health()        // Check server status
   api.train()         // Train ML model
   api.predict()       // Make prediction
   api.getMetrics()    // Get model metrics
   api.getExplanation()// Get SHAP explanations
   ```

3. **Error Formatting**
   ```typescript
   formatErrorMessage(error) // User-friendly error display
   ```

## API Endpoints

### Base URL
```
http://localhost:8000/api
```

### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check API health |
| POST | `/train` | Train ML models |
| POST | `/predict` | Get claim prediction |
| GET | `/metrics` | Get model metrics |
| POST | `/explain` | Get SHAP explanations |

## Backend Configuration

### CORS Setup
The backend includes CORS middleware configured to accept requests from:
- `http://localhost:3000` (Next.js dev server)
- `http://localhost:3001` (Alternative port)
- `http://127.0.0.1:3000`

### Configuration File: `app/main.py`
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", ...],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Frontend Pages Integration

### 1. Prediction Page (`app/predict/page.tsx`)
**Features:**
- Form submission with comprehensive validation
- Real-time loading spinner during prediction
- Error alerts with detailed messages
- Confidence probability visualization
- Top 5 influencing features
- Smooth animations and transitions

**API Call:**
```typescript
const response = await api.predict(features)
```

**Response Handling:**
- Display prediction (Approved/Rejected)
- Show confidence percentage with progress bar
- Render feature importance bars
- Allow new predictions

### 2. Dashboard Page (`app/dashboard/page.tsx`)
**Features:**
- Automatic data fetching on page load
- Performance metrics display (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix visualization
- Global feature importance chart
- Helpful tooltips and descriptions

**API Calls:**
```typescript
const metrics = await api.getMetrics()
const explanation = await api.getExplanation({})
```

**Real-Time Services:**
- Progress loading state
- Graceful error recovery
- Informative metric cards with hover effects

### 3. Explanation Panel (`app/explain/page.tsx`)
**Features:**
- Global feature importance loading
- Detailed SHAP explanation
- Interactive feature visualization
- Navigation to local explanations
- Educational content

**API Call:**
```typescript
const explanation = await api.getExplanation({})
```

## Environment Configuration

### Frontend `.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

The `NEXT_PUBLIC_` prefix makes it available in the browser.

### Default Fallback
If not specified, defaults to: `http://localhost:8000/api`

## Error Handling Strategy

### Error Types

1. **Network Errors**
   - Connection refused
   - Backend not running
   - Network timeout
   
   Display: "Network error: Unable to connect to the backend..."

2. **API Errors**
   - Invalid request
   - Server validation failure
   - Business logic errors
   
   Display: API-specific error message

3. **Unknown Errors**
   - Unexpected exceptions
   - Parse failures
   
   Display: "An unexpected error occurred"

### User Feedback

Every API call provides:
- Loading state indicator (spinner)
- Error messages with recovery suggestions
- Success confirmations with data display
- Clear call-to-action buttons

## Loading States

### Implementation Pattern

1. **Form Submission**
   ```typescript
   setLoading(true)
   try {
     const result = await api.predict(data)
     setResult(result)
   } catch (err) {
     setError(formatErrorMessage(err))
   } finally {
     setLoading(false)
   }
   ```

2. **Page Load**
   ```typescript
   useEffect(() => {
     setLoading(true)
     fetchData().finally(() => setLoading(false))
   }, [])
   ```

3. **Visual Indicators**
   - Animated spinner during loading
   - Disabled form buttons while loading
   - "Loading..." text updates
   - Skeleton states (optional)

## Testing the Integration

### 1. Health Check
```bash
curl http://localhost:8000/api/health
# Response: {"status": "ok"}
```

### 2. Train Model
```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_filename": "insurance_claims.csv",
    "target_column": "fraud",
    "test_size": 0.2
  }'
```

### 3. Make Prediction
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35,
      "policy_deductable": 1000,
      ...
    },
    "model_name": "best_model"
  }'
```

## Best Practices

### 1. Error Recovery
- Provide clear error messages
- Suggest next steps
- Allow retry actions
- Log errors for debugging

### 2. Loading States
- Show spinners early
- Disable form submissions
- Display estimation time if long operations

### 3. Real-Time Updates
- Use `useEffect` for data loading
- Implement proper cleanup
- Handle race conditions
- Cache when appropriate

### 4. Type Safety
- Use TypeScript interfaces
- Validate API responses
- Handle optional fields
- Document response shapes

### 5. Performance
- Minimize API calls
- Use loading states efficiently
- Cache responses when appropriate
- Optimize network requests

## Troubleshooting

### Backend Not Running
**Error:** "Network error: Unable to connect to the backend"
**Solution:** Start backend with `python -m uvicorn app.api.predict:app --host 0.0.0.0 --port 8000`

### CORS Errors
**Error:** "No 'Access-Control-Allow-Origin' header"
**Solution:** Ensure CORS middleware is configured in `app/main.py`

### Wrong API URL
**Error:** 404 Not Found
**Solution:** Check `.env.local` and API endpoints use `/api` prefix

### Model Not Trained
**Error:** "Model not found"
**Solution:** Call `/api/train` endpoint before `/api/predict`

## Next Steps

1. Verify both servers are running
2. Test predictions on the Prediction page
3. View performance on the Dashboard
4. Explore explainability on the Explain page
5. Monitor console for any API errors

## Debugging

Enable detailed logging by:
1. Check browser console (Ctrl+Shift+J)
2. Check backend console output
3. Use Network tab in DevTools
4. Add `console.error()` in catch blocks

---

**Integration Status: ✅ Complete**
All pages are connected to the backend with proper error handling and loading states.