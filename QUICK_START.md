# Quick Start Guide

## Prerequisites
- Node.js 18+
- Python 3.8+
- Two terminal windows (one for frontend, one for backend)

## Running the Application

### Step 1: Start the Backend

```bash
# Terminal 1
cd d:\InsuranceWala\backend
python -m uvicorn app.api.predict:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
⚠ Invalid next.config.js options detected (can ignore)
✓ Ready in 1757ms
```

### Step 2: Start the Frontend

```bash
# Terminal 2
cd d:\InsuranceWala\frontend
npm run dev
```

**Expected Output:**
```
   ▲ Next.js 14.0.4
   - Local:        http://localhost:3000
   ✓ Ready in 1757ms
```

### Step 3: Access the Application

Open your browser and navigate to: **http://localhost:3000**

## What You Can Do

### 🏠 Home Page
- View project overview
- Understand key features
- Navigate to other pages
- See how-it-works section

### 🔮 Prediction Page
- Fill in insurance claim details
- Get instant AI prediction (Approved/Rejected)
- See confidence percentage
- View top factors influencing decision
- Make multiple predictions

### 📊 Dashboard
- View model performance metrics
  - Accuracy: Overall correctness
  - Precision: Positive prediction accuracy
  - Recall: True positive detection rate
  - F1-Score: Balanced metric
- See confusion matrix
- Analyze feature importance
- Review training details

### 🧠 Explainability Panel
- Understand global feature importance
- Learn about SHAP explanations
- Access educational content
- Navigate to local explanations

## Testing the System

### Option 1: Test with Sample Data

1. Go to http://localhost:3000/predict
2. Fill in any claim details (or try the example below)
3. Click "Get Prediction"
4. See instant results with explanations

### Sample Test Data
```
Months as Customer: 12
Age: 35
Policy Deductible: 1000
Policy Annual Premium: 1500.50
Umbrella Limit: 5000
Insured ZIP: 12345
Capital Gains: 0
Capital Loss: 0
Incident Hour: 14
Vehicles Involved: 2
Bodily Injuries: 1
Witnesses: 2
Total Claim Amount: 25000
... (fill other fields as needed)
```

### Option 2: Check Backend Health

```bash
# In another terminal
curl http://localhost:8000/api/health
```

Response:
```json
{"status": "ok"}
```

### Option 3: View API Documentation

Navigate to: **http://localhost:8000/docs**

This shows all available endpoints with interactive testing.

## Troubleshooting

### Issue: "Cannot connect to backend"
**Solution:**
1. Check backend terminal - should show "Uvicorn running"
2. Verify port 8000 is not in use: `netstat -ano | findstr :8000`
3. Restart backend server

### Issue: "Module not found" on backend
**Solution:**
1. Ensure you're in the backend directory
2. Install dependencies: `pip install -r requirements.txt`
3. Restart backend

### Issue: Next.js showing warnings
**Solution:**
1. This is normal and can be ignored
2. Frontend still works properly
3. Just wait for "Ready" message

### Issue: Form submission does nothing
**Solution:**
1. Check browser console (F12 → Console tab)
2. Verify backend is running
3. Check for network errors in DevTools (F12 → Network tab)

## File Structure

```
d:\InsuranceWala\
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root layout
│   │   ├── page.tsx         # Home page
│   │   ├── globals.css      # Global styles
│   │   ├── predict/
│   │   │   └── page.tsx     # Prediction page
│   │   ├── dashboard/
│   │   │   └── page.tsx     # Dashboard page
│   │   └── explain/
│   │       └── page.tsx     # Explanation page
│   ├── lib/
│   │   └── api.ts           # API client
│   ├── .env.local           # Environment config
│   ├── package.json
│   ├── next.config.js
│   └── tsconfig.json
│
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app with CORS
│   │   ├── api/
│   │   │   └── predict.py   # API endpoints
│   │   └── core/
│   │       ├── train.py     # Model training
│   │       ├── evaluate.py  # Model evaluation
│   │       └── explain.py   # SHAP explanations
│   ├── requirements.txt
│   └── data/                # Dataset storage
│
├── README.md
└── API_INTEGRATION.md
```

## Key Technologies

- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, scikit-learn, SHAP, pandas
- **ML Models**: Logistic Regression + Random Forest
- **Explainability**: SHAP (SHapley Additive exPlanations)

## Common Commands

```bash
# Backend
cd backend
pip install -r requirements.txt     # Install dependencies
python -m uvicorn app.api.predict:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install                          # Install dependencies
npm run dev                          # Start dev server
npm run build                        # Build for production
npm run lint                         # Run linter
```

## Next Steps

1. ✅ Both servers running
2. ✅ Frontend accessible at http://localhost:3000
3. ✅ Backend API at http://localhost:8000
4. 📊 Make a prediction on the Prediction page
5. 📈 Check Dashboard for model metrics
6. 🧠 Explore explanations on the Explain page

## Performance Tips

- **First load**: Models are lazy-loaded on first prediction
- **Subsequent predictions**: Will be faster after initial load
- **Data caching**: Dashboard caches metrics after first load
- **Network**: Use localhost for fastest performance

## Support & Debugging

1. Check `/memories/session/` for session notes
2. Review API_INTEGRATION.md for detailed documentation
3. Check browser console for frontend errors
4. Check backend terminal for server errors
5. Use DevTools Network tab to inspect API calls

---

**Happy prediction making! 🚀**

For detailed integration information, see `API_INTEGRATION.md`