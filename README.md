# Smart Expense Tracker with Data Visualization (Multi-User)

A Python + Flask college project featuring automated AI category classification (Naive Bayes), data visualization (Matplotlib), spending trend prediction (Linear Regression), proactive budget alerts, and **multi-user private data isolation**.

---

## 📌 Problem Statement

In today's fast-paced life, students, working professionals, and families often find it difficult to keep track of daily spending. Most people record expenses in notebooks, memorize them, or don't record them at all. This leads to:
- **Lack of awareness:** People do not know which categories (Food, Shopping, Transport, etc.) consume most of their income.
- **Overspending:** Without proactive budget checks, spending frequently crosses the monthly limit before being noticed.
- **No planning:** Since past spending is not systematically analyzed, it is hard to accurately estimate next month's requirement.
- **Privacy & multi-user needs:** Different members of a household or group of roommates need their own separate accounts and budgets without seeing or overwriting each other's expenses.

---

## 💡 Key Features

1. **👥 Multi-User Accounts & Complete Data Isolation:**
   - Anyone can register their own account with a username and password.
   - Each user gets their own private expenses (`user_data/<username>/expenses.csv`), private budget targets (`user_data/<username>/budget.txt`), and personalized Matplotlib dashboard charts (`static/charts/dashboard_<username>.png`).
   - User A cannot view, modify, or delete User B's transactions.
   - Includes a one-click **Instant Demo Login** (`demo` / `demo123`) for effortless testing.
2. **🤖 AI Feature 1: Naive Bayes Auto-Categorization (`CountVectorizer` + `MultinomialNB`):**
   - Automatically suggests the category from the expense description in real-time as the user types (via live AJAX `/suggest-category`).
3. **📈 AI Feature 2: Linear Regression Spending Prediction:**
   - Evaluates the logged-in user's past completed monthly spending trends (ignoring the current incomplete month) to forecast next month's total expenditure.
4. **📊 Matplotlib Visual Analytics Dashboard:**
   - **Pie Chart:** Spending breakdown by category in soft pastel colors.
   - **Bar Chart:** Monthly expenditure totals over time with the peak month highlighted.
   - **Line Chart:** Recent daily spending pattern and trend line.
5. **⚠️ Proactive Budget Alerts:**
   - Triggers a **warning alert (80% reached)** when spending hits 80% of the monthly target.
   - Triggers an **over-budget alert (100% exceeded)** when the limit is breached.
6. **📥 Data Export:**
   - Each user can export their personal transactions to CSV with one click (`/export`).
7. **🎲 One-Click Demo Data Generator (`/load-sample`):**
   - Loads 6 months of realistic sample data exclusively into the active user's account for immediate demonstration of all graphs and ML models.

---

## 📁 Project Structure

```
smart-expense-tracker/
├── app.py                     # Flask backend with multi-user auth & ML routes
├── users.csv                  # User credentials (username, password_hash, created_at)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & problem statement
├── start_tunnel.ps1           # Deployment launcher script
├── user_data/                 # Isolated per-user data folders
│   ├── demo/
│   │   ├── expenses.csv       # Demo user's private expenses
│   │   └── budget.txt         # Demo user's monthly budget
│   └── <username>/
│       ├── expenses.csv       # User-specific private expenses
│       └── budget.txt         # User-specific monthly budget
├── static/
│   ├── style.css              # Custom soft pastel CSS stylesheet
│   └── charts/
│       ├── dashboard_demo.png # Demo user's generated chart
│       └── dashboard_<user>.png
└── templates/
    ├── login.html             # User Sign In & Instant Demo Login
    ├── register.html          # New account registration
    ├── index.html             # User's personal dashboard (KPIs, charts, form)
    ├── expenses.html          # All expenses table view (newest first)
    ├── add.html               # Dedicated expense recording form
    ├── budget.html            # Monthly budget configuration
    └── predict.html           # Linear Regression prediction breakdown
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ installed
- `pip` package manager

### 1. Run the Application
```bash
cd smart-expense-tracker
python app.py
```

### 2. Access in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Login or Register
- Sign in with the default demo account:  
  **Username:** `demo` | **Password:** `demo123`
- Or click **Create an Account** to register your own private username. Every account is completely separate!
