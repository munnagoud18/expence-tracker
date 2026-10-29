"""
=============================================================================
SMART EXPENSE TRACKER - MULTI-USER FLASK BACKEND (app.py)
A Python college project with Multi-User Data Isolation, AI Categorization,
Data Visualization & Spending Trend Prediction.

Tech Stack: Python, Flask, pandas, matplotlib, scikit-learn
Storage:
  - users.csv (username, password_hash, created_at)
  - user_data/<username>/expenses.csv (date, description, amount, category)
  - user_data/<username>/budget.txt
  - static/charts/dashboard_<username>.png
=============================================================================
"""

import os
import csv
import time
import math
import random
import re
import hashlib
from datetime import datetime, date, timedelta
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file, session, make_response
)

# ---------------------------------------------------------------------------
# Attempt Imports for Data Science & ML Packages with Fallbacks
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend for web servers
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# App Initialization & Path Configurations
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "smart-expense-tracker-multiuser-secret-key-2026"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.csv")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

# Standard Expense Categories
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"]

# Pastel Colors Matching style.css (:root palette)
PASTEL_COLORS = {
    "Food": "#99F6E4",           # Soft teal
    "Transport": "#FDE68A",      # Soft yellow
    "Shopping": "#A5B4FC",       # Soft indigo
    "Bills": "#FED7AA",         # Soft light orange/amber
    "Entertainment": "#E9D5FF",  # Soft purple
    "Other": "#CBD5E1"           # Soft slate grey
}


# ===========================================================================
# 1. USER AUTHENTICATION & MULTI-USER ISOLATION HELPERS
# ===========================================================================
def hash_password(password: str) -> str:
    """Hash password using SHA-256 for secure local storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_storage():
    """Ensure system directories and users file exist."""
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password_hash", "created_at"])

    # Seed default 'demo' account if it doesn't exist
    users = _read_users_raw()
    if "demo" not in users:
        _register_user_internal("demo", "demo123")
        # Migrate any existing legacy expenses to demo account
        legacy_csv = os.path.join(BASE_DIR, "expenses.csv")
        legacy_budget = os.path.join(BASE_DIR, "budget.txt")
        demo_expenses = get_user_expenses_file("demo")
        demo_budget = get_user_budget_file("demo")

        if os.path.exists(legacy_csv) and os.path.getsize(legacy_csv) > 30:
            try:
                import shutil
                shutil.copyfile(legacy_csv, demo_expenses)
            except Exception:
                pass

        if os.path.exists(legacy_budget):
            try:
                import shutil
                shutil.copyfile(legacy_budget, demo_budget)
            except Exception:
                pass


def _read_users_raw() -> dict:
    """Internal helper to read users.csv without recursion."""
    users = {}
    if not os.path.exists(USERS_FILE):
        return users
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row.get("username", "").strip().lower()
                p = row.get("password_hash", "").strip()
                if u and p:
                    users[u] = p
    except Exception:
        pass
    return users


def load_all_users() -> dict:
    """Load users from users.csv into a dictionary {username: password_hash}."""
    if not os.path.exists(USERS_FILE):
        init_storage()
    return _read_users_raw()


def _register_user_internal(clean_user: str, password: str):
    """Write user directly to users.csv."""
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([clean_user, hash_password(password), datetime.now().isoformat()])
    get_user_expenses_file(clean_user)
    get_user_budget_file(clean_user)
    generate_user_dashboard_charts(clean_user)


def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user and initialize their private data directory."""
    clean_user = username.strip().lower()
    if not re.match(r"^[a-zA-Z0-9_]{3,20}$", clean_user):
        return False, "Username must be 3-20 characters long and contain only letters, numbers, or underscores."

    if len(password) < 4:
        return False, "Password must be at least 4 characters long."

    users = load_all_users()
    if clean_user in users:
        return False, f"Username '{clean_user}' is already taken. Please choose another."

    # Write to users.csv
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([clean_user, hash_password(password), datetime.now().isoformat()])

    # Initialize user's private data storage
    get_user_expenses_file(clean_user)
    get_user_budget_file(clean_user)
    generate_user_dashboard_charts(clean_user)

    return True, "Account created successfully!"


def verify_user(username: str, password: str) -> bool:
    """Validate username and password against users.csv."""
    users = load_all_users()
    clean_user = username.strip().lower()
    return clean_user in users and users[clean_user] == hash_password(password)


def login_required(f):
    """Decorator to require login for user-specific routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            flash("Please sign in to access your personal expenses.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_current_user():
    """Make current_user available to all Jinja2 templates automatically."""
    return dict(current_user=session.get("username"))


# ---------------------------------------------------------------------------
# User Data Paths
# ---------------------------------------------------------------------------
def get_user_folder(username: str) -> str:
    """Return user's isolated data folder (sanitized against traversal)."""
    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "", username).lower()
    folder = os.path.join(USER_DATA_DIR, safe_user)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_user_expenses_file(username: str) -> str:
    """Return path to user's private expenses.csv, creating header if absent."""
    folder = get_user_folder(username)
    file_path = os.path.join(folder, "expenses.csv")
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "description", "amount", "category"])
    return file_path


def get_user_budget_file(username: str) -> str:
    """Return path to user's private budget.txt, initializing default 20000.0."""
    folder = get_user_folder(username)
    file_path = os.path.join(folder, "budget.txt")
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("20000.0")
    return file_path


def get_user_chart_path(username: str) -> str:
    """Return path to user's private dashboard image in static/charts/."""
    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "", username).lower()
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, f"dashboard_{safe_user}.png")


def get_user_budget(username: str) -> float:
    """Read user's monthly budget from their private budget.txt."""
    file_path = get_user_budget_file(username)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        pass
    return 20000.0


def set_user_budget(username: str, amount: float):
    """Save user's monthly budget to their private budget.txt."""
    file_path = get_user_budget_file(username)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{amount:.2f}")


def load_user_expenses(username: str):
    """Load expenses belonging exclusively to the specified user."""
    file_path = get_user_expenses_file(username)
    expenses = []
    if not os.path.exists(file_path):
        return expenses

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                amt = float(row.get("amount", 0))
                d_str = row.get("date", "").strip()
                desc = row.get("description", "").strip()
                cat = row.get("category", "Other").strip()
                expenses.append({
                    "id": idx,
                    "date": d_str,
                    "description": desc,
                    "amount": amt,
                    "category": cat
                })
            except Exception:
                continue
    return expenses


def save_user_expense(username: str, date_str: str, desc: str, amount: float, category: str):
    """Append a new expense row to the user's private expenses.csv."""
    file_path = get_user_expenses_file(username)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([date_str, desc, f"{amount:.2f}", category])


def delete_user_expense(username: str, expense_id: int) -> bool:
    """Delete an expense record by index from user's private expenses.csv."""
    expenses = load_user_expenses(username)
    if 0 <= expense_id < len(expenses):
        del expenses[expense_id]
        file_path = get_user_expenses_file(username)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "description", "amount", "category"])
            for item in expenses:
                writer.writerow([item["date"], item["description"], f"{item['amount']:.2f}", item["category"]])
        return True
    return False


# ===========================================================================
# 2. AI FEATURE 1: NAIVE BAYES AUTO-CATEGORIZATION
# ===========================================================================
SAMPLE_TRAINING_DATA = [
    # Food
    ("lunch at subway restaurant", "Food"),
    ("pizza dinner with friends", "Food"),
    ("mcdonalds burger and fries", "Food"),
    ("morning coffee and pastry at starbucks", "Food"),
    ("grocery shopping at supermarket vegetables fruits milk", "Food"),
    ("zomato swiggy food delivery order", "Food"),
    ("dinner buffet restaurant bill", "Food"),
    ("snacks chips tea and biscuits", "Food"),
    ("bakery bread cake cookies", "Food"),
    ("ice cream parlor sundae dessert", "Food"),
    ("biryani lunch with classmates", "Food"),

    # Transport
    ("uber cab ride to campus", "Transport"),
    ("uber cab ride to airport", "Transport"),
    ("ola auto ride to metro station", "Transport"),
    ("metro card smart recharge", "Transport"),
    ("petrol fuel refill at gas station", "Transport"),
    ("diesel for car", "Transport"),
    ("train ticket booking irctc", "Transport"),
    ("bus ticket fare city commute", "Transport"),
    ("parking fee parking lot ticket", "Transport"),
    ("highway toll plaza tax fastag recharge", "Transport"),
    ("flight ticket domestic booking", "Transport"),

    # Shopping
    ("bought new shirt and jeans at mall", "Shopping"),
    ("amazon online order sneakers shoes", "Shopping"),
    ("flipkart sale electronics wireless mouse", "Shopping"),
    ("cotton t-shirt hoodie jacket myntra", "Shopping"),
    ("college study textbooks and notebook", "Shopping"),
    ("leather wallet belt wristwatch", "Shopping"),
    ("sunglasses and backpack luggage bag", "Shopping"),
    ("running shoes sports gear gym gloves", "Shopping"),

    # Bills & Utilities
    ("electricity light power bill payment", "Bills"),
    ("broadband wifi internet monthly recharge", "Bills"),
    ("mobile phone postpaid plan jio airtel", "Bills"),
    ("apartment monthly flat rent payment", "Bills"),
    ("lpg gas cylinder booking refill", "Bills"),
    ("water bill municipal tax", "Bills"),
    ("monthly building society maintenance charges", "Bills"),
    ("netflix streaming subscription monthly", "Bills"),
    ("spotify premium music subscription", "Bills"),
    ("gym annual membership fee", "Bills"),

    # Entertainment
    ("movie tickets cinema popcorn multiplex", "Entertainment"),
    ("concert music festival live show pass", "Entertainment"),
    ("video game purchase steam playstation", "Entertainment"),
    ("bowling alley arcade gaming arena", "Entertainment"),
    ("weekend outing amusement park resort", "Entertainment"),
    ("night club party drinks and dance", "Entertainment"),
    ("theatre drama comedy show entry ticket", "Entertainment"),

    # Other
    ("medical pharmacy cough syrup medicine tablets", "Other"),
    ("doctor clinic consultation fees", "Other"),
    ("dentist dental cleanup checkup", "Other"),
    ("charity donation non-profit ngo contribution", "Other"),
    ("birthday gift wrapping paper for sister", "Other"),
    ("barber salon haircut and grooming", "Other"),
    ("laundry dry clean clothes washing service", "Other"),
    ("laptop screen repair mobile tempered glass", "Other"),
    ("pen pencil diary stationary supplies", "Other")
]


class PurePythonNaiveBayes:
    """Pure Python Naive Bayes classifier with Laplace smoothing."""
    def __init__(self):
        self.class_word_counts = {cat: {} for cat in CATEGORIES}
        self.class_totals = {cat: 0 for cat in CATEGORIES}
        self.class_doc_counts = {cat: 0 for cat in CATEGORIES}
        self.vocabulary = set()
        self.total_docs = 0

    def _tokenize(self, text):
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def train(self, data):
        self.total_docs = len(data)
        for desc, cat in data:
            if cat not in self.class_word_counts:
                continue
            self.class_doc_counts[cat] += 1
            words = self._tokenize(desc)
            for w in words:
                self.vocabulary.add(w)
                self.class_word_counts[cat][w] = self.class_word_counts[cat].get(w, 0) + 1
                self.class_totals[cat] += 1

    def predict(self, text):
        words = self._tokenize(text)
        if not words:
            return "Other"

        vocab_size = max(len(self.vocabulary), 1)
        best_cat = "Other"
        best_log_prob = -float("inf")

        for cat in CATEGORIES:
            doc_count = self.class_doc_counts.get(cat, 0)
            prior = math.log((doc_count + 1) / (self.total_docs + len(CATEGORIES)))
            cat_total_words = self.class_totals.get(cat, 0)

            likelihood = 0.0
            for w in words:
                w_count = self.class_word_counts[cat].get(w, 0)
                word_prob = (w_count + 1) / (cat_total_words + vocab_size)
                likelihood += math.log(word_prob)

            total_prob = prior + likelihood
            if total_prob > best_log_prob:
                best_log_prob = total_prob
                best_cat = cat

        return best_cat


pure_nb = PurePythonNaiveBayes()
pure_nb.train(SAMPLE_TRAINING_DATA)

nb_classifier = None
vectorizer = None
if HAS_SKLEARN:
    try:
        texts = [item[0] for item in SAMPLE_TRAINING_DATA]
        labels = [item[1] for item in SAMPLE_TRAINING_DATA]
        vectorizer = CountVectorizer(stop_words="english")
        X_train = vectorizer.fit_transform(texts)
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, labels)
    except Exception:
        nb_classifier = None


def predict_category(description: str) -> str:
    """Predict category from description using Naive Bayes."""
    if not description or not description.strip():
        return "Other"
    clean_desc = description.strip()

    if HAS_SKLEARN and nb_classifier is not None and vectorizer is not None:
        try:
            vec = vectorizer.transform([clean_desc])
            return str(nb_classifier.predict(vec)[0])
        except Exception:
            pass

    return pure_nb.predict(clean_desc)


# ===========================================================================
# 3. AI FEATURE 2: LINEAR REGRESSION SPENDING PREDICTION (PER-USER)
# ===========================================================================
def predict_user_next_month_spending(username: str):
    """
    Runs Linear Regression on user's completed past monthly totals.
    Ignores the current incomplete month.
    """
    expenses = load_user_expenses(username)
    if not expenses:
        return None, "No expense records found. Add expenses or load sample data to see predictions.", []

    today = date.today()
    current_month_str = today.strftime("%Y-%m")

    # Group expenses by year-month
    monthly_totals = {}
    for item in expenses:
        try:
            d = datetime.strptime(item["date"], "%Y-%m-%d").date()
            ym = d.strftime("%Y-%m")
            if ym < current_month_str:
                monthly_totals[ym] = monthly_totals.get(ym, 0.0) + item["amount"]
        except Exception:
            continue

    sorted_past_months = sorted(monthly_totals.keys())
    past_months_data = [{"month": m, "total": monthly_totals[m]} for m in sorted_past_months]
    k = len(sorted_past_months)

    if k < 3:
        msg = (
            f"Need at least 3 completed months of data to predict next month. "
            f"Currently have {k} completed month(s). Click 'Load Sample Data' in the menu to test."
        )
        return None, msg, past_months_data

    # Linear Regression: X = 0, 1, 2, ... k-1, y = totals
    y_vals = [monthly_totals[m] for m in sorted_past_months]
    predicted_val = None

    if HAS_SKLEARN:
        try:
            import numpy as np
            X = np.arange(k).reshape(-1, 1)
            y = np.array(y_vals)
            reg = LinearRegression()
            reg.fit(X, y)
            pred = float(reg.predict([[k]])[0])
            predicted_val = max(0.0, pred)
        except Exception:
            predicted_val = None

    # Fallback exact closed-form Ordinary Least Squares formula
    if predicted_val is None:
        n = k
        x_mean = (n - 1) / 2.0
        y_mean = sum(y_vals) / n
        numerator = sum((i - x_mean) * (y_vals[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0
        intercept = y_mean - slope * x_mean
        predicted_val = max(0.0, slope * k + intercept)

    msg = f"Based on {k} past completed months using Linear Regression."
    return round(predicted_val, 2), msg, past_months_data


# ===========================================================================
# 4. DATA VISUALIZATION: MATPLOTLIB DASHBOARD (PER-USER)
# ===========================================================================
def generate_user_dashboard_charts(username: str) -> str:
    """
    Generates an updated personalized 3-panel dashboard chart for the given user:
    1. Pie chart: Spending by Category with percentages and total sum
    2. Bar chart: Monthly spending totals with value labels on bars
    3. Line chart: Recent daily spending trend
    Saved to static/charts/dashboard_<username>.png and returns the file path.
    """
    chart_file = get_user_chart_path(username)
    expenses = load_user_expenses(username)

    if not HAS_MATPLOTLIB:
        return chart_file

    plt.close("all")
    fig = plt.figure(figsize=(15, 4.5), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")

    if not expenses:
        ax = fig.add_subplot(111)
        ax.set_facecolor("#F8FAFC")
        ax.text(
            0.5, 0.5,
            f"No Expenses Logged Yet for {username}\nAdd an expense or load sample data to generate your charts",
            ha="center", va="center", fontsize=13, color="#64748B", fontweight="500"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(chart_file, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return chart_file

    # Data aggregation
    category_totals = {c: 0.0 for c in CATEGORIES}
    monthly_totals = {}
    daily_totals = {}
    total_spent = 0.0

    for item in expenses:
        cat = item["category"] if item["category"] in category_totals else "Other"
        amt = item["amount"]
        category_totals[cat] = category_totals.get(cat, 0.0) + amt
        total_spent += amt

        d_str = item["date"]
        daily_totals[d_str] = daily_totals.get(d_str, 0.0) + amt
        ym = d_str[:7]
        monthly_totals[ym] = monthly_totals.get(ym, 0.0) + amt

    active_cats = {k: v for k, v in category_totals.items() if v > 0}
    if not active_cats:
        active_cats = {"No Data": 1}

    pie_labels = list(active_cats.keys())
    pie_values = list(active_cats.values())
    pie_colors = [PASTEL_COLORS.get(c, "#A5B4FC") for c in pie_labels]

    # --- Subplot 1: Pie Chart (Category Breakdown) ---
    ax1 = fig.add_subplot(131)
    ax1.set_facecolor("#FFFFFF")
    wedges, texts, autotexts = ax1.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.68, edgecolor="#FFFFFF", linewidth=2),
        textprops=dict(color="#334155", fontsize=8.5, fontweight="500")
    )
    for at in autotexts:
        at.set_color("#1E1B4B")
        at.set_fontsize(8)
        at.set_fontweight("600")
    ax1.set_title(f"Category Breakdown (Total: Rs. {total_spent:,.0f})", fontsize=11, fontweight="600", color="#334155", pad=10)

    # --- Subplot 2: Bar Chart (Monthly Totals) ---
    ax2 = fig.add_subplot(132)
    ax2.set_facecolor("#F8FAFC")
    sorted_months = sorted(monthly_totals.keys())[-6:]
    m_vals = [monthly_totals[m] for m in sorted_months]

    bars = ax2.bar(
        sorted_months, m_vals,
        color="#99F6E4", edgecolor="#5EEAD4", linewidth=1.5,
        width=0.55, zorder=3
    )
    if m_vals:
        max_idx = m_vals.index(max(m_vals))
        bars[max_idx].set_color("#A5B4FC")
        bars[max_idx].set_edgecolor("#818CF8")

    # Add numeric labels on top of bars
    if hasattr(ax2, "bar_label"):
        ax2.bar_label(bars, fmt="Rs.%.0f", padding=3, fontsize=7.5, color="#334155")
    ax2.margins(y=0.18)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)
    ax2.set_title("Monthly Spending Totals", fontsize=11, fontweight="600", color="#334155", pad=10)
    ax2.tick_params(axis="x", rotation=25, labelsize=8, colors="#64748B")
    ax2.tick_params(axis="y", labelsize=8, colors="#64748B")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#E2E8F0")
    ax2.spines["bottom"].set_color("#E2E8F0")

    # --- Subplot 3: Line Chart (Daily Spending Trend) ---
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor("#F8FAFC")
    sorted_dates = sorted(daily_totals.keys())[-14:]
    d_vals = [daily_totals[d] for d in sorted_dates]
    formatted_dates = [d[5:] for d in sorted_dates]

    if len(sorted_dates) == 1:
        ax3.scatter(formatted_dates, d_vals, color="#818CF8", s=80, zorder=4)
        ax3.annotate(
            f"Rs.{d_vals[0]:,.0f}",
            (0, d_vals[0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="#334155",
            fontweight="600"
        )
        ax3.set_ylim(0, max(d_vals[0] * 1.3, 1000))
    elif len(sorted_dates) > 1:
        ax3.plot(
            formatted_dates, d_vals,
            marker="o", markersize=5,
            color="#818CF8", linewidth=2.2,
            markerfacecolor="#A5B4FC", markeredgecolor="#4338CA",
            zorder=4
        )
        ax3.fill_between(formatted_dates, d_vals, color="#A5B4FC", alpha=0.25, zorder=3)
        ax3.set_ylim(bottom=0)
        ax3.margins(y=0.15)

    ax3.grid(axis="both", linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)
    ax3.set_title("Recent Daily Spending Trend", fontsize=11, fontweight="600", color="#334155", pad=10)
    ax3.tick_params(axis="x", rotation=35, labelsize=8, colors="#64748B")
    ax3.tick_params(axis="y", labelsize=8, colors="#64748B")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#E2E8F0")
    ax3.spines["bottom"].set_color("#E2E8F0")

    plt.tight_layout(pad=2.0)
    plt.savefig(chart_file, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return chart_file


# ===========================================================================
# 5. USER SUMMARY STATS CALCULATOR
# ===========================================================================
def calculate_user_summary_stats(username: str):
    """Calculates KPI summary cards for the user's current month."""
    expenses = load_user_expenses(username)
    budget = get_user_budget(username)
    today = date.today()
    current_ym = today.strftime("%Y-%m")

    this_month_spent = 0.0
    category_spent = {c: 0.0 for c in CATEGORIES}

    for item in expenses:
        amt = item["amount"]
        d_str = item["date"]
        if d_str.startswith(current_ym):
            this_month_spent += amt
            cat = item["category"] if item["category"] in category_spent else "Other"
            category_spent[cat] = category_spent.get(cat, 0.0) + amt

    top_cat = "None"
    top_cat_amt = 0.0
    if this_month_spent > 0:
        top_cat, top_cat_amt = max(category_spent.items(), key=lambda x: x[1])
    top_cat_pct = (top_cat_amt / this_month_spent * 100.0) if this_month_spent > 0 else 0.0

    category_breakdown = []
    if this_month_spent > 0:
        for cat in CATEGORIES:
            amt = category_spent.get(cat, 0.0)
            if amt > 0:
                pct = (amt / this_month_spent) * 100.0
                category_breakdown.append({
                    "category": cat,
                    "amount": amt,
                    "pct": round(pct, 1),
                    "color": PASTEL_COLORS.get(cat, "#A5B4FC")
                })
        category_breakdown.sort(key=lambda x: x["amount"], reverse=True)

    pct_budget = (this_month_spent / budget * 100.0) if budget > 0 else 0.0
    budget_left = budget - this_month_spent
    if this_month_spent > budget:
        budget_status_label = f"₹{abs(budget_left):,.2f} Over Limit"
        budget_status_sub = "Alert: Exceeded monthly limit"
        budget_card_class = "accent-danger"
    elif pct_budget >= 80.0:
        budget_status_label = f"₹{budget_left:,.2f} Left"
        budget_status_sub = f"Warning: {pct_budget:.1f}% used"
        budget_card_class = "accent-warning"
    else:
        budget_status_label = f"₹{budget_left:,.2f} Left"
        budget_status_sub = f"Healthy: {pct_budget:.1f}% used"
        budget_card_class = "accent-secondary"

    pred_val, pred_msg, _ = predict_user_next_month_spending(username)
    if pred_val is not None:
        prediction_val_str = f"₹{pred_val:,.2f}"
        prediction_sub_str = "Predicted via Linear Regression"
    else:
        prediction_val_str = "N/A"
        prediction_sub_str = "Need 3+ completed months"

    return {
        "budget": budget,
        "this_month_spent": this_month_spent,
        "pct_budget": pct_budget,
        "top_category": top_cat,
        "top_category_amount": top_cat_amt,
        "top_category_pct": top_cat_pct,
        "category_breakdown": category_breakdown,
        "budget_status_label": budget_status_label,
        "budget_status_sub": budget_status_sub,
        "budget_card_class": budget_card_class,
        "prediction_val_str": prediction_val_str,
        "prediction_sub_str": prediction_sub_str,
        "today_str": today.strftime("%Y-%m-%d")
    }


# ===========================================================================
# 6. AUTHENTICATION ROUTES
# ===========================================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    """User Sign In route."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if verify_user(username, password):
            clean_user = username.strip().lower()
            session["username"] = clean_user
            flash(f"Welcome back, {clean_user}!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        else:
            flash("Invalid username or password. Please try again.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """User Registration route."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if password != confirm:
            flash("Passwords do not match. Please re-enter.", "danger")
            return redirect(url_for("register"))

        success, msg = register_user(username, password)
        if success:
            session["username"] = username.strip().lower()
            flash(f"Account created successfully! Welcome, {username}!", "success")
            return redirect(url_for("index"))
        else:
            flash(msg, "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    """Clear session and sign user out."""
    user = session.pop("username", None)
    if user:
        flash(f"You have been signed out. See you soon, {user}!", "info")
    return redirect(url_for("login"))


# ===========================================================================
# 7. MAIN APPLICATION ROUTES (USER ISOLATED)
# ===========================================================================

@app.route("/dashboard-chart")
@login_required
def dashboard_chart():
    """Serve dynamic, real-time generated dashboard chart with no-cache headers."""
    user = session["username"]
    chart_path = generate_user_dashboard_charts(user)
    resp = make_response(send_file(chart_path, mimetype="image/png"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
@login_required
def index():
    """Personalized user dashboard view."""
    user = session["username"]
    generate_user_dashboard_charts(user)

    stats = calculate_user_summary_stats(user)
    all_expenses = load_user_expenses(user)
    sorted_expenses = sorted(all_expenses, key=lambda x: x["date"], reverse=True)
    recent_expenses = sorted_expenses[:5]

    return render_template(
        "index.html",
        stats=stats,
        recent_expenses=recent_expenses,
        categories=CATEGORIES,
        cache_buster=int(time.time() * 1000)
    )


@app.route("/add", methods=["GET", "POST"])
@login_required
def add_expense():
    """Add a new expense to the current user's personal records."""
    user = session["username"]

    if request.method == "GET":
        return render_template(
            "add.html",
            categories=CATEGORIES,
            today_str=date.today().strftime("%Y-%m-%d")
        )

    date_str = request.form.get("date", "").strip()
    description = request.form.get("description", "").strip()
    amount_str = request.form.get("amount", "").strip()
    category = request.form.get("category", "").strip()

    if not description:
        flash("Description cannot be empty. Please enter expense details.", "danger")
        return redirect(request.referrer or url_for("index"))

    try:
        amount = float(amount_str)
        if amount <= 0:
            flash("Amount must be a positive number greater than 0.", "danger")
            return redirect(request.referrer or url_for("index"))
    except (ValueError, TypeError):
        flash("Invalid amount entered. Please enter a valid numeric value.", "danger")
        return redirect(request.referrer or url_for("index"))

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        flash("Invalid date format. Please select a valid date (YYYY-MM-DD).", "danger")
        return redirect(request.referrer or url_for("index"))

    if category not in CATEGORIES:
        category = predict_category(description)

    # Save to user's private CSV and regenerate chart
    save_user_expense(user, date_str, description, amount, category)
    generate_user_dashboard_charts(user)

    # Budget alert check
    budget = get_user_budget(user)
    current_ym = date.today().strftime("%Y-%m")
    all_exp = load_user_expenses(user)
    this_month_spent = sum(
        item["amount"] for item in all_exp if item["date"].startswith(current_ym)
    )

    if this_month_spent > budget:
        over_by = this_month_spent - budget
        flash(
            f"🚨 Budget Exceeded! Monthly spending (₹{this_month_spent:,.2f}) has exceeded "
            f"your limit of ₹{budget:,.2f} by ₹{over_by:,.2f}!",
            "danger"
        )
    elif this_month_spent >= 0.8 * budget:
        pct = (this_month_spent / budget) * 100.0
        flash(
            f"⚠️ Budget Warning! You have reached {pct:.1f}% of your monthly limit "
            f"(₹{this_month_spent:,.2f} / ₹{budget:,.2f}).",
            "warning"
        )
    else:
        flash(f"Expense of ₹{amount:,.2f} added to '{category}' successfully!", "success")

    return redirect(url_for("index"))


@app.route("/expenses")
@login_required
def expenses_list():
    """Table of all expenses belonging exclusively to current user."""
    user = session["username"]
    all_exp = load_user_expenses(user)
    sorted_expenses = sorted(all_exp, key=lambda x: x["date"], reverse=True)
    total_amount = sum(item["amount"] for item in sorted_expenses)
    return render_template(
        "expenses.html",
        expenses=sorted_expenses,
        total_amount=total_amount,
        count=len(sorted_expenses)
    )


@app.route("/delete/<int:id>", methods=["POST", "GET"])
@login_required
def delete_expense(id):
    """Delete an expense record from current user's records."""
    user = session["username"]
    success = delete_user_expense(user, id)
    if success:
        generate_user_dashboard_charts(user)
        flash("Expense record deleted successfully.", "success")
    else:
        flash("Expense record could not be found or was already deleted.", "danger")
    return redirect(request.referrer or url_for("index"))


@app.route("/budget", methods=["GET", "POST"])
@login_required
def budget():
    """View and update current user's monthly budget limit."""
    user = session["username"]

    if request.method == "POST":
        budget_str = request.form.get("budget", "").strip()
        try:
            val = float(budget_str)
            if val <= 0:
                flash("Budget must be a positive number greater than 0.", "danger")
                return redirect(url_for("budget"))
            set_user_budget(user, val)
            flash(f"Monthly budget limit updated to ₹{val:,.2f}!", "success")
            return redirect(url_for("index"))
        except (ValueError, TypeError):
            flash("Please enter a valid numeric budget amount.", "danger")
            return redirect(url_for("budget"))

    current_budget = get_user_budget(user)
    return render_template("budget.html", budget=current_budget)


@app.route("/predict")
@login_required
def predict_view():
    """Detailed spending trend prediction page for current user."""
    user = session["username"]
    predicted_val, message, past_months = predict_user_next_month_spending(user)
    return render_template(
        "predict.html",
        predicted_val=predicted_val,
        message=message,
        past_months=past_months
    )


@app.route("/suggest-category")
def suggest_category():
    """
    Public AJAX endpoint: Suggest category from description using Naive Bayes.
    Live AJAX helper available to all users.
    """
    desc = request.args.get("description", "").strip()
    suggested = predict_category(desc)
    return jsonify({"category": suggested})


@app.route("/load-sample")
@login_required
def load_sample():
    """Loads 6 months of sample data into current user's private account."""
    user = session["username"]
    today = date.today()

    sample_templates = [
        ("Grocery shopping at supermarket", (800, 2500), "Food"),
        ("Domino's Pizza dinner with classmates", (450, 950), "Food"),
        ("Starbucks coffee & sandwich", (250, 600), "Food"),
        ("Subway lunch with team", (220, 500), "Food"),
        ("Metro smart card monthly recharge", (400, 800), "Transport"),
        ("Uber cab ride to railway station", (280, 650), "Transport"),
        ("Petrol fuel tank refill", (1200, 2200), "Transport"),
        ("Ola auto fare commute", (80, 200), "Transport"),
        ("College textbooks & stationery", (650, 1500), "Shopping"),
        ("Amazon electronics order", (999, 2800), "Shopping"),
        ("Clothing purchase at mall", (1100, 3200), "Shopping"),
        ("Broadband wifi high-speed internet", (699, 999), "Bills"),
        ("Mobile phone recharge postpaid", (399, 699), "Bills"),
        ("Electricity bill payment", (1200, 2400), "Bills"),
        ("Netflix monthly streaming subscription", (499, 649), "Bills"),
        ("Movie tickets IMAX multiplex", (400, 900), "Entertainment"),
        ("Bowling and arcade games outing", (500, 1200), "Entertainment"),
        ("Medical prescription pharmacy", (250, 850), "Other"),
        ("Haircut and grooming salon", (150, 450), "Other"),
    ]

    new_rows = []
    for month_offset in range(5, -1, -1):
        target_year = today.year
        target_month = today.month - month_offset
        while target_month <= 0:
            target_month += 12
            target_year -= 1

        max_day = 28
        if month_offset == 0:
            max_day = max(1, today.day)

        num_entries = random.randint(6, 8)
        for _ in range(num_entries):
            desc, amt_range, cat = random.choice(sample_templates)
            day = random.randint(1, max_day)
            dt_str = f"{target_year:04d}-{target_month:02d}-{day:02d}"
            amt = round(random.uniform(amt_range[0], amt_range[1]), 2)
            new_rows.append([dt_str, desc, f"{amt:.2f}", cat])

    new_rows.sort(key=lambda x: x[0])

    file_path = get_user_expenses_file(user)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "description", "amount", "category"])
        writer.writerows(new_rows)

    generate_user_dashboard_charts(user)
    flash(
        f"Loaded {len(new_rows)} sample expense records for {user} spanning the last 6 months! "
        f"Your personal charts and Linear Regression predictions are now fully populated.",
        "success"
    )
    return redirect(url_for("index"))


@app.route("/export")
@login_required
def export_csv():
    """Allows user to download their own private expenses.csv directly."""
    user = session["username"]
    file_path = get_user_expenses_file(user)
    if os.path.exists(file_path):
        return send_file(
            file_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"expenses_{user}.csv"
        )
    flash("No expenses recorded yet to export.", "warning")
    return redirect(url_for("index"))


# ===========================================================================
# 8. APPLICATION ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    init_storage()
    print("=" * 65)
    print(" Smart Expense Tracker (Multi-User) is running!")
    print(" Open in your browser: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False)
