/**
 * =============================================================================
 * SMART EXPENSE TRACKER - CLIENT-SIDE ENGINE (app.js)
 * Designed for GitHub Pages & Serverless Deployments.
 * Runs 100% in the browser with Zero Host / Server Required!
 *
 * Features:
 * 1. Naive Bayes Classifier in JS (AI Category Suggestion live as you type)
 * 2. Linear Regression Model in JS (Spending trend forecasting for next month)
 * 3. Real-Time Chart.js Visualizations (Pie, Monthly Bar, Daily Line charts)
 * 4. Multi-User Profile Isolation via LocalStorage
 * 5. Proactive Budget Alerts (80% Warning, 100% Danger)
 * 6. CSV Export & CSV Import
 * 7. One-Click 6-Month Realistic Sample Data Generator
 * =============================================================================
 */

// -----------------------------------------------------------------------------
// 1. CONSTANTS & PASTEL PALETTE
// -----------------------------------------------------------------------------
const CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"];

const PASTEL_COLORS = {
  "Food": "#A5B4FC",          // Soft indigo
  "Transport": "#99F6E4",     // Soft teal
  "Shopping": "#FDE68A",      // Soft yellow
  "Bills": "#BBF7D0",         // Soft green
  "Entertainment": "#FECACA", // Soft red
  "Other": "#E2E8F0"          // Soft slate grey
};

const PASTEL_BORDER_COLORS = {
  "Food": "#818CF8",
  "Transport": "#5EEAD4",
  "Shopping": "#FCD34D",
  "Bills": "#86EFAC",
  "Entertainment": "#FCA5A5",
  "Other": "#CBD5E1"
};

// -----------------------------------------------------------------------------
// 2. AI FEATURE 1: NAIVE BAYES TEXT CLASSIFIER (PURE JAVASCRIPT)
// -----------------------------------------------------------------------------
const SAMPLE_TRAINING_DATA = [
  // Food
  ["lunch at subway restaurant", "Food"],
  ["pizza dinner with friends", "Food"],
  ["mcdonalds burger and fries", "Food"],
  ["morning coffee and pastry at starbucks", "Food"],
  ["grocery shopping at supermarket vegetables fruits milk", "Food"],
  ["zomato swiggy food delivery order", "Food"],
  ["dinner buffet restaurant bill", "Food"],
  ["snacks chips tea and biscuits", "Food"],
  ["bakery bread cake cookies", "Food"],
  ["ice cream parlor sundae dessert", "Food"],
  ["biryani lunch with classmates", "Food"],

  // Transport
  ["uber cab ride to campus", "Transport"],
  ["uber cab ride to airport", "Transport"],
  ["ola auto ride to metro station", "Transport"],
  ["metro card smart recharge", "Transport"],
  ["petrol fuel refill at gas station", "Transport"],
  ["diesel for car", "Transport"],
  ["train ticket booking irctc", "Transport"],
  ["bus ticket fare city commute", "Transport"],
  ["parking fee parking lot ticket", "Transport"],
  ["highway toll plaza tax fastag recharge", "Transport"],
  ["flight ticket domestic booking", "Transport"],

  // Shopping
  ["bought new shirt and jeans at mall", "Shopping"],
  ["amazon online order sneakers shoes", "Shopping"],
  ["flipkart sale electronics wireless mouse", "Shopping"],
  ["cotton t-shirt hoodie jacket myntra", "Shopping"],
  ["college study textbooks and notebook", "Shopping"],
  ["leather wallet belt wristwatch", "Shopping"],
  ["sunglasses and backpack luggage bag", "Shopping"],
  ["running shoes sports gear gym gloves", "Shopping"],

  // Bills & Utilities
  ["electricity light power bill payment", "Bills"],
  ["broadband wifi internet monthly recharge", "Bills"],
  ["mobile phone postpaid plan jio airtel", "Bills"],
  ["apartment monthly flat rent payment", "Bills"],
  ["lpg gas cylinder booking refill", "Bills"],
  ["water bill municipal tax", "Bills"],
  ["monthly building society maintenance charges", "Bills"],
  ["netflix streaming subscription monthly", "Bills"],
  ["spotify premium music subscription", "Bills"],
  ["gym annual membership fee", "Bills"],

  // Entertainment
  ["movie tickets cinema popcorn multiplex", "Entertainment"],
  ["concert music festival live show pass", "Entertainment"],
  ["video game purchase steam playstation", "Entertainment"],
  ["bowling alley arcade gaming arena", "Entertainment"],
  ["weekend outing amusement park resort", "Entertainment"],
  ["night club party drinks and dance", "Entertainment"],
  ["theatre drama comedy show entry ticket", "Entertainment"],

  // Other
  ["medical pharmacy cough syrup medicine tablets", "Other"],
  ["doctor clinic consultation fees", "Other"],
  ["dentist dental cleanup checkup", "Other"],
  ["charity donation non-profit ngo contribution", "Other"],
  ["birthday gift wrapping paper for sister", "Other"],
  ["barber salon haircut and grooming", "Other"],
  ["laundry dry clean clothes washing service", "Other"],
  ["laptop screen repair mobile tempered glass", "Other"],
  ["pen pencil diary stationary supplies", "Other"]
];

class PureJSNaiveBayes {
  constructor() {
    this.classWordCounts = {};
    this.classTotals = {};
    this.classDocCounts = {};
    this.vocabulary = new Set();
    this.totalDocs = 0;

    CATEGORIES.forEach(cat => {
      this.classWordCounts[cat] = {};
      this.classTotals[cat] = 0;
      this.classDocCounts[cat] = 0;
    });
  }

  tokenize(text) {
    if (!text) return [];
    return (text.toLowerCase().match(/\b[a-z]{2,}\b/g) || []);
  }

  train(data) {
    this.totalDocs = data.length;
    data.forEach(([desc, cat]) => {
      if (!this.classDocCounts[cat] && this.classDocCounts[cat] !== 0) return;
      this.classDocCounts[cat]++;
      const words = this.tokenize(desc);
      words.forEach(w => {
        this.vocabulary.add(w);
        this.classWordCounts[cat][w] = (this.classWordCounts[cat][w] || 0) + 1;
        this.classTotals[cat]++;
      });
    });
  }

  predict(text) {
    const words = this.tokenize(text);
    if (!words.length) return "Other";

    const vocabSize = Math.max(this.vocabulary.size, 1);
    let bestCat = "Other";
    let bestLogProb = -Infinity;

    CATEGORIES.forEach(cat => {
      const docCount = this.classDocCounts[cat] || 0;
      const prior = Math.log((docCount + 1) / (this.totalDocs + CATEGORIES.length));

      const catTotalWords = this.classTotals[cat] || 0;
      let likelihood = 0;
      words.forEach(w => {
        const wCount = this.classWordCounts[cat][w] || 0;
        const wordProb = (wCount + 1) / (catTotalWords + vocabSize);
        likelihood += Math.log(wordProb);
      });

      const totalProb = prior + likelihood;
      if (totalProb > bestLogProb) {
        bestLogProb = totalProb;
        bestCat = cat;
      }
    });

    return bestCat;
  }
}

const aiClassifier = new PureJSNaiveBayes();
aiClassifier.train(SAMPLE_TRAINING_DATA);

// -----------------------------------------------------------------------------
// 3. AI FEATURE 2: LINEAR REGRESSION SPENDING PREDICTION
// -----------------------------------------------------------------------------
function predictNextMonthSpending(expenses) {
  if (!expenses || expenses.length === 0) {
    return {
      predicted: null,
      message: "No expense records found. Add expenses or load sample data to see predictions.",
      pastMonths: []
    };
  }

  const now = new Date();
  const currentYM = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

  // Group past expenses by YYYY-MM excluding current incomplete month
  const monthlyTotals = {};
  expenses.forEach(item => {
    const ym = item.date.slice(0, 7);
    if (ym < currentYM) {
      monthlyTotals[ym] = (monthlyTotals[ym] || 0) + Number(item.amount);
    }
  });

  const sortedMonths = Object.keys(monthlyTotals).sort();
  const pastMonths = sortedMonths.map(m => ({ month: m, total: monthlyTotals[m] }));
  const k = sortedMonths.length;

  if (k < 3) {
    return {
      predicted: null,
      message: `Need at least 3 completed months of data to predict next month. Currently have ${k} completed month(s). Click 'Load Sample Data' in the menu to test.`,
      pastMonths: pastMonths
    };
  }

  // Ordinary Least Squares (OLS) Linear Regression: x = 0, 1, ... k-1
  const yVals = sortedMonths.map(m => monthlyTotals[m]);
  const xMean = (k - 1) / 2.0;
  const yMean = yVals.reduce((a, b) => a + b, 0) / k;

  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < k; i++) {
    numerator += (i - xMean) * (yVals[i] - yMean);
    denominator += Math.pow(i - xMean, 2);
  }

  const slope = denominator !== 0 ? numerator / denominator : 0;
  const intercept = yMean - slope * xMean;
  const predicted = Math.max(0, slope * k + intercept);

  return {
    predicted: Math.round(predicted * 100) / 100,
    message: `Based on ${k} past completed months using Linear Regression.`,
    pastMonths: pastMonths,
    slope: slope,
    intercept: intercept
  };
}

// -----------------------------------------------------------------------------
// 4. STORAGE MANAGER (MULTI-USER LOCALSTORAGE)
// -----------------------------------------------------------------------------
const StorageManager = {
  getCurrentUser() {
    return localStorage.getItem("tracker_active_user") || "demo";
  },

  setCurrentUser(username) {
    const clean = username.trim().toLowerCase();
    localStorage.setItem("tracker_active_user", clean);
    this.ensureUser(clean);
  },

  getAllUsers() {
    try {
      const users = JSON.parse(localStorage.getItem("tracker_user_list"));
      if (Array.isArray(users) && users.length > 0) return users;
    } catch (e) {}
    const defaultUsers = ["demo"];
    localStorage.setItem("tracker_user_list", JSON.stringify(defaultUsers));
    return defaultUsers;
  },

  ensureUser(username) {
    const users = this.getAllUsers();
    if (!users.includes(username)) {
      users.push(username);
      localStorage.setItem("tracker_user_list", JSON.stringify(users));
    }
  },

  getBudget(username) {
    const b = localStorage.getItem(`tracker_budget_${username}`);
    return b ? parseFloat(b) : 20000.0;
  },

  setBudget(username, amount) {
    localStorage.setItem(`tracker_budget_${username}`, Number(amount).toFixed(2));
  },

  getExpenses(username) {
    try {
      const raw = localStorage.getItem(`tracker_expenses_${username}`);
      if (raw) return JSON.parse(raw);
    } catch (e) {}
    return [];
  },

  saveExpenses(username, expenses) {
    localStorage.setItem(`tracker_expenses_${username}`, JSON.stringify(expenses));
  },

  addExpense(username, expense) {
    const expenses = this.getExpenses(username);
    expenses.unshift({
      id: Date.now() + Math.floor(Math.random() * 1000),
      date: expense.date,
      description: expense.description,
      amount: parseFloat(expense.amount),
      category: expense.category
    });
    this.saveExpenses(username, expenses);
  },

  deleteExpense(username, id) {
    let expenses = this.getExpenses(username);
    expenses = expenses.filter(item => item.id !== id);
    this.saveExpenses(username, expenses);
  },

  seedDemoDataIfEmpty(username) {
    const existing = this.getExpenses(username);
    if (existing.length === 0) {
      this.generateSampleData(username);
    }
  },

  generateSampleData(username) {
    const sampleTemplates = [
      ["Grocery shopping at supermarket", [800, 2500], "Food"],
      ["Domino's Pizza dinner with classmates", [450, 950], "Food"],
      ["Starbucks coffee & sandwich", [250, 600], "Food"],
      ["Subway lunch with team", [220, 500], "Food"],
      ["Metro smart card monthly recharge", [400, 800], "Transport"],
      ["Uber cab ride to railway station", [280, 650], "Transport"],
      ["Petrol fuel tank refill", [1200, 2200], "Transport"],
      ["Ola auto fare commute", [80, 200], "Transport"],
      ["College textbooks & stationery", [650, 1500], "Shopping"],
      ["Amazon electronics order", [999, 2800], "Shopping"],
      ["Clothing purchase at mall", [1100, 3200], "Shopping"],
      ["Broadband wifi high-speed internet", [699, 999], "Bills"],
      ["Mobile phone recharge postpaid", [399, 699], "Bills"],
      ["Electricity bill payment", [1200, 2400], "Bills"],
      ["Netflix monthly streaming subscription", [499, 649], "Bills"],
      ["Movie tickets IMAX multiplex", [400, 900], "Entertainment"],
      ["Bowling and arcade games outing", [500, 1200], "Entertainment"],
      ["Medical prescription pharmacy", [250, 850], "Other"],
      ["Haircut and grooming salon", [150, 450], "Other"]
    ];

    const today = new Date();
    const rows = [];

    for (let offset = 5; offset >= 0; offset--) {
      const targetDate = new Date(today.getFullYear(), today.getMonth() - offset, 1);
      const targetYear = targetDate.getFullYear();
      const targetMonth = targetDate.getMonth() + 1;
      const maxDay = offset === 0 ? Math.max(1, today.getDate()) : 28;
      const numEntries = Math.floor(Math.random() * 3) + 6; // 6 to 8 entries

      for (let j = 0; j < numEntries; j++) {
        const t = sampleTemplates[Math.floor(Math.random() * sampleTemplates.length)];
        const day = Math.floor(Math.random() * maxDay) + 1;
        const dtStr = `${targetYear}-${String(targetMonth).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
        const amt = Math.round((Math.random() * (t[1][1] - t[1][0]) + t[1][0]) * 100) / 100;
        rows.push({
          id: Date.now() + Math.floor(Math.random() * 100000) + j,
          date: dtStr,
          description: t[0],
          amount: amt,
          category: t[2]
        });
      }
    }

    rows.sort((a, b) => b.date.localeCompare(a.date));
    this.saveExpenses(username, rows);
  }
};

// -----------------------------------------------------------------------------
// 5. CHART.JS VISUALIZATION CONTROLLER
// -----------------------------------------------------------------------------
let pieChartInstance = null;
let barChartInstance = null;
let lineChartInstance = null;

function renderCharts(expenses) {
  const categoryTotals = { "Food": 0, "Transport": 0, "Shopping": 0, "Bills": 0, "Entertainment": 0, "Other": 0 };
  const monthlyTotals = {};
  const dailyTotals = {};
  let totalSpent = 0;

  expenses.forEach(item => {
    const cat = CATEGORIES.includes(item.category) ? item.category : "Other";
    const amt = Number(item.amount);
    categoryTotals[cat] = (categoryTotals[cat] || 0) + amt;
    totalSpent += amt;

    const ym = item.date.slice(0, 7);
    monthlyTotals[ym] = (monthlyTotals[ym] || 0) + amt;

    dailyTotals[item.date] = (dailyTotals[item.date] || 0) + amt;
  });

  // Filter active categories for pie chart
  const activeLabels = [];
  const activeData = [];
  const activeColors = [];
  const activeBorders = [];

  CATEGORIES.forEach(cat => {
    if (categoryTotals[cat] > 0) {
      activeLabels.push(cat);
      activeData.push(categoryTotals[cat]);
      activeColors.push(PASTEL_COLORS[cat]);
      activeBorders.push(PASTEL_BORDER_COLORS[cat]);
    }
  });

  if (activeLabels.length === 0) {
    activeLabels.push("No Expenses");
    activeData.push(1);
    activeColors.push("#E2E8F0");
    activeBorders.push("#CBD5E1");
  }

  // 1. PIE / DOUGHNUT CHART (Category Breakdown)
  const pieCtx = document.getElementById("categoryPieChart");
  if (pieCtx && typeof Chart !== "undefined") {
    if (pieChartInstance) pieChartInstance.destroy();
    pieChartInstance = new Chart(pieCtx, {
      type: "doughnut",
      data: {
        labels: activeLabels,
        datasets: [{
          data: activeData,
          backgroundColor: activeColors,
          borderColor: "#FFFFFF",
          borderWidth: 2,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 12, font: { family: "Poppins", size: 11 }, color: "#334155" }
          },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                const val = ctx.raw || 0;
                const pct = totalSpent > 0 ? ((val / totalSpent) * 100).toFixed(1) : 0;
                return ` ${ctx.label}: ₹${val.toLocaleString('en-IN', {minimumFractionDigits: 2})} (${pct}%)`;
              }
            }
          }
        },
        cutout: "62%"
      }
    });
  }

  // 2. BAR CHART (Monthly Totals)
  const barCtx = document.getElementById("monthlyBarChart");
  if (barCtx && typeof Chart !== "undefined") {
    if (barChartInstance) barChartInstance.destroy();

    const sortedMonths = Object.keys(monthlyTotals).sort().slice(-6);
    const monthValues = sortedMonths.map(m => monthlyTotals[m]);
    const maxVal = monthValues.length ? Math.max(...monthValues) : 0;
    const barColors = monthValues.map(v => v === maxVal && maxVal > 0 ? "#A5B4FC" : "#99F6E4");
    const barBorderColors = monthValues.map(v => v === maxVal && maxVal > 0 ? "#818CF8" : "#5EEAD4");

    barChartInstance = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: sortedMonths,
        datasets: [{
          label: "Monthly Spend",
          data: monthValues,
          backgroundColor: barColors,
          borderColor: barBorderColors,
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return ` ₹${ctx.raw.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Poppins", size: 10 }, color: "#64748B" }
          },
          y: {
            beginAtZero: true,
            grid: { color: "#F1F5F9" },
            ticks: {
              font: { family: "Poppins", size: 10 },
              color: "#64748B",
              callback: val => '₹' + (val >= 1000 ? (val/1000).toFixed(0) + 'k' : val)
            }
          }
        }
      }
    });
  }

  // 3. LINE CHART (Recent Daily Spending Trend)
  const lineCtx = document.getElementById("dailyLineChart");
  if (lineCtx && typeof Chart !== "undefined") {
    if (lineChartInstance) lineChartInstance.destroy();

    const sortedDates = Object.keys(dailyTotals).sort().slice(-14);
    const dayValues = sortedDates.map(d => dailyTotals[d]);
    const displayDates = sortedDates.map(d => d.slice(5));

    lineChartInstance = new Chart(lineCtx, {
      type: "line",
      data: {
        labels: displayDates,
        datasets: [{
          label: "Daily Spend",
          data: dayValues,
          borderColor: "#818CF8",
          backgroundColor: "rgba(165, 180, 252, 0.25)",
          borderWidth: 2.5,
          tension: 0.35,
          fill: true,
          pointBackgroundColor: "#A5B4FC",
          pointBorderColor: "#4338CA",
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return ` ₹${ctx.raw.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Poppins", size: 9 }, color: "#64748B", maxRotation: 45 }
          },
          y: {
            beginAtZero: true,
            grid: { color: "#F1F5F9" },
            ticks: {
              font: { family: "Poppins", size: 10 },
              color: "#64748B",
              callback: val => '₹' + (val >= 1000 ? (val/1000).toFixed(0) + 'k' : val)
            }
          }
        }
      }
    });
  }
}

// -----------------------------------------------------------------------------
// 6. UI RENDERER & EVENT COORDINATOR
// -----------------------------------------------------------------------------
function refreshAppUI() {
  const currentUser = StorageManager.getCurrentUser();
  const expenses = StorageManager.getExpenses(currentUser);
  const budget = StorageManager.getBudget(currentUser);

  // Update Navbar user pill & label
  const userPill = document.getElementById("currentUserPill");
  if (userPill) userPill.innerText = `👤 ${currentUser}`;
  const userLabel = document.getElementById("currentUserLabel");
  if (userLabel) userLabel.innerText = currentUser;
  const modalActiveUser = document.getElementById("modalActiveUser");
  if (modalActiveUser) modalActiveUser.innerText = currentUser;
  const budgetModalUser = document.getElementById("budgetModalUser");
  if (budgetModalUser) budgetModalUser.innerText = currentUser;

  // Update profile pills inside modal
  const profilesList = document.getElementById("userProfilesList");
  if (profilesList) {
    const users = StorageManager.getAllUsers();
    profilesList.innerHTML = users.map(u => `
      <button type="button" class="user-profile-pill ${u === currentUser ? 'active' : ''}" onclick="switchToUser('${u}')">
        👤 ${u} ${u === currentUser ? '✓' : ''}
      </button>
    `).join('');
  }

  // Current month stats
  const now = new Date();
  const currentYM = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
  let thisMonthSpent = 0;
  const categorySpent = { "Food": 0, "Transport": 0, "Shopping": 0, "Bills": 0, "Entertainment": 0, "Other": 0 };

  expenses.forEach(item => {
    if (item.date.startsWith(currentYM)) {
      const amt = Number(item.amount);
      thisMonthSpent += amt;
      const cat = CATEGORIES.includes(item.category) ? item.category : "Other";
      categorySpent[cat] = (categorySpent[cat] || 0) + amt;
    }
  });

  // Top category
  let topCat = "None";
  let topCatAmt = 0;
  CATEGORIES.forEach(cat => {
    if (categorySpent[cat] > topCatAmt) {
      topCatAmt = categorySpent[cat];
      topCat = cat;
    }
  });
  const topCatPct = thisMonthSpent > 0 ? ((topCatAmt / thisMonthSpent) * 100).toFixed(1) : 0;

  // Budget calculations
  const pctBudget = budget > 0 ? (thisMonthSpent / budget) * 100 : 0;
  const budgetLeft = budget - thisMonthSpent;

  // Linear Regression Prediction
  const predResult = predictNextMonthSpending(expenses);

  // Update KPI Cards
  const kpiSpent = document.getElementById("kpiTotalSpent");
  const kpiSpentSub = document.getElementById("kpiTotalSpentSub");
  if (kpiSpent) kpiSpent.innerText = `₹${thisMonthSpent.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  if (kpiSpentSub) kpiSpentSub.innerText = `${pctBudget.toFixed(1)}% of ₹${budget.toLocaleString('en-IN', {maximumFractionDigits: 0})} budget`;

  const kpiTopCat = document.getElementById("kpiTopCat");
  const kpiTopCatSub = document.getElementById("kpiTopCatSub");
  if (kpiTopCat) kpiTopCat.innerText = topCat;
  if (kpiTopCatSub) {
    kpiTopCatSub.innerText = topCatAmt > 0
      ? `₹${topCatAmt.toLocaleString('en-IN', {minimumFractionDigits: 2})} (${topCatPct}%)`
      : "No spending yet";
  }

  const kpiBudgetCard = document.getElementById("kpiBudgetCard");
  const kpiBudget = document.getElementById("kpiBudgetStatus");
  const kpiBudgetSub = document.getElementById("kpiBudgetStatusSub");
  if (kpiBudgetCard) {
    kpiBudgetCard.classList.remove("accent-secondary", "accent-warning", "accent-danger");
    if (thisMonthSpent > budget) {
      kpiBudgetCard.classList.add("accent-danger");
      if (kpiBudget) kpiBudget.innerText = `₹${Math.abs(budgetLeft).toLocaleString('en-IN', {minimumFractionDigits: 2})} Over Limit`;
      if (kpiBudgetSub) kpiBudgetSub.innerText = "Alert: Exceeded monthly limit";
    } else if (pctBudget >= 80.0) {
      kpiBudgetCard.classList.add("accent-warning");
      if (kpiBudget) kpiBudget.innerText = `₹${budgetLeft.toLocaleString('en-IN', {minimumFractionDigits: 2})} Left`;
      if (kpiBudgetSub) kpiBudgetSub.innerText = `Warning: ${pctBudget.toFixed(1)}% used`;
    } else {
      kpiBudgetCard.classList.add("accent-secondary");
      if (kpiBudget) kpiBudget.innerText = `₹${budgetLeft.toLocaleString('en-IN', {minimumFractionDigits: 2})} Left`;
      if (kpiBudgetSub) kpiBudgetSub.innerText = `Healthy: ${pctBudget.toFixed(1)}% used`;
    }
  }

  const kpiPred = document.getElementById("kpiPrediction");
  const kpiPredSub = document.getElementById("kpiPredictionSub");
  if (kpiPred) {
    kpiPred.innerText = predResult.predicted !== null
      ? `₹${predResult.predicted.toLocaleString('en-IN', {minimumFractionDigits: 2})}`
      : "N/A";
  }
  if (kpiPredSub) {
    kpiPredSub.innerText = predResult.predicted !== null
      ? "Predicted via Linear Regression"
      : "Need 3+ completed months";
  }

  // Budget Alerts Banner
  const alertContainer = document.getElementById("alertContainer");
  if (alertContainer) {
    if (thisMonthSpent > budget) {
      const overBy = thisMonthSpent - budget;
      alertContainer.innerHTML = `
        <div class="alert alert-danger">
          <span class="alert-icon">🚨</span>
          <div class="alert-content">
            <strong>Budget Exceeded!</strong> This month's spending (₹${thisMonthSpent.toLocaleString('en-IN', {minimumFractionDigits: 2})}) 
            has exceeded your monthly limit of ₹${budget.toLocaleString('en-IN')} by ₹${overBy.toLocaleString('en-IN', {minimumFractionDigits: 2})}!
          </div>
        </div>`;
    } else if (pctBudget >= 80.0) {
      alertContainer.innerHTML = `
        <div class="alert alert-warning">
          <span class="alert-icon">⚠️</span>
          <div class="alert-content">
            <strong>Budget Warning!</strong> You have spent ${pctBudget.toFixed(1)}% of your monthly limit 
            (₹${thisMonthSpent.toLocaleString('en-IN', {minimumFractionDigits: 2})} / ₹${budget.toLocaleString('en-IN')}).
          </div>
        </div>`;
    } else {
      alertContainer.innerHTML = "";
    }
  }

  // Render Category Progress Bars
  const barsContainer = document.getElementById("categoryBarsContainer");
  if (barsContainer) {
    const activeBreakdown = [];
    CATEGORIES.forEach(cat => {
      const amt = categorySpent[cat];
      if (amt > 0) {
        const pct = ((amt / thisMonthSpent) * 100).toFixed(1);
        activeBreakdown.push({ category: cat, amount: amt, pct: pct, color: PASTEL_COLORS[cat] });
      }
    });
    activeBreakdown.sort((a, b) => b.amount - a.amount);

    if (activeBreakdown.length > 0) {
      barsContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; margin-top: 1rem; border-top: 1px solid var(--border-color); padding-top: 0.85rem;">
          <span style="font-size: 0.825rem; font-weight: 600; color: var(--text-main);">This Month Category Distribution</span>
          <span style="font-size: 0.775rem; color: var(--text-muted);">${activeBreakdown.length} active categories</span>
        </div>
        ${activeBreakdown.map(item => `
          <div class="category-bar-item">
            <div class="category-bar-header">
              <span><strong>${item.category}</strong>: ₹${item.amount.toLocaleString('en-IN', {minimumFractionDigits: 2})}</span>
              <span>${item.pct}%</span>
            </div>
            <div class="category-bar-track">
              <div class="category-bar-fill" style="width: ${item.pct}%; background-color: ${item.color};"></div>
            </div>
          </div>
        `).join('')}
      `;
    } else {
      barsContainer.innerHTML = `
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 1rem; text-align: center;">
          No transactions logged yet this month. Add an expense above to see category distribution!
        </p>`;
    }
  }

  // Render Transactions Table (Newest 10 or all)
  const tableBody = document.getElementById("expensesTableBody");
  if (tableBody) {
    if (expenses.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
            No expenses recorded yet for <strong>${currentUser}</strong>. Add your first expense above or click <strong>Load Sample Data</strong>!
          </td>
        </tr>`;
    } else {
      tableBody.innerHTML = expenses.slice(0, 15).map(item => `
        <tr>
          <td>${item.date}</td>
          <td>${escapeHtml(item.description)}</td>
          <td><span class="badge badge-${item.category.toLowerCase()}">${item.category}</span></td>
          <td class="td-amount">₹${Number(item.amount).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
          <td>
            <button class="btn-sm-danger" onclick="deleteTransaction(${item.id})">Delete</button>
          </td>
        </tr>
      `).join('');
    }
  }

  // Render Charts
  renderCharts(expenses);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

// -----------------------------------------------------------------------------
// 7. USER ACTIONS (ADD, DELETE, IMPORT, EXPORT, SAMPLE DATA)
// -----------------------------------------------------------------------------
function handleAddExpense(e) {
  e.preventDefault();
  const dateInput = document.getElementById("expDate");
  const descInput = document.getElementById("expDescription");
  const amtInput = document.getElementById("expAmount");
  const catInput = document.getElementById("expCategory");

  const dateVal = dateInput.value.trim();
  const descVal = descInput.value.trim();
  const amtVal = parseFloat(amtInput.value);
  let catVal = catInput.value;

  if (!descVal) {
    alert("Please enter an expense description.");
    return;
  }
  if (isNaN(amtVal) || amtVal <= 0) {
    alert("Please enter a valid positive expense amount.");
    return;
  }
  if (!dateVal) {
    alert("Please select a date.");
    return;
  }

  if (!CATEGORIES.includes(catVal)) {
    catVal = aiClassifier.predict(descVal);
  }

  const currentUser = StorageManager.getCurrentUser();
  StorageManager.addExpense(currentUser, {
    date: dateVal,
    description: descVal,
    amount: amtVal,
    category: catVal
  });

  // Reset form
  descInput.value = "";
  amtInput.value = "";
  descInput.focus();

  refreshAppUI();
}

function deleteTransaction(id) {
  if (confirm("Delete this expense record?")) {
    const currentUser = StorageManager.getCurrentUser();
    StorageManager.deleteExpense(currentUser, id);
    refreshAppUI();
  }
}

function handleLoadSampleData() {
  const currentUser = StorageManager.getCurrentUser();
  if (confirm(`Load 6 months of sample data for ${currentUser}? This will populate all 3 charts and the Linear Regression model.`)) {
    StorageManager.generateSampleData(currentUser);
    refreshAppUI();
  }
}

function handleExportCSV() {
  const currentUser = StorageManager.getCurrentUser();
  const expenses = StorageManager.getExpenses(currentUser);
  if (expenses.length === 0) {
    alert("No expenses recorded yet to export.");
    return;
  }

  let csvContent = "date,description,amount,category\n";
  expenses.forEach(item => {
    const escapedDesc = `"${item.description.replace(/"/g, '""')}"`;
    csvContent += `${item.date},${escapedDesc},${Number(item.amount).toFixed(2)},${item.category}\n`;
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `expenses_${currentUser}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function handleImportCSV(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length <= 1) {
      alert("CSV file is empty or missing data rows.");
      return;
    }

    const imported = [];
    for (let i = 1; i < lines.length; i++) {
      // Basic CSV splitter handling quotes
      const row = lines[i].match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || lines[i].split(',');
      if (row.length >= 4) {
        const dStr = row[0].replace(/"/g, '').trim();
        const desc = row[1].replace(/"/g, '').trim();
        const amt = parseFloat(row[2].replace(/"/g, '').trim());
        const cat = row[3].replace(/"/g, '').trim();

        if (dStr && desc && !isNaN(amt) && amt > 0) {
          imported.push({
            id: Date.now() + Math.floor(Math.random() * 100000) + i,
            date: dStr,
            description: desc,
            amount: amt,
            category: CATEGORIES.includes(cat) ? cat : "Other"
          });
        }
      }
    }

    if (imported.length > 0) {
      const currentUser = StorageManager.getCurrentUser();
      const current = StorageManager.getExpenses(currentUser);
      const combined = [...imported, ...current];
      combined.sort((a, b) => b.date.localeCompare(a.date));
      StorageManager.saveExpenses(currentUser, combined);
      refreshAppUI();
      alert(`Successfully imported ${imported.length} expenses into ${currentUser}'s account!`);
    } else {
      alert("Could not parse valid expense rows. Required format: date,description,amount,category");
    }
  };
  reader.readAsText(file);
}

// -----------------------------------------------------------------------------
// USER MANAGEMENT & MODAL CONTROLLERS
// -----------------------------------------------------------------------------
function openUserModal() {
  const modal = document.getElementById("userModal");
  if (modal) {
    modal.style.display = "flex";
    const input = document.getElementById("newUsernameInput");
    if (input) {
      input.value = "";
      setTimeout(() => input.focus(), 100);
    }
  }
}

function closeUserModal() {
  const modal = document.getElementById("userModal");
  if (modal) modal.style.display = "none";
}

function handleCreateUser(e) {
  e.preventDefault();
  const input = document.getElementById("newUsernameInput");
  const raw = input ? input.value.trim().toLowerCase() : "";
  const clean = raw.replace(/[^a-z0-9_]/g, '');

  if (clean.length < 2) {
    alert("Please enter a valid username of at least 2 alphanumeric characters.");
    return;
  }

  StorageManager.ensureUser(clean);
  StorageManager.setCurrentUser(clean);
  StorageManager.seedDemoDataIfEmpty(clean);
  closeUserModal();
  refreshAppUI();

  // Show friendly alert
  const alertContainer = document.getElementById("alertContainer");
  if (alertContainer) {
    alertContainer.innerHTML = `
      <div class="alert alert-success">
        <span class="alert-icon">✅</span>
        <div class="alert-content">
          <strong>Account Ready!</strong> Switched to user profile <strong>${clean}</strong>. Your expenses, budget, and charts are completely private.
        </div>
      </div>`;
    setTimeout(() => {
      if (alertContainer.innerHTML.includes("Account Ready!")) alertContainer.innerHTML = "";
    }, 6000);
  }
}

function switchToUser(username) {
  StorageManager.setCurrentUser(username);
  closeUserModal();
  refreshAppUI();
}

function openBudgetModal() {
  const modal = document.getElementById("budgetModal");
  const currentUser = StorageManager.getCurrentUser();
  const currentBudget = StorageManager.getBudget(currentUser);
  const input = document.getElementById("budgetModalInput");
  const userEl = document.getElementById("budgetModalUser");
  if (userEl) userEl.innerText = currentUser;
  if (input) input.value = currentBudget;
  if (modal) modal.style.display = "flex";
}

function closeBudgetModal() {
  const modal = document.getElementById("budgetModal");
  if (modal) modal.style.display = "none";
}

function handleSaveBudget(e) {
  e.preventDefault();
  const input = document.getElementById("budgetModalInput");
  const val = parseFloat(input.value);
  if (isNaN(val) || val <= 0) {
    alert("Please enter a valid positive budget amount.");
    return;
  }
  const currentUser = StorageManager.getCurrentUser();
  StorageManager.setBudget(currentUser, val);
  closeBudgetModal();
  refreshAppUI();
}

// -----------------------------------------------------------------------------
// 8. INITIALIZATION & LIVE AI EVENT BINDING
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", function() {
  // Set default date to today
  const dateInput = document.getElementById("expDate");
  if (dateInput) {
    const todayStr = new Date().toISOString().split("T")[0];
    dateInput.value = todayStr;
  }

  // Ensure demo user has data on initial launch
  const initialUser = StorageManager.getCurrentUser();
  StorageManager.ensureUser(initialUser);
  StorageManager.seedDemoDataIfEmpty(initialUser);

  // Live Naive Bayes auto-categorization
  const descInput = document.getElementById("expDescription");
  const catSelect = document.getElementById("expCategory");
  const aiBadge = document.getElementById("aiBadge");
  let debounceTimer = null;

  if (descInput && catSelect) {
    descInput.addEventListener("input", function() {
      clearTimeout(debounceTimer);
      const text = descInput.value.trim();
      if (text.length < 3) return;

      debounceTimer = setTimeout(() => {
        const predicted = aiClassifier.predict(text);
        if (predicted) {
          catSelect.value = predicted;
          if (aiBadge) {
            aiBadge.innerText = `AI Suggested: ${predicted}`;
            aiBadge.style.backgroundColor = "var(--secondary)";
          }
        }
      }, 200);
    });
  }

  // Form submit binding
  const expenseForm = document.getElementById("expenseForm");
  if (expenseForm) expenseForm.addEventListener("submit", handleAddExpense);

  // Button bindings
  const loadSampleBtn = document.getElementById("btnLoadSample");
  if (loadSampleBtn) loadSampleBtn.addEventListener("click", handleLoadSampleData);

  const exportBtn = document.getElementById("btnExportCSV");
  if (exportBtn) exportBtn.addEventListener("click", handleExportCSV);

  const importFileInput = document.getElementById("importFileInput");
  if (importFileInput) importFileInput.addEventListener("change", handleImportCSV);

  const budgetBtn = document.getElementById("btnSetBudget");
  if (budgetBtn) budgetBtn.addEventListener("click", openBudgetModal);

  const openUserModalBtn = document.getElementById("btnOpenUserModal");
  if (openUserModalBtn) openUserModalBtn.addEventListener("click", openUserModal);

  const userBtn = document.getElementById("currentUserPill");
  if (userBtn) userBtn.addEventListener("click", openUserModal);

  const refreshChartBtn = document.getElementById("btnRefreshCharts");
  if (refreshChartBtn) refreshChartBtn.addEventListener("click", () => {
    refreshAppUI();
  });

  // Initial UI refresh
  refreshAppUI();
});
