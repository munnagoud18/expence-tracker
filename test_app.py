"""
Comprehensive Multi-User Test Suite for Smart Expense Tracker.
Tests registration, authentication, data isolation between multiple users,
AI features, and all routes.
"""
import unittest
from app import (
    app, init_storage, register_user, load_user_expenses,
    get_user_budget, set_user_budget
)

class MultiUserExpenseTrackerTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        init_storage()

    def test_01_user_registration_and_login(self):
        # Register user Alice
        res = self.client.post('/register', data={
            'username': 'alice',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        if b"already taken" in res.data:
            self.client.post('/login', data={'username': 'alice', 'password': 'password123'}, follow_redirects=True)
        self.client.get('/logout')

        # Login Alice
        res = self.client.post('/login', data={
            'username': 'alice',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Welcome back, alice", res.data)
        print("[PASS] User Login (Alice)")

    def test_02_demo_login(self):
        res = self.client.post('/login', data={
            'username': 'demo',
            'password': 'demo123'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"demo", res.data)
        print("[PASS] Instant Demo Login")

    def test_03_multiuser_data_isolation(self):
        # 1. Register Bob or login if already exists
        reg_res = self.client.post('/register', data={
            'username': 'bob',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        if b"already taken" in reg_res.data:
            self.client.post('/login', data={
                'username': 'bob',
                'password': 'password123'
            }, follow_redirects=True)

        # 2. Add an expense for Bob
        res = self.client.post('/add', data={
            'date': '2026-10-29',
            'description': 'Bob private laptop bag',
            'amount': '1500.00',
            'category': 'Shopping'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Bob private laptop bag", res.data)

        # 3. Log out Bob and log in as Alice
        self.client.get('/logout')
        self.client.post('/login', data={
            'username': 'alice',
            'password': 'password123'
        }, follow_redirects=True)

        # 4. Verify Alice CANNOT see Bob's expense
        alice_expenses_page = self.client.get('/expenses')
        self.assertNotIn(b"Bob private laptop bag", alice_expenses_page.data)
        print("[PASS] Data Isolation: Alice cannot see Bob's expenses!")

        # 5. Add expense for Alice
        self.client.post('/add', data={
            'date': '2026-10-29',
            'description': 'Alice coffee and croissant',
            'amount': '250.00',
            'category': 'Food'
        }, follow_redirects=True)

        # 6. Log out Alice and log in as Bob
        self.client.get('/logout')
        self.client.post('/login', data={
            'username': 'bob',
            'password': 'password123'
        }, follow_redirects=True)

        # 7. Verify Bob CANNOT see Alice's expense
        bob_expenses_page = self.client.get('/expenses')
        self.assertNotIn(b"Alice coffee and croissant", bob_expenses_page.data)
        self.assertIn(b"Bob private laptop bag", bob_expenses_page.data)
        print("[PASS] Data Isolation: Bob cannot see Alice's expenses!")

    def test_04_budget_isolation(self):
        # Set Alice's budget to 35000
        self.client.get('/logout')
        self.client.post('/login', data={'username': 'alice', 'password': 'password123'})
        self.client.post('/budget', data={'budget': '35000'}, follow_redirects=True)
        self.assertEqual(get_user_budget('alice'), 35000.0)

        # Set Bob's budget to 12000
        self.client.get('/logout')
        self.client.post('/login', data={'username': 'bob', 'password': 'password123'})
        self.client.post('/budget', data={'budget': '12000'}, follow_redirects=True)
        self.assertEqual(get_user_budget('bob'), 12000.0)

        # Confirm Alice's budget did not change
        self.assertEqual(get_user_budget('alice'), 35000.0)
        print("[PASS] Budget Isolation: Each user has their own distinct budget!")

    def test_05_ai_category_suggestion(self):
        res = self.client.get('/suggest-category?description=Dominos+Cheese+Burst+Pizza')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json.get("category"), "Food")

        res2 = self.client.get('/suggest-category?description=Uber+cab+ride')
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json.get("category"), "Transport")
        print("[PASS] AI Feature 1: Naive Bayes auto-categorization works for all users")

    def test_06_load_sample_data_per_user(self):
        bob_expenses_before = len(load_user_expenses('bob'))
        self.client.post('/login', data={'username': 'alice', 'password': 'password123'})
        res = self.client.get('/load-sample', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        alice_expenses = load_user_expenses('alice')
        self.assertGreater(len(alice_expenses), 20)

        # Bob's expense count should remain unchanged
        bob_expenses_after = len(load_user_expenses('bob'))
        self.assertEqual(bob_expenses_after, bob_expenses_before)
        print("[PASS] Load Sample Data only applies to the active logged-in user!")

    def test_07_export_csv_per_user(self):
        self.client.post('/login', data={'username': 'alice', 'password': 'password123'})
        res = self.client.get('/export')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/csv')
        self.assertIn(b"expenses_alice.csv", res.headers.get("Content-Disposition", "").encode())
        print("[PASS] Export CSV downloads user-specific file")

    def test_08_dashboard_chart_dynamic(self):
        self.client.post('/login', data={'username': 'alice', 'password': 'password123'})
        res = self.client.get('/dashboard-chart')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'image/png')
        self.assertEqual(res.headers.get('Cache-Control'), 'no-cache, no-store, must-revalidate, max-age=0')
        self.assertGreater(len(res.data), 1000)
        print("[PASS] Dynamic dashboard chart route generates and serves no-cache PNG!")

if __name__ == '__main__':
    unittest.main()
