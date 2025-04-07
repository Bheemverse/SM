from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and prepare data
def load_data():
    if not os.path.exists('supermarket_sales.xlsx'):
        raise FileNotFoundError("File 'supermarket_sales.xlsx' not found.")
    df = pd.read_excel('supermarket_sales.xlsx')
    if df.empty:
        raise ValueError("The Excel file is empty")
    return df

def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

def generate_frequent_itemsets(transactions, min_support=0.01):
    return apriori(transactions, min_support=min_support, use_colnames=True)

def generate_rules(frequent_itemsets, min_confidence=0.3):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    return rules.sort_values('confidence', ascending=False)

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/products',
            '/api/rules',
            '/api/download/rules',
            '/api/associate-products',
            '/api/frequent-products'
        ]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        df = load_data()
        products = sorted(df['Product'].unique().tolist())
        return jsonify({
            'status': 'success',
            'products_count': len(products),
            'products': products
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)

        rules_list = []
        for _, row in rules.iterrows():
            rules_list.append({
                'antecedents': row['antecedents'],
                'consequents': row['consequents'],
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            })

        return jsonify({
            'status': 'success',
            'rules_count': len(rules_list),
            'rules': rules_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules_

