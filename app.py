from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

DATA_FILE = 'supermarket_sales.xlsx'

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"File '{DATA_FILE}' not found.")
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError("The Excel file is empty.")
    if 'Product' not in df.columns:
        raise ValueError("The required column 'Product' was not found.")
    return df

def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

def generate_frequent_itemsets(transactions, min_support):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Supermarket Association Rules API!',
        'available_endpoints': [
            '/api/products',
            '/api/frequent-items',
            '/api/rules',
            '/api/download/frequent-items',
            '/api/download/rules'
        ]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        df = load_data()
        products = sorted(df['Product'].unique().tolist())
        return jsonify({
            "status": "success",
            "products_count": len(products),
            "products": products
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/frequent-items', methods=['GET'])
def get_frequent_items():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1.")

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)

        frequent_items_list = frequent_itemsets.to_dict(orient='records')

        return jsonify({
            "status": "success",
            "frequent_items_count": len(frequent_items_list),
            "frequent_items": frequent_items_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))

        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1.")
        if not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1.")

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)

        rules_list = []
        for _, row in rules.iterrows():
            rules_list.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            })

        return jsonify({
            "status": "success",
            "rules_count": len(rules_list),
            "rules": rules_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/frequent-items', methods=['GET'])
def download_frequent_items():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1.")

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)

        temp_file = 'temp_frequent_items.json'
        frequent_itemsets.to_json(temp_file, orient='records')

        return send_file(
            temp_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='frequent_items.json'
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if os.path.exists('temp_frequent_items.json'):
            os.remove('temp_frequent_items.json')

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)

        rules_json = []
        for _, row in rules.iterrows():
            rules_json.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            })

        temp_file = 'temp_rules.json'
        with open(temp_file, 'w') as f:
            json.dump(rules_json, f, indent=2)

        return send_file(
            temp_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='association_rules.json'
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if os.path.exists('temp_rules.json'):
            os.remove('temp_rules.json')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
