from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Load the dataset
def load_data():
    if not os.path.exists('supermarket_sales.xlsx'):
        raise FileNotFoundError("File 'supermarket_sales.xlsx' not found.")
    df = pd.read_excel('supermarket_sales.xlsx')
    if df.empty:
        raise ValueError("The Excel file is empty.")
    return df

# Prepare transactions
def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

# Generate frequent itemsets
def generate_frequent_itemsets(transactions, min_support=0.01):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# Generate association rules
def generate_rules(frequent_itemsets, min_confidence=0.3):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values('confidence', ascending=False)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/products',
            '/api/rules',
            '/api/frequent-items',
            '/api/download/rules'
        ]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        df = load_data()
        products = df['Product'].unique().tolist()
        return jsonify({
            "status": "success",
            "products_count": len(products),
            "products": products
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
                'antecedents': list(map(str, row['antecedents'])),
                'consequents': list(map(str, row['consequents'])),
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

@app.route('/api/frequent-items', methods=['GET'])
def get_frequent_items():
    try:
        min_support = float(request.args.get('min_support', 0.01))

        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1.")

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)

        frequent_list = []
        for _, row in frequent_itemsets.iterrows():
            frequent_list.append({
                'items': list(map(str, row['itemsets'])),
                'support': float(row['support'])
            })

        return jsonify({
            "status": "success",
            "frequent_itemsets_count": len(frequent_list),
            "frequent_itemsets": frequent_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
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
                'antecedents': list(map(str, row['antecedents'])),
                'consequents': list(map(str, row['consequents'])),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            })

        temp_file = 'temp_rules.json'
        pd.DataFrame(rules_list).to_json(temp_file, orient='records', indent=2)

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
            try:
                os.remove('temp_rules.json')
            except:
                pass

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True,
        port=5000
    )
