from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Constants
FILE_PATH = 'dummy_supermarket_sales (2).xlsx'

# Global cache
data_cache = {
    'transactions': None,
    'products': None
}

def load_data():
    """Load and prepare transaction data"""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File '{FILE_PATH}' not found.")
    
    df = pd.read_excel(FILE_PATH)
    if df.empty:
        raise ValueError("The Excel file is empty")
    
    if 'Quantity' in df.columns:
        df = df.drop(columns=['Quantity'])

    transaction_data = df.groupby(['Invoice ID', 'Product']).size().unstack().fillna(0)
    transaction_data = (transaction_data > 0).astype(int)
    
    if transaction_data.empty:
        raise ValueError("No valid transactions found.")

    data_cache['transactions'] = transaction_data
    data_cache['products'] = df['Product'].dropna().unique().tolist()
    logging.info("Data loaded successfully")

def generate_rules(min_support=None, min_confidence=0.1):
    """Generate association rules"""
    transaction_data = data_cache.get('transactions')
    if transaction_data is None:
        load_data()
        transaction_data = data_cache['transactions']

    avg_product_freq = transaction_data.sum(axis=0).mean()
    total_transactions = len(transaction_data)

    min_support = min_support or max(0.005, avg_product_freq / total_transactions * 0.1)

    frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found. Try lowering min_support.")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        raise ValueError("No association rules found. Try lowering min_confidence.")

    return rules

def filter_rules(rules, product, role='any'):
    """Filter rules by product and role"""
    filtered = []
    for _, row in rules.iterrows():
        ant = list(row['antecedents'])
        cons = list(row['consequents'])

        if (role == 'antecedent' and product in ant) or \
           (role == 'consequent' and product in cons) or \
           (role == 'any' and (product in ant or product in cons)):
            filtered.append(row)
    return pd.DataFrame(filtered)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/rules',
            '/api/download/rules',
            '/api/frequent_products',
            '/api/products',
            '/api/rules/by_antecedent?product=...',
            '/api/rules/by_consequent?product=...',
            '/api/rules/by_product?product=...'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = request.args.get('min_support', type=float)
        min_confidence = request.args.get('min_confidence', type=float, default=0.1)

        rules = generate_rules(min_support, min_confidence)
        rule_list = []
        for _, row in rules.iterrows():
            rule_list.append({
                "antecedents": list(row['antecedents']),
                "consequents": list(row['consequents']),
                "support": float(row['support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift'])
            })

        return jsonify({
            "status": "success",
            "rules_count": len(rule_list),
            "rules": rule_list
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        rules = generate_rules()

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(x))

        temp_file = 'association_rules.csv'
        rules.to_csv(temp_file, index=False)

        return send_file(temp_file, mimetype='text/csv', as_attachment=True, download_name='association_rules.csv')

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/frequent_products', methods=['GET'])
def frequent_products():
    try:
        if data_cache.get('transactions') is None:
            load_data()
        products_freq = data_cache['transactions'].sum(axis=0).sort_values(ascending=False).index.tolist()

        return jsonify(products_freq)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        if data_cache.get('products') is None:
            load_data()

        return jsonify({
            'status': 'success',
            'products': data_cache['products'],
            'count': len(data_cache['products'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_antecedent', methods=['GET'])
def rules_by_antecedent():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({"status": "error", "message": "Product parameter is required"}), 400

        rules = generate_rules()
        filtered = filter_rules(rules, product, role='antecedent')

        consequents = set()
        for conseq in filtered['consequents']:
            consequents.update(conseq)

        return jsonify(sorted(consequents))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_consequent', methods=['GET'])
def rules_by_consequent():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({"status": "error", "message": "Product parameter is required"}), 400

        rules = generate_rules()
        filtered = filter_rules(rules, product, role='consequent')

        antecedents = set()
        for ant in filtered['antecedents']:
            antecedents.update(ant)

        return jsonify(sorted(antecedents))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_product', methods=['GET'])
def rules_by_product():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({"status": "error", "message": "Product parameter is required"}), 400

        rules = generate_rules()
        filtered = filter_rules(rules, product, role='any')

        related = set()
        for _, row in filtered.iterrows():
            related.update(row['antecedents'])
            related.update(row['consequents'])

        related.discard(product)

        return jsonify(sorted(related))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    load_data()
    app.run(
        host='0.0.0.0',
        debug=True,
        port=5000
    )
