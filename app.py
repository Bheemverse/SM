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
CORS(app)  # Enable CORS for all routes

file_path = 'dummy_supermarket_sales (2).xlsx'

def generate_rules(min_support=None, min_confidence=None):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        if 'Quantity' in df.columns:
            df = df.drop(columns=['Quantity'])

        transactions = df.groupby(['Invoice ID', 'Product']).size().unstack().fillna(0)
        transactions = (transactions > 0).astype(int)

        if transactions.shape[0] == 0 or transactions.shape[1] == 0:
            raise ValueError("No valid transactions were found after grouping.")

        if min_support is None:
            avg_product_freq = transactions.sum(axis=0).mean()
            total_transactions = len(transactions)
            min_support = max(0.005, avg_product_freq / total_transactions * 0.1)

        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)

        if min_confidence is None:
            min_confidence = 0.1

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return rules.sort_values('confidence', ascending=False)

    except Exception as e:
        raise Exception(f"Error generating rules: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/rules',
            '/api/download/rules',
            '/api/frequent_products',
            '/api/products'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = request.args.get('min_support', None)
        min_confidence = request.args.get('min_confidence', None)

        min_support = float(min_support) if min_support is not None else None
        min_confidence = float(min_confidence) if min_confidence is not None else None

        if min_support is not None and not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if min_confidence is not None and not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")

        rules = generate_rules(min_support, min_confidence)

        rules_list = []
        for idx, row in rules.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            full_rule = f"{', '.join(antecedents)} -> {', '.join(consequents)}"
            rule_dict = {
                'rule': full_rule,
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            }
            rules_list.append(rule_dict)

        table_data = "Rule                                             | Support | Confidence | Lift\n"
        table_data += "-" * 90 + "\n"

        for rule in rules_list:
            table_data += f"{rule['rule']:<50} | {rule['support']:<7.4f} | {rule['confidence']:<10.4f} | {rule['lift']:<5.4f}\n"

        return jsonify({
            "status": "success",
            "rules_count": len(rules_list),
            "rules_table": table_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        rules = generate_rules()
        temp_file = 'temp_rules.json'
        rules.to_json(temp_file, orient='records')

        return send_file(
            temp_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='association_rules.json'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frequent_products', methods=['GET'])
def frequent_products():
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        product_frequency = df['Product'].value_counts()

        product_info = []
        for product in product_frequency.index:
            product_data = {
                'product': product,
                'frequency': int(product_frequency[product])
            }
            if 'Total' in df.columns:
                product_sales = df.groupby('Product')['Total'].sum()
                product_data['total_sales'] = float(product_sales.get(product, 0))
            product_info.append(product_data)

        return jsonify({
            'status': 'success',
            'frequent_products': product_info
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# New API to fetch all unique products
@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        products = df['Product'].dropna().unique().tolist()

        return jsonify({
            'status': 'success',
            'products': products,
            'count': len(products)
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True,
        port=5000
    )
