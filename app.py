from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to your dataset
DATA_FILE = 'supermarket_sales.xlsx'

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"File '{DATA_FILE}' not found.")
    df = pd.read_excel(DATA_FILE)
    if df.empty:
        raise ValueError("The Excel file is empty.")
    return df

def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

def generate_rules(min_support=0.01, min_confidence=0.3):
    df = load_data()
    transactions = prepare_transactions(df)
    
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return transactions.columns.tolist(), frequent_itemsets, rules.sort_values('confidence', ascending=False)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/products',
            '/api/frequent_itemsets',
            '/api/rules',
            '/api/download/rules'
        ]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        products, _, _ = generate_rules()
        return jsonify({
            'status': 'success',
            'products': products
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/frequent_itemsets', methods=['GET'])
def get_frequent_itemsets():
    try:
        _, frequent_itemsets, _ = generate_rules()
        itemsets = frequent_itemsets.to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'frequent_itemsets': itemsets
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))
        
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")
        
        _, _, rules = generate_rules(min_support, min_confidence)
        
        rules_list = []
        for idx, row in rules.iterrows():
            rule_dict = {
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            }
            rules_list.append(rule_dict)
        
        return jsonify({
            'status': 'success',
            'rules_count': len(rules_list),
            'rules': rules_list
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        _, _, rules = generate_rules()
        temp_file = 'temp_rules.json'
        rules.to_json(temp_file, orient='records')
        
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
    app.run(host='0.0.0.0', debug=True, port=5000)
