from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

DATA_FILE = 'supermarket_sales.xlsx'

# ---------------- Helper Functions ---------------- #

def read_data(file_path=DATA_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("The Excel file is empty")
    
    return df

def create_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

def get_products(df):
    return df['Product line'].unique().tolist()

def get_frequent_itemsets(transactions, min_support=0.01):
    return apriori(transactions, min_support=min_support, use_colnames=True)

def get_association_rules(frequent_itemsets, min_confidence=0.3):
    return association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# ---------------- API Endpoints ---------------- #

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/products',
            '/api/frequent-itemsets',
            '/api/rules',
            '/api/download/rules'
        ]
    })

@app.route('/api/products', methods=['GET'])
def api_get_products():
    try:
        df = read_data()
        products = get_products(df)
        return jsonify({
            "status": "success",
            "products_count": len(products),
            "products": products
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/frequent-itemsets', methods=['GET'])
def api_get_frequent_itemsets():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        df = read_data()
        transactions = create_transactions(df)
        frequent_itemsets = get_frequent_itemsets(transactions, min_support)
        
        itemsets_list = frequent_itemsets.to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "frequent_itemsets_count": len(itemsets_list),
            "frequent_itemsets": itemsets_list
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/rules', methods=['GET'])
def api_get_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))
        
        df = read_data()
        transactions = create_transactions(df)
        frequent_itemsets = get_frequent_itemsets(transactions, min_support)
        rules = get_association_rules(frequent_itemsets, min_confidence)
        
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
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def api_download_rules():
    try:
        df = read_data()
        transactions = create_transactions(df)
        frequent_itemsets = get_frequent_itemsets(transactions)
        rules = get_association_rules(frequent_itemsets)
        
        temp_file = 'temp_rules.json'
        rules.to_json(temp_file, orient='records')
        
        return send_file(
            temp_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='association_rules.json'
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists('temp_rules.json'):
            os.remove('temp_rules.json')

# ---------------- Main ---------------- #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
