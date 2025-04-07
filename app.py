import pandas as pd
import random
import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from mlxtend.frequent_patterns import apriori, association_rules
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
    if not os.path.exists('dummy_supermarket_sales.xlsx'):  # Changed to new file
        raise FileNotFoundError("File 'dummy_supermarket_sales.xlsx' not found.")
    df = pd.read_excel('dummy_supermarket_sales.xlsx')  # Changed to new file
    if df.empty:
        raise ValueError("The Excel file is empty")
    return df

def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product'])['Product'].count().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

def generate_frequent_itemsets(transactions, min_support=0.01):
    return apriori(transactions, min_support=min_support, use_colnames=True)

def generate_rules(frequent_itemsets, min_confidence=0.3):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    return rules.sort_values('confidence', ascending=False)

# Function to nicely print rules as a table
def print_rules_table(rules):
    if rules.empty:
        print("\nNo rules generated.\n")
        return

    print("\nAssociation Rules:\n")
    print("{:<40} {:<40} {:<10} {:<10} {:<10}".format('Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'))
    print("-" * 120)
    for _, row in rules.iterrows():
        print("{:<40} {:<40} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            ', '.join(row['antecedents']),
            ', '.join(row['consequents']),
            row['support'],
            row['confidence'],
            row['lift']
        ))
    print("\n")

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

        # Print rules in table format
        print_rules_table(rules)

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
def download_rules():
    try:
        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions)
        rules = generate_rules(frequent_itemsets)

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

@app.route('/api/associate-products', methods=['POST'])
def associate_products():
    try:
        data = request.get_json()
        selected_products = data.get('products', [])
        min_support = float(data.get('min_support', 0.01))
        min_confidence = float(data.get('min_confidence', 0.3))

        if not selected_products:
            return jsonify({'status': 'error', 'message': 'Product list is required'}), 400

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)

        recommended_products = []

        for _, row in rules.iterrows():
            antecedents = set(row['antecedents'])
            consequents = set(row['consequents'])

            if antecedents.issubset(set(selected_products)):
                recommended_products.append({
                    'based_on': list(antecedents),
                    'recommend': list(consequents),
                    'support': float(row['support']),
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift'])
                })

        if not recommended_products:
            return jsonify({
                'status': 'success',
                'message': 'No recommendations found based on selected products',
                'recommendations': []
            })

        return jsonify({
            'status': 'success',
            'selected_products': selected_products,
            'recommendations_count': len(recommended_products),
            'recommendations': recommended_products
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/frequent-products', methods=['GET'])
def get_frequent_products():
    try:
        min_support = float(request.args.get('min_support', 0.01))

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)

        frequent_list = []
        for _, row in frequent_itemsets.iterrows():
            frequent_list.append({
                'products': list(row['itemsets']),
                'support': float(row['support'])
            })

        return jsonify({
            'status': 'success',
            'frequent_products_count': len(frequent_list),
            'frequent_products': frequent_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# Run the App
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',  # Makes the server publicly available
        debug=True,      # Enable debug mode for development
        port=5000        # Running on port 5000
    )
