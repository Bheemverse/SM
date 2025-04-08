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

        transaction_data = df.groupby(['Invoice ID', 'Product']).size().unstack().fillna(0)
        transaction_data = (transaction_data > 0).astype(int)
        invoice_ids = transaction_data.index.tolist()

        if transaction_data.shape[0] == 0 or transaction_data.shape[1] == 0:
            raise ValueError("No valid transactions were found after grouping.")

        if min_support is None:
            avg_product_freq = transaction_data.sum(axis=0).mean()
            total_transactions = len(transaction_data)
            min_support = max(0.005, avg_product_freq / total_transactions * 0.1)

        frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

        if min_confidence is None:
            min_confidence = 0.1

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        rule_records = []
        for _, row in rules.iterrows():
            ant = list(row['antecedents'])
            cons = list(row['consequents'])

            for invoice_id in invoice_ids:
                products = set(df[df['Invoice ID'] == invoice_id]['Product'])
                if set(ant).issubset(products):
                    rule_records.append({
                        'invoice_id': invoice_id,
                        'antecedents': ant,
                        'consequents': cons,
                        'support': float(row['support']),
                        'confidence': float(row['confidence']),
                        'lift': float(row['lift'])
                    })

        return pd.DataFrame(rule_records)

    except Exception as e:
        raise Exception(f"Error generating rules: {str(e)}")

def filter_rules_by_product(rules_df, product, role='any'):
    filtered = []
    for _, row in rules_df.iterrows():
        ant = row['antecedents']
        cons = row['consequents']

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
    try:@app.route('/api/rules/products', methods=['GET'])
def get_rule_products():
    try:
        rules_df = generate_rules()

        unique_products = set()

        for _, row in rules_df.iterrows():
            unique_products.update(row['antecedents'])
            unique_products.update(row['consequents'])

        return jsonify(sorted(list(unique_products)))

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/rules/products', methods=['GET'])
def download_rule_products():
    try:
        rules_df = generate_rules()

        unique_products = set()
        for _, row in rules_df.iterrows():
            unique_products.update(row['antecedents'])
            unique_products.update(row['consequents'])

        product_df = pd.DataFrame(sorted(list(unique_products)), columns=['product'])

        temp_csv_file = 'unique_rule_products.csv'
        product_df.to_csv(temp_csv_file, index=False)

        return send_file(
            temp_csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='unique_rule_products.csv'
        )

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/rules/by_antecedent')
def rules_by_antecedent():
    try:
        product = request.args.get('product')
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, 'antecedent')

        # Get unique consequent products
        unique_consequents = set()
        for consequents in filtered_df['consequents']:
            unique_consequents.update(consequents)

        return jsonify(sorted(unique_consequents))  # Sorted optional

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/rules/by_consequent')
def rules_by_consequent():
    try:
        product = request.args.get('product')
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, 'consequent')

        # Extract unique antecedents
        unique_antecedents = set()
        for antecedents in filtered_df['antecedents']:
            unique_antecedents.update(antecedents)

        return jsonify(sorted(unique_antecedents))

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/rules/by_product')
def rules_by_product():
    try:
        product = request.args.get('product')
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, 'any')

        # Collect unique products from both antecedents and consequents
        related_products = set()
        for _, row in filtered_df.iterrows():
            related_products.update(row['antecedents'])
            related_products.update(row['consequents'])

        # Remove the original product (optional, based on your needs)
        related_products.discard(product)

        return jsonify(sorted(related_products))

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/frequent_products', methods=['GET'])
def frequent_products():
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        # Extract product names sorted by frequency
        product_names = df['Product'].value_counts().index.tolist()

        return jsonify(product_names)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
