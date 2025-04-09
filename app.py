from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

file_path = 'dummy_supermarket_sales (2).xlsx'

def load_transaction_data():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("The Excel file is empty.")

    if 'Quantity' in df.columns:
        df = df.drop(columns=['Quantity'])

    transaction_data = df.groupby(['Invoice ID', 'Product']).size().unstack(fill_value=0)
    transaction_data = (transaction_data > 0).astype(int)
    return df, transaction_data

def generate_rules(min_support=0.01, min_confidence=0.2, min_lift=1.0):
    try:
        df, transaction_data = load_transaction_data()
        invoice_ids = transaction_data.index.tolist()

        if transaction_data.empty:
            raise ValueError("No valid transactions found.")

        frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            raise ValueError("No frequent itemsets found. Try lowering min_support.")

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            raise ValueError("No association rules found. Try lowering min_confidence.")

        # Apply min_lift filter
        rules = rules[rules['lift'] >= min_lift]

        if rules.empty:
            raise ValueError("No rules found after applying lift filter. Try lowering min_lift.")

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
            '/api/rules/filter',
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
        rules_df = generate_rules()

        grouped_rules = {}
        for _, row in rules_df.iterrows():
            invoice_id = row['invoice_id']
            rule = {
                "antecedents": row['antecedents'],
                "consequents": row['consequents'],
                "support": row['support'],
                "confidence": row['confidence'],
                "lift": row['lift']
            }
            grouped_rules.setdefault(invoice_id, []).append(rule)

        return jsonify({
            "status": "success",
            "rules_count": len(rules_df),
            "rules_by_invoice": grouped_rules
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/filter', methods=['GET'])
def filter_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.2))
        min_lift = float(request.args.get('min_lift', 1.0))

        rules_df = generate_rules(min_support, min_confidence, min_lift)

        rules_list = []
        for _, row in rules_df.iterrows():
            rules_list.append({
                'invoice_id': row['invoice_id'],
                'antecedents': row['antecedents'],
                'consequents': row['consequents'],
                'support': row['support'],
                'confidence': row['confidence'],
                'lift': row['lift']
            })

        return jsonify({
            'status': 'success',
            'filtered_rules_count': len(rules_list),
            'filtered_rules': rules_list
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        rules_df = generate_rules()
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(x))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(x))
        temp_csv = 'association_rules.csv'
        rules_df.to_csv(temp_csv, index=False)

        return send_file(
            temp_csv,
            mimetype='text/csv',
            as_attachment=True,
            download_name='association_rules.csv'
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/frequent_products', methods=['GET'])
def frequent_products():
    try:
        df = pd.read_excel(file_path)
        product_names = df['Product'].value_counts().index.tolist()
        return jsonify(product_names)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        df = pd.read_excel(file_path)
        products = df['Product'].dropna().unique().tolist()
        return jsonify({'status': 'success', 'products': products, 'count': len(products)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_antecedent', methods=['GET'])
def rules_by_antecedent():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({'status': 'error', 'message': 'Product parameter is required'}), 400

        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='antecedent')

        unique_consequents = set()
        for cons in filtered_df['consequents']:
            unique_consequents.update(cons)

        return jsonify(sorted(unique_consequents))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_consequent', methods=['GET'])
def rules_by_consequent():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({'status': 'error', 'message': 'Product parameter is required'}), 400

        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='consequent')

        unique_antecedents = set()
        for ant in filtered_df['antecedents']:
            unique_antecedents.update(ant)

        return jsonify(sorted(unique_antecedents))

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rules/by_product', methods=['GET'])
def rules_by_product():
    try:
        product = request.args.get('product')
        if not product:
            return jsonify({'status': 'error', 'message': 'Product parameter is required'}), 400

        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='any')

        related_products = set()
        for _, row in filtered_df.iterrows():
            related_products.update(row['antecedents'])
            related_products.update(row['consequents'])

        related_products.discard(product)

        return jsonify(sorted(related_products))

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
    app.run(host='0.0.0.0', debug=True, port=5000)
