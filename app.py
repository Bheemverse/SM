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

file_path = 'dummy_supermarket_sales (2).xlsx'

def generate_rules(min_support=0.1, min_confidence=0.5, min_lift=1.0):
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

        if transaction_data.empty:
            raise ValueError("No valid transactions found.")

        frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

        # Apply confidence filtering separately
        rules = rules[rules['confidence'] >= min_confidence]

        rule_records = []
        for _, row in rules.iterrows():
            rule_records.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
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
            '/api/rules/by_antecedent?product=...',
            '/api/rules/by_consequent?product=...',
            '/api/rules/by_product?product=...',
            '/api/products',
            '/api/frequent_products',
            '/api/download/rules',
            '/api/download/rules/products'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules_products_only():
    try:
        # Get query parameters with default values
        min_support = float(request.args.get('min_support', 0.1))
        min_confidence = float(request.args.get('min_confidence', 0.5))
        min_lift = float(request.args.get('min_lift', 1.0))

        rules_df = generate_rules(min_support=min_support, min_confidence=min_confidence)

        # Filter rules based on min_lift
        rules_df = rules_df[rules_df['lift'] >= min_lift]

        # Return the rules
        return jsonify(rules_df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rule_products_only():
    try:
        # Get query parameters with default values
        min_support = float(request.args.get('min_support', 0.1))
        min_confidence = float(request.args.get('min_confidence', 0.5))
        min_lift = float(request.args.get('min_lift', 1.0))

        rules_df = generate_rules(min_support=min_support, min_confidence=min_confidence)

        # Filter rules based on min_lift
        rules_df = rules_df[rules_df['lift'] >= min_lift]

        temp_csv_file = 'rules_filtered.csv'
        rules_df.to_csv(temp_csv_file, index=False)

        return send_file(
            temp_csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='rules_filtered.csv'
        )

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
        unique_antecedents = sorted({item for ant in filtered_df['antecedents'] for item in ant})
        return jsonify(unique_antecedents)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/rules/by_consequent')
def rules_by_consequent():
    try:
        product = request.args.get('product')
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, 'consequent')
        unique_antecedents = sorted({item for ant in filtered_df['antecedents'] for item in ant})
        return jsonify(unique_antecedents)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/rules/by_product')
def rules_by_product():
    try:
        product = request.args.get('product')
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, 'any')
        unique_products = sorted({item for row in filtered_df.itertuples() for item in row.antecedents + row.consequents})
        return jsonify(unique_products)
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
