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

def generate_rules(min_support=None, min_confidence=None):
    try:
        # Updated file path with new filename
        file_path = 'dummy_supermarket_sales (2).xlsx'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        # Ensure there is no 'Quantity' column in the dataset (if it doesn't exist, this will be ignored)
        if 'Quantity' in df.columns:
            df = df.drop(columns=['Quantity'])

        # Check the basic info about the dataset
        logging.info(f"Number of unique invoices: {df['Invoice ID'].nunique()}")
        logging.info(f"Number of unique products: {df['Product'].nunique()}")

        # Create transactions (using 'Invoice ID' and 'Product' only)
        transactions = df.groupby(['Invoice ID', 'Product']).size().unstack().fillna(0)
        transactions = (transactions > 0).astype(int)

        # Check if there are enough transactions
        if transactions.shape[0] == 0 or transactions.shape[1] == 0:
            raise ValueError("No valid transactions were found after grouping.")

        # Dynamically calculate min_support if not provided
        if min_support is None:
            avg_product_freq = transactions.sum(axis=0).mean()
            total_transactions = len(transactions)
            min_support = max(0.005, avg_product_freq / total_transactions * 0.1)  # Adjusting for lower threshold

        # Generate frequent itemsets
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)

        # Dynamically choose min_confidence if not provided
        if min_confidence is None:
            min_confidence = 0.1  # Setting a lower default confidence

        # Generate rules
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
            '/api/frequent_products'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        # Get parameters
        min_support = request.args.get('min_support', None)
        min_confidence = request.args.get('min_confidence', None)

        # Convert to float if provided
        min_support = float(min_support) if min_support is not None else None
        min_confidence = float(min_confidence) if min_confidence is not None else None

        # Validate if provided
        if min_support is not None and not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if min_confidence is not None and not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")

        # Generate rules
        rules = generate_rules(min_support, min_confidence)

        # Convert rules to dictionary
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

        # Generate table-like string
        table_data = "Antecedents                        | Consequents   | Support | Confidence | Lift\n"
        table_data += "-" * 90 + "\n"

        for rule in rules_list:
            antecedents = ', '.join(rule['antecedents'])
            consequents = ', '.join(rule['consequents'])
            table_data += f"{antecedents:<35} | {consequents:<12} | {rule['support']:<7.4f} | {rule['confidence']:<10.4f} | {rule['lift']:<5.4f}\n"

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
        # Generate rules with default thresholds
        rules = generate_rules()

        # Save to temp file
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
        # Read data
        file_path = 'dummy_supermarket_sales (2).xlsx'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        # Calculate product frequency
        product_frequency = df['Product'].value_counts()

        # If there are sales or other relevant columns, calculate total sales per product
        if 'Total' in df.columns:  # Assuming 'Total' is a column for sales amount
            product_sales = df.groupby('Product')['Total'].sum()

        # Prepare product info (Frequency and Sales if available)
        product_info = []
        for product in product_frequency.index:
            product_data = {
                'product': product,
                'frequency': int(product_frequency[product])  # Convert to native int
            }
            if 'Total' in df.columns:
                product_data['total_sales'] = float(product_sales.get(product, 0))  # Convert to native float
            
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
