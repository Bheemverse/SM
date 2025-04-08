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

@app.post("/api/rules")
async def get_association_rules(file: UploadFile = File(...)):
    try:
        # Your existing code to read and process the file...
        df = pd.read_excel(file.file)
        transactions = df.values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit_transform(transactions)
        df_tf = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_tf, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        # Convert the rules DataFrame into clean JSON
        rules_table_json = rules.reset_index().to_dict(orient="records")
        
        return {
            "rules_count": len(rules),
            "rules_table": rules_table_json
        }
    
    except Exception as e:
        logging.error(f"Error while generating association rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate association rules.")
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
