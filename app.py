from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging
import random
import string
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Product list
PRODUCTS = [
    'Milk', 'Bread', 'Butter', 'Eggs', 'Cheese',
    'Apple', 'Banana', 'Orange', 'Tomato', 'Potato',
    'Rice', 'Sugar', 'Flour', 'Salt', 'Pepper',
    'Chicken', 'Beef', 'Fish', 'Pasta', 'Cereal'
]

def generate_dummy_data():
    """Generate dummy supermarket_sales.xlsx if not exists."""
    if not os.path.exists('supermarket_sales.xlsx'):
        logging.info("Generating dummy dataset...")
        data = []
        for _ in range(500):  # 500 transactions
            invoice_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            num_items = random.randint(1, 5)
            chosen_products = random.sample(PRODUCTS, num_items)
            for product in chosen_products:
                quantity = random.randint(1, 5)
                data.append([invoice_id, product, quantity])
        
        df = pd.DataFrame(data, columns=['Invoice ID', 'Product line', 'Quantity'])
        df.to_excel('supermarket_sales.xlsx', index=False)
        logging.info("Dummy dataset generated successfully.")

def generate_rules(min_support=0.01, min_confidence=0.3):
    try:
        generate_dummy_data()  # Ensure dataset is available

        # Read the Excel file
        df = pd.read_excel('supermarket_sales.xlsx')
        if df.empty:
            raise ValueError("The Excel file is empty")
        
        # Create transactions
        transactions = df.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
        transactions = (transactions > 0).astype(int)

        # Generate frequent itemsets
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
        print("\n=== Frequent Itemsets ===")
        print(frequent_itemsets)

        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        print("\n=== Association Rules ===")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        return frequent_itemsets, rules.sort_values('confidence', ascending=False)
    
    except Exception as e:
        raise Exception(f"Error generating rules: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/rules',
            '/api/frequent_items',
            '/api/products',
            '/api/download/rules',
            '/api/download/products'
        ]
    })

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        return jsonify({
            "status": "success",
            "products_count": len(PRODUCTS),
            "products": PRODUCTS
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))
        
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")
        
        _, rules = generate_rules(min_support, min_confidence)
        
        rules_list = []
        for _, row in rules.iterrows():
            rule_dict = {
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': float(row['support']),
                'confidence': float(row['confidence']),
                'lift': float(row['lift'])
            }
            rules_list.append(rule_dict)
        
        return jsonify({
            "status": "success",
            "rules_count": len(rules_list),
            "rules": rules_list
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frequent_items', methods=['GET'])
def get_frequent_items():
    try:
        min_support = float(request.args.get('min_support', 0.01))
        
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        
        frequent_itemsets, _ = generate_rules(min_support)
        
        frequent_list = []
        for _, row in frequent_itemsets.iterrows():
            item_dict = {
                'items': list(row['itemsets']),
                'support': float(row['support'])
            }
            frequent_list.append(item_dict)
        
        return jsonify({
            "status": "success",
            "frequent_items_count": len(frequent_list),
            "frequent_items": frequent_list
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/rules', methods=['GET'])
def download_rules():
    try:
        _, rules = generate_rules()
        
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
    finally:
        if os.path.exists('temp_rules.json'):
            try:
                os.remove('temp_rules.json')
            except:
                pass

@app.route('/api/download/products', methods=['GET'])
def download_products():
    try:
        temp_file = 'temp_products.json'
        with open(temp_file, 'w') as f:
            json.dump(PRODUCTS, f, indent=4)
        
        return send_file(
            temp_file,
            mimetype='application/json',
            as_attachment=True,
            download_name='products_list.json'
        )
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if os.path.exists('temp_products.json'):
            try:
                os.remove('temp_products.json')
            except:
                pass

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
