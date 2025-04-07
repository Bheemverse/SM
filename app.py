from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_rules(min_support=0.01, min_confidence=0.3):
    # Check if file exists
    if not os.path.exists('supermarket_sales.xlsx'):
        raise FileNotFoundError("File 'supermarket_sales.xlsx' not found.")

    # Read the Excel file
    df = pd.read_excel('supermarket_sales.xlsx')
    
    # Create transactions
    transactions = df.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values('confidence', ascending=False)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/rules',
            '/api/download/rules'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        # Get parameters from query string
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))
        
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
        
        return jsonify({
            'status': 'success',
            'rules_count': len(rules_list),
            'rules': rules_list
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
        
        # Save rules to temporary JSON file
        temp_file = 'temp_rules.json'
        rules.to_json(temp_file, orient='records')
        
        # Send the file
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
        # Clean up: delete temp file after sending
        if os.path.exists('temp_rules.json'):
            try:
                os.remove('temp_rules.json')
            except:
                pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
