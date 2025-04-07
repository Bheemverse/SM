from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_rules(min_support=0.01, min_confidence=0.3):
    try:
        # Check if file exists
        if not os.path.exists('supermarket_sales.xlsx'):
            raise FileNotFoundError("File 'supermarket_sales.xlsx' not found.")

        # Read the Excel file
        df = pd.read_excel('supermarket_sales.xlsx')
        if df.empty:
            raise ValueError("The Excel file is empty")
            
        # Create transactions
        transactions = df.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
        transactions = (transactions > 0).astype(int)
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
        
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
            '/api/download/rules'
        ]
    })

@app.route('/api/rules', methods=['GET'])
def get_rules():
    try:
        # Get and validate parameters
        min_support = float(request.args.get('min_support', 0.01))
        min_confidence = float(request.args.get('min_confidence', 0.3))
        
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if not (0 < min_confidence <= 1):
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
        
        # Generate table-like string for rules
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
        host='0.0.0.0',  # Makes the server publicly available
        debug=True,      # Enable debug mode for development
        port=5000
    )
