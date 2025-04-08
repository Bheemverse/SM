from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# File path
file_path = 'dummy_supermarket_sales (2).xlsx'

def generate_rules(min_support: float = None, min_confidence: float = None):
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

def filter_rules_by_product(rules_df, product: str, role: str = 'any'):
    filtered = []
    for _, row in rules_df.iterrows():
        if (role == 'antecedent' and product in row['antecedents']) or \
           (role == 'consequent' and product in row['consequents']) or \
           (role == 'any' and (product in row['antecedents'] or product in row['consequents'])):
            filtered.append(row)
    return pd.DataFrame(filtered)

@app.get("/")
def home():
    return {
        "message": "Welcome to the Association Rules API!",
        "available_endpoints": [
            "/api/rules",
            "/api/download/rules",
            "/api/frequent_products",
            "/api/products",
            "/api/rules/by_antecedent?product=...",
            "/api/rules/by_consequent?product=...",
            "/api/rules/by_product?product=..."
        ]
    }

@app.get("/api/rules")
def get_rules(min_support: float = Query(default=None), min_confidence: float = Query(default=None)):
    try:
        rules_df = generate_rules(min_support, min_confidence)
        return {
            "status": "success",
            "rules_count": len(rules_df),
            "rules": rules_df.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/rules")
def download_rules():
    try:
        rules = generate_rules()
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(x))

        temp_csv_file = 'association_rules.csv'
        rules.to_csv(temp_csv_file, index=False)

        return FileResponse(temp_csv_file, media_type='text/csv', filename='association_rules.csv')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products")
def get_products():
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        products = df['Product'].dropna().unique().tolist()
        return {
            'status': 'success',
            'products': products,
            'count': len(products)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/frequent_products")
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

        return {
            'status': 'success',
            'frequent_products': product_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules/by_antecedent")
def rules_by_antecedent(product: str):
    try:
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='antecedent')
        return filtered_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules/by_consequent")
def rules_by_consequent(product: str):
    try:
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='consequent')
        return filtered_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules/by_product")
def rules_by_product(product: str):
    try:
        rules_df = generate_rules()
        filtered_df = filter_rules_by_product(rules_df, product, role='any')
        return filtered_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True,
        port=5000
    )
