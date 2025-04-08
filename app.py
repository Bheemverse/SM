from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to generate rules
def generate_rules(min_support: float = None, min_confidence: float = None):
    try:
        file_path = 'dummy_supermarket_sales (2).xlsx'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        if 'Quantity' in df.columns:
            df = df.drop(columns=['Quantity'])

        logging.info(f"Number of unique invoices: {df['Invoice ID'].nunique()}")
        logging.info(f"Number of unique products: {df['Product'].nunique()}")

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

@app.get("/")
async def home():
    return JSONResponse({
        'message': 'Welcome to the Association Rules API!',
        'available_endpoints': [
            '/api/rules',
            '/api/download/rules',
            '/api/frequent_products'
        ]
    })

@app.get("/api/rules")
async def get_rules(min_support: float = Query(default=None), min_confidence: float = Query(default=None)):
    try:
        if min_support is not None and not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if min_confidence is not None and not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")

        rules = generate_rules(min_support, min_confidence)

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

        # Table-like string
        table_data = "Antecedents                        | Consequents   | Support | Confidence | Lift\n"
        table_data += "-" * 90 + "\n"

        for rule in rules_list:
            antecedents = ', '.join(rule['antecedents'])
            consequents = ', '.join(rule['consequents'])
            table_data += f"{antecedents:<35} | {consequents:<12} | {rule['support']:<7.4f} | {rule['confidence']:<10.4f} | {rule['lift']:<5.4f}\n"

        return JSONResponse({
            "status": "success",
            "rules_count": len(rules_list),
            "rules_table": table_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/rules")
async def download_rules():
    try:
        rules = generate_rules()
        temp_file = 'temp_rules.json'
        rules.to_json(temp_file, orient='records')

        return FileResponse(
            temp_file,
            media_type='application/json',
            filename='association_rules.json'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/frequent_products")
async def frequent_products():
    try:
        file_path = 'dummy_supermarket_sales (2).xlsx'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")

        df = pd.read_excel(file_path)
        if df.empty:
            raise ValueError("The Excel file is empty")

        product_frequency = df['Product'].value_counts()

        if 'Total' in df.columns:
            product_sales = df.groupby('Product')['Total'].sum()

        product_info = []
        for product in product_frequency.index:
            product_data = {
                'product': product,
                'frequency': int(product_frequency[product])
            }
            if 'Total' in df.columns:
                product_data['total_sales'] = float(product_sales.get(product, 0))
            product_info.append(product_data)

        return JSONResponse({
            'status': 'success',
            'frequent_products': product_info
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found_error(request, exc):
    return JSONResponse(
        status_code=404,
        content={'status': 'error', 'message': 'Endpoint not found'}
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    logging.error(f"Server Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={'status': 'error', 'message': 'Internal server error'}
    )
