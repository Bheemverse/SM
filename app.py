from fastapi import FastAPI, HTTPException
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load and prepare data function
def load_data():
    if not os.path.exists('dummy_supermarket_sales.xlsx'):  # Check file path
        raise FileNotFoundError("File 'dummy_supermarket_sales.xlsx' not found.")
    df = pd.read_excel('dummy_supermarket_sales.xlsx')  # Read data
    if df.empty:
        raise ValueError("The Excel file is empty")
    return df

# Function to prepare transactions for association rule mining
def prepare_transactions(df):
    transactions = df.groupby(['Invoice ID', 'Product'])['Product'].count().unstack().fillna(0)
    transactions = (transactions > 0).astype(int)
    return transactions

# Generate frequent itemsets and association rules
def generate_frequent_itemsets(transactions, min_support=0.01):
    return apriori(transactions, min_support=min_support, use_colnames=True)

def generate_rules(frequent_itemsets, min_confidence=0.3):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    return rules.sort_values('confidence', ascending=False)

# API Route to get all products
@app.get("/api/products")
def get_products():
    try:
        df = load_data()
        products = sorted(df['Product'].unique().tolist())
        return {"status": "success", "products_count": len(products), "products": products}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# API Route to get association rules in table format
@app.get("/api/rules")
def get_rules(min_support: float = 0.01, min_confidence: float = 0.3):
    try:
        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)
        
        # Format rules into a plain-text table format
        table_data = "Antecedents                        | Consequents   | Support | Confidence | Lift\n"
        table_data += "-" * 90 + "\n"
        
        for _, row in rules.iterrows():
            antecedents = ', '.join(row['antecedents'])
            consequents = ', '.join(row['consequents'])
            table_data += f"{antecedents:<35} | {consequents:<12} | {row['support']:<7.4f} | {row['confidence']:<10.4f} | {row['lift']:<5.4f}\n"

        return {"status": "success", "rules_table": table_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# API Route to get frequent products based on minimum support
@app.get("/api/frequent-products")
def get_frequent_products(min_support: float = 0.01):
    try:
        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        
        # Format frequent products into a plain-text table format
        table_data = "Products                         | Support\n"
        table_data += "-" * 50 + "\n"
        
        for _, row in frequent_itemsets.iterrows():
            products = ', '.join(row['itemsets'])
            table_data += f"{products:<30} | {row['support']:<7.4f}\n"

        return {"status": "success", "frequent_products_table": table_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# API Route to recommend products based on selected products
@app.post("/api/associate-products")
def associate_products(products: list, min_support: float = 0.01, min_confidence: float = 0.3):
    try:
        if not products:
            raise HTTPException(status_code=400, detail="Product list is required")

        df = load_data()
        transactions = prepare_transactions(df)
        frequent_itemsets = generate_frequent_itemsets(transactions, min_support)
        rules = generate_rules(frequent_itemsets, min_confidence)

        recommended_products = []

        for _, row in rules.iterrows():
            antecedents = set(row['antecedents'])
            consequents = set(row['consequents'])

            if antecedents.issubset(set(products)):
                recommended_products.append({
                    'based_on': list(antecedents),
                    'recommend': list(consequents),
                    'support': float(row['support']),
                    'confidence': float(row['confidence']),
                    'lift': float(row['lift'])
                })

        if not recommended_products:
            return {"status": "success", "message": "No recommendations found based on selected products", "recommendations": []}

        return {"status": "success", "recommendations": recommended_products}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
