"""
app.py
Command-line demo for the Product Recommendation System.

Usage:
    python app.py --id 1
    python app.py --search "bluetooth headphones"
"""

import argparse
from src.data_loader import load_products, get_default_path
from src.recommender import ProductRecommender


def print_results(df):
    if df.empty:
        print("No recommendations found.")
        return
    for _, row in df.iterrows():
        print(f"  [{row['id']}] {row['name']} ({row['category']}) "
              f"- score: {row['similarity_score']}")


def main():
    parser = argparse.ArgumentParser(description="Product Recommendation System")
    parser.add_argument("--id", type=int, help="Product ID to get recommendations for")
    parser.add_argument("--search", type=str, help="Free-text search query")
    parser.add_argument("--top", type=int, default=3, help="Number of recommendations")
    parser.add_argument("--catalog", type=str, default=None, help="Path to products CSV")
    args = parser.parse_args()

    csv_path = args.catalog or get_default_path()
    products = load_products(csv_path)
    recommender = ProductRecommender(products)

    if args.id is not None:
        product_row = products[products["id"] == args.id]
        if product_row.empty:
            print(f"Product ID {args.id} not found.")
            return
        print(f"Because you viewed: {product_row.iloc[0]['name']}\n")
        results = recommender.recommend(args.id, top_n=args.top)
        print_results(results)

    elif args.search:
        print(f"Search results for: '{args.search}'\n")
        results = recommender.recommend_by_text(args.search, top_n=args.top)
        print_results(results)

    else:
        print("Provide either --id <product_id> or --search '<query>'.")
        print("Example: python app.py --id 1")
        print("Example: python app.py --search 'bluetooth headphones'")


if __name__ == "__main__":
    main()
