# Product Recommendation System

A content-based product recommendation engine built with **pandas** and **scikit-learn**. It uses **TF-IDF vectorization** and **cosine similarity** to recommend products that are similar to a given product, or to a free-text search query — similar to "customers also viewed" or a smart search bar on an e-commerce site.

## How It Works

1. Each product's `category` and `description` are combined into a single text field.
2. The combined text is vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency), which turns text into numerical vectors that emphasize distinctive words.
3. **Cosine similarity** is computed between all product vectors to measure how alike any two products are.
4. Given a product ID (or a search query), the engine returns the most similar products, ranked by similarity score.

## Project Structure

```
product-recommender/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── products.csv        # sample product catalog
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV loading + validation
│   └── recommender.py       # TF-IDF + cosine similarity engine
├── app.py                   # CLI entry point
└── tests/
    └── test_recommender.py  # unit tests
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/product-recommender.git
cd product-recommender

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Get recommendations based on a product ID:**
```bash
python app.py --id 1
```
```
Because you viewed: SonicPods Pro

  [3] BoomBox Mini (Audio) - score: 0.212
  [2] BassBoost Wired (Audio) - score: 0.184
  [4] StudioSound Headset (Audio) - score: 0.156
```

**Search by free text:**
```bash
python app.py --search "gaming laptop with graphics card"
```

**Change the number of recommendations:**
```bash
python app.py --id 1 --top 5
```

**Use your own catalog:**
```bash
python app.py --id 1 --catalog path/to/your_products.csv
```

## Running Tests

```bash
pytest tests/
```

## Possible Extensions

- Swap the CSV for a real database (PostgreSQL/MongoDB)
- Add a Flask/FastAPI REST API on top of `ProductRecommender`
- Add collaborative filtering (based on user purchase history) alongside content-based filtering
- Build a simple React frontend for browsing recommendations
- Deploy as a live demo on Render/Railway

## Tech Stack

- Python 3.9+
- pandas
- scikit-learn
- pytest
