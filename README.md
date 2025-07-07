**Glassmorphic Movie Explorer** is a smart movie recommendation system that helps users discover relevant films based on genre preferences, keyword queries (e.g., *"romance"*, *"crime"*, *"space"*), and content similarity. Built with a focus on both intelligent retrieval and sleek user experience, the system mimics modern streaming platforms that personalize content using data-driven insights.

The core system blends traditional Information Retrieval (IR) methods with Machine Learning (ML) to provide ranked movie suggestions. The backend handles text preprocessing, BM25 scoring, and TF-IDF vectorization to compute initial relevance. Further, we enhance recommendation quality using ML models trained on user behavior data.

Users interact via a glassmorphic web interface that supports genre selection, keyword input, result customization, and sort-by filters (e.g., rating, year, ML relevance).

**Key Features:**

* Hybrid recommendation using IR + ML ranking
* Genre and keyword-based filtering
* Real-time score computation and relevance sorting
* Lightweight Flask backend with interactive HTML/CSS frontend

**Technologies Used:**

* Python, Flask, HTML/CSS/JS
* Libraries: `scikit-learn`, `xgboost`, `rank_bm25`, `nltk`, `pandas`, `TfidfVectorizer`
* Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
* Metrics: nDCG, Precision\@K, Recall

**Authours:**
* Arnav Sharma
* Katya Chadha
* Amisha Mittal
