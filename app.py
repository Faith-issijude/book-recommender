from flask import Flask, render_template, request, url_for
import random
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Preprocessed_data.csv")
data.drop(['location', 'city', 'state', 'country', 'Unnamed: 0', 'age', 'publisher', 'img_s', 'img_m', 
            'img_l', 'Summary', 'Language'], axis=1, inplace=True)

# Collaborative Filtering Model (SVD)
reader = Reader(rating_scale=(0, 10))
data_surprise = Dataset.load_from_df(data[['user_id', 'isbn', 'rating']], reader)
trainset = data_surprise.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# TF-IDF Vectorization for book titles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['book_title'])

def get_similar_books(book_title, n=10):
    try:
        # Find similar books based on collaborative filtering
        book_id = data[data['book_title'] == book_title]['isbn'].iloc[0]
        user_id = random.choice(trainset.all_users())
        test_data = [(user_id, book_id, 0) for _ in range(algo.trainset.n_items)]
        predictions = algo.test(test_data)
        sim_books_collab = [pred.iid for pred in predictions]
        
        # Find similar books based on title similarity
        book_idx = data[data['book_title'] == book_title].index[0]
        title_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix)
        sim_books_title = [trainset.to_raw_iid(idx) for idx in title_sim.argsort()[0][-n-1:-1][::-1]]
        
        # Merge and deduplicate recommendations
        recommendations = list(set(sim_books_collab + sim_books_title))
        
        # Return top 10 book titles
        return data.loc[data['isbn'].isin(recommendations[:n]), 'book_title'].tolist()
    except Exception as e:
        return [str(e)]  # Return error message

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            book_title = request.form['book_title']
            recommended_books = get_similar_books(book_title)
            return render_template('recommendations.html', books=recommended_books)
        except Exception as e:
            return render_template('error.html', message=str(e))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
