from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

class RecommendationSystem:
    def __init__(self):
        self.data = None
        self.algorithm = None
        self.model = None

    def load_data(self, data, features, algorithm='content-based'):
        """Load and preprocess the dataset."""
        self.data = pd.DataFrame(data)
        self.algorithm = algorithm
        self._initialize_model(features)

    def _initialize_model(self, features):
        """Initialize the recommendation model based on the selected algorithm."""
        if self.algorithm == 'content-based':
            self.model = ContentBasedModel(features)
        elif self.algorithm == 'collaborative-filtering':
            self.model = CollaborativeFilteringModel()
        elif self.algorithm == 'hybrid':
            self.model = HybridModel(features)
        else:
            raise ValueError("Invalid algorithm specified.")

    def get_recommendations(self, item_id, top_n=10):
        """Get top N recommendations."""
        if self.model is None:
            return []

        return self.model.get_recommendations(item_id, top_n)

class ContentBasedModel:
    def __init__(self, features):
        self.features = features
        self.cosine_sim = None

    def _compute_similarity(self):
        """Compute the cosine similarity matrix."""
        tfidf = TfidfVectorizer(stop_words='english')
        feature_texts = self.data[self.features].apply(lambda x: ' '.join(x), axis=1)
        tfidf_matrix = tfidf.fit_transform(feature_texts)
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, item_id, top_n):
        """Get top N recommendations based on item ID."""
        if self.cosine_sim is None:
            self._compute_similarity()

        idx = self.data[self.data['id'] == item_id].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        item_indices = [i[0] for i in sim_scores]
        return self.data.iloc[item_indices].to_dict('records')

class CollaborativeFilteringModel:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None

    def _build_user_item_matrix(self):
        """Build the user-item interaction matrix."""
        # Create the user-item interaction matrix
        self.user_item_matrix = self.data.pivot_table(index='idpers', columns='idprod', values='note', fill_value=0)

    def _compute_user_similarity(self):
        """Compute the user similarity matrix."""
        self.user_similarity_matrix = self.user_item_matrix.T.corr(method='pearson')

    def get_recommendations(self, user_id, top_n):
        """Get top N recommendations for a given user."""
        if self.user_item_matrix is None:
            self._build_user_item_matrix()

        if self.user_similarity_matrix is None:
            self._compute_user_similarity()

        # Get the user's past interactions
        user_history = self.user_item_matrix.loc[user_id]

        # Calculate weighted sum of similarities for each item
        item_scores = self.user_similarity_matrix[user_id].mul(user_history).sum(axis=1)

        # Sort the items by score and return the top N
        recommendations = item_scores.sort_values(ascending=False).head(top_n).index.tolist()
        return self.data.loc[self.data['idprod'].isin(recommendations)].to_dict('records')

class HybridModel:
    def __init__(self, features):
        self.content_model = ContentBasedModel(features)
        self.collaborative_model = CollaborativeFilteringModel()

    def load_data(self, data, features):
        self.content_model.load_data(data, features)
        self.collaborative_model.load_data(data)

    def get_recommendations(self, user_id, item_id, top_n):
        content_recommendations = self.content_model.get_recommendations(item_id, top_n // 2)
        collaborative_recommendations = self.collaborative_model.get_recommendations(user_id, top_n // 2)
        return content_recommendations + collaborative_recommendations


rec_sys = RecommendationSystem()

@app.route('/load_data', methods=['POST'])
def load_data():
    data = request.json
    dataset = data['dataset']
    features = data['features']
    algorithm = data.get('algorithm', 'content-based')  # Default to content-based
    rec_sys.load_data(dataset, features, algorithm)
    return jsonify({"message": "Data loaded successfully."})

@app.route('/recommend', methods=['GET'])
def recommend():
    item_id = int(request.args.get('item_id'))
    top_n = int(request.args.get('top_n', 10))
    recommendations = rec_sys.get_recommendations(item_id, top_n)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
