from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RecommendationSystem:
    def __init__(self):
        self.data = None
        self.algorithm = None
        self.model = None

    def load_data(self, data, features, algorithm='content-based'):
        """Load and preprocess the dataset."""
        self.data = pd.DataFrame(data)
        logger.debug("Dataset loaded:")
        logger.debug(self.data.head())
        self.algorithm = algorithm
        self._initialize_model(features)

    def _initialize_model(self, features):
        """Initialize the recommendation model based on the selected algorithm."""
        logger.debug(f"Initializing {self.algorithm} model...")
        if self.algorithm == 'content-based':
            self.model = ContentBasedModel(self.data, features)
        elif self.algorithm == 'collaborative-filtering':
            self.model = CollaborativeFilteringModel(self.data)
        elif self.algorithm == 'hybrid':
            self.model = HybridModel(self.data, features)
        else:
            raise ValueError("Invalid algorithm specified.")

    def get_recommendations(self, item_id, user_id=None, top_n=10):
        """Get top N recommendations."""
        if self.model is None:
            return []

        return self.model.get_recommendations(item_id, user_id, top_n)

class ContentBasedModel:
    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.cosine_sim = None

    def _compute_similarity(self):
        """Compute the cosine similarity matrix."""
        tfidf = TfidfVectorizer(stop_words='english')
        feature_texts = self.data[self.features].apply(lambda x: ' '.join(map(str, x)), axis=1)
        tfidf_matrix = tfidf.fit_transform(feature_texts)
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, item_id, user_id=None, top_n=10):
        """Get top N recommendations based on item ID."""
        if self.cosine_sim is None:
            logger.debug("Computing the cosine similarity matrix...")
            self._compute_similarity()

        idx = self.data[self.data['id'] == item_id].index[0]
        logger.debug(f"Getting recommendations for item with ID: {item_id}")
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        item_indices = [i[0] for i in sim_scores]
        return self.data.iloc[item_indices].to_dict('records')

class CollaborativeFilteringModel:
    def __init__(self, data):
        self.data = data
        self.user_item_matrix = None
        self.user_similarity_matrix = None

    def load_data(self, data):
        self.data = data
        self._build_user_item_matrix()

    def _build_user_item_matrix(self):
        """Build the user-item interaction matrix."""
        self.user_item_matrix = self.data.pivot(index='idpers', columns='idprod', values='note').fillna(0)

    def _compute_user_similarity(self):
        """Compute the user similarity matrix."""
        self.user_similarity_matrix = self.user_item_matrix.T.corr(method='pearson')

    def get_recommendations(self, item_id=None, user_id=None, top_n=10):
        """Get top N recommendations for a given user."""
        if self.user_item_matrix is None:
            self._build_user_item_matrix()

        if self.user_similarity_matrix is None:
            self._compute_user_similarity()

        if user_id is None:
            raise ValueError("User ID must be provided for collaborative filtering recommendations.")

        # Get the user's past interactions
        user_history = self.user_item_matrix.loc[user_id]

        # Calculate weighted sum of similarities for each item
        weighted_sums = self.user_similarity_matrix[user_id].dot(self.user_item_matrix).div(self.user_similarity_matrix[user_id].sum())
        weighted_sums = weighted_sums[user_history == 0]  # Exclude items the user has already interacted with
        recommendations = weighted_sums.sort_values(ascending=False).head(top_n)

        return self.data[self.data['idprod'].isin(recommendations.index)].to_dict('records')

class HybridModel:
    def __init__(self, data, features):
        self.content_model = ContentBasedModel(data, features)
        self.collaborative_model = CollaborativeFilteringModel(data)

    def get_recommendations(self, item_id, user_id, top_n):
        content_recommendations = self.content_model.get_recommendations(item_id, top_n=top_n)
        collaborative_recommendations = self.collaborative_model.get_recommendations(user_id=user_id, top_n=top_n)
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
    try:
        item_id = int(request.args.get('item_id'))
        user_id = request.args.get('user_id', type=int)
        top_n = int(request.args.get('top_n', 10))
        recommendations = rec_sys.get_recommendations(item_id, user_id, top_n)
        if recommendations:
            return jsonify(recommendations)
        else:
            return jsonify({'success': False, 'error': 'No recommendations found', 'reco': recommendations}), 404
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid item ID or top_n parameter'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dataset', methods=['GET'])
def get_dataset():
    if rec_sys.data is None:
        return jsonify({'error': 'No dataset loaded'}), 404
    else:
        return jsonify(rec_sys.data.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
