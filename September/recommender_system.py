import numpy as np

print("=" * 50)
print("COLLABORATIVE FILTERING — FROM SCRATCH")
print("=" * 50)

# User-Movie rating matrix
# Rows = users, Columns = movies
# 0 = not rated yet
ratings = np.array([
    [5, 4, 0, 1, 0],
    [0, 3, 5, 0, 4],
    [4, 0, 4, 3, 0],
    [1, 0, 0, 5, 3],
    [0, 2, 4, 0, 5],
])

movies = ['Inception', 'Avengers', 'The Matrix',
          'Notebook', 'Titanic']
users  = ['Ali', 'Sara', 'Mehr', 'Ahmed', 'Fatima']

print("\nRating Matrix (0 = not rated):")
print(f"{'':8}", end="")
for m in movies: print(f"{m:12}", end="")
print()
for i, user in enumerate(users):
    print(f"{user:8}", end="")
    for r in ratings[i]:
        print(f"{'?' if r==0 else r:12}", end="")
    print()

# Simple user-based collaborative filtering
# For a target user find most similar user
# Recommend what similar user rated highly

def cosine_similarity(v1, v2):
    mask = (v1 > 0) & (v2 > 0)
    if mask.sum() == 0:
        return 0
    v1m, v2m = v1[mask], v2[mask]
    return np.dot(v1m, v2m) / (np.linalg.norm(v1m) * np.linalg.norm(v2m))

def recommend_for_user(target_idx, ratings, movies, top_n=2):
    target = ratings[target_idx]
    similarities = []
    for i in range(len(ratings)):
        if i == target_idx: continue
        sim = cosine_similarity(target, ratings[i])
        similarities.append((i, sim))
    similarities.sort(key=lambda x: -x[1])
    most_similar_idx, sim_score = similarities[0]
    print(f"\nMost similar to {users[target_idx]}: "
          f"{users[most_similar_idx]} (similarity={sim_score:.3f})")
    similar_user = ratings[most_similar_idx]
    recommendations = []
    for j, (rating, movie) in enumerate(zip(similar_user, movies)):
        if target[j] == 0 and rating > 3:
            recommendations.append((movie, rating))
    recommendations.sort(key=lambda x: -x[1])
    print(f"Recommended movies for {users[target_idx]}:")
    for movie, rating in recommendations[:top_n]:
        print(f"  {movie} (similar user rated: {rating})")

recommend_for_user(0, ratings, movies)
recommend_for_user(2, ratings, movies)

print("\nKEY CONCEPTS:")
print("Collaborative filtering: similar users → similar taste")
print("Cosine similarity: measures direction not magnitude")
print("Cold start problem: new users have no ratings")
print("Content based: use item features, no other users needed")