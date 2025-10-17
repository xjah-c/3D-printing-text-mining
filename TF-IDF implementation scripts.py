# ===== TF-IDF =====
all_terms = set().union(*[set(c.keys()) for c in group_counters.values()])
n_groups = len(group_counters)

df_term = {t: sum(1 for gid, cnt in group_counters.items() if t in cnt and cnt[t] > 0)
           for t in all_terms}

idf = {t: (log((1 + n_groups) / (1 + df_term[t])) + 1.0)
       for t in all_terms}  # smoothed IDF

results_tfidf = []
for gid, cnt in group_counters.items():
    tfidf_scores = {t: cnt[t] * idf[t] for t in cnt}
    top20 = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    for rank, (term, score) in enumerate(top20, 1):
        results_tfidf.append({
            "group_id": gid,
            "rank": rank,
            "word": term,
            "tf": cnt[term],
            "idf": round(idf[term], 4),
            "tfidf": round(score, 4),
        })