# === TF-IDF WordCloud Visualization ===
wc = WordCloud(width=800, height=400, background_color="white")
wc.generate_from_frequencies(tfidf_scores)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title(f"Group {gid} WordCloud (TF-IDF)")
plt.tight_layout()
plt.show()


# === Bar Chart Visualization of Top-20 TF-IDF ===
if top20:
    words, scores = zip(*top20)
else:
    words, scores = [], []

plt.figure(figsize=(10, 5))
plt.bar(words, scores)
plt.xticks(rotation=45, ha="right")
plt.title(f"Group {gid} Top-20 (TF-IDF)")
plt.ylabel("TF-IDF Score")
plt.tight_layout()
plt.show()