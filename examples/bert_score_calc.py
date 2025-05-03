from bert_score import score

predictions = [
    "This feature is called the Microbial Stability and Sweetness Synergy Index.",
    "This feature is called the Non-Alcoholic Body and Mineral Content Index.",
    "This feature is called the Acidity Structure and Freshness Index.",
    "This feature is called the Oxidative Stability and Mouthfeel Balance Index."
]


references = [
    "This feature moderately contributes to wine classification. It is strongly positively"
    " correlated with total and free sulfur dioxide and residual sugar, and strongly negatively"
    " correlated with volatile acidity and chlorides. These relationships suggest it reflects"
    " combined variations in preservation agents and acidity levels.",

    "This feature has the highest SHAP importance and is strongly correlated with increased density,"
    " residual sugar, and fixed acidity, while being negatively correlated with alcohol. It captures"
    " a trade-off pattern between dense, sweet, low-alcohol profiles and more alcoholic ones, which"
    " is predictive in classifying wine quality.",

    "This feature is a strong contributor to the model and is positively correlated with citric acid,"
    " fixed acidity, and alcohol, while negatively correlated with volatile acidity and pH. It "
    "represents a contrast between structured acidic-alcoholic profiles and more volatile or less"
    " acidic ones.",

    "This feature has moderate importance and shows positive correlations with sulphates, pH, and"
    " free sulfur dioxide. It reflects underlying variations related to chemical stability and "
    "preservative content."
]


# Compute BERTScore
P, R, F1 = score(
    predictions,
    references,
    lang="en",
    model_type="bert-base-uncased",
    rescale_with_baseline=True
)

# Show F1 scores (the main metric people report)
for i in range(len(predictions)):
    print(f"Label {i+1}:")
    print(f"Prediction: {predictions[i]}")
    print(f"Reference: {references[i]}")
    print(f"BERTScore F1: {F1[i].item():.4f}")
    print("------")

from transformers import pipeline
nli = pipeline("text-classification", model="roberta-large-mnli")

for i in range(len(predictions)):
    result = nli(f"{predictions[i]} </s> {references[i]}")
    print(f"Label {i+1}: {result[0]}")

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

for i in range(len(predictions)):
    sim = util.cos_sim(model.encode(predictions[i]), model.encode(references[i]))[0][0].item()
    print(f"Label {i+1} SBERT Cosine Similarity: {sim:.4f}")