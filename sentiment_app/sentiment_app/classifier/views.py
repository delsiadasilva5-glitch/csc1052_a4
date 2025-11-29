from django.shortcuts import render
import os
import joblib
from lime.lime_text import LimeTextExplainer

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "classifier", "models")

# -------------------------------------------------------
# Load models
# -------------------------------------------------------
vectorizer_bow = joblib.load(os.path.join(MODEL_DIR, "vectorizer_bow.pkl"))
logreg_bow    = joblib.load(os.path.join(MODEL_DIR, "logreg_bow.pkl"))
mlp_bow       = joblib.load(os.path.join(MODEL_DIR, "mlp_bow.pkl"))
ensemble_bow  = joblib.load(os.path.join(MODEL_DIR, "soft_ensemble.pkl"))
nb_bow        = joblib.load(os.path.join(MODEL_DIR, "nb_bow.pkl"))
dt3_bow       = joblib.load(os.path.join(MODEL_DIR, "dt3_bow.pkl"))

explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

# -------------------------------------------------------
# Model performance summary (no notes)
# -------------------------------------------------------
MODEL_PERFORMANCE = {
    "logreg": {
        "name": "Logistic Regression (BoW)",
        "val_accuracy": 0.85,
        "test_accuracy": 0.84,
        "train_time_s": 6.32,
    },
    "mlp": {
        "name": "MLP (BoW)",
        "val_accuracy": 0.86,
        "test_accuracy": 0.828,
        "train_time_s": 10.60,
    },
    "nb": {
        "name": "Naive Bayes (BoW)",
        "val_accuracy": 0.82,
        "test_accuracy": 0.81,
        "train_time_s": 0.20,
    },
    "dt3": {
        "name": "Decision Tree (BoW)",
        "val_accuracy": 0.74,
        "test_accuracy": 0.72,
        "train_time_s": 3.67,
    },
    "ensemble": {
        "name": "Soft Ensemble (DT+NB+LR+MLP, BoW)",
        "val_accuracy": 0.85,
        "test_accuracy": 0.8388,
        "train_time_s": 0.50,
    },
}


# -------------------------------------------------------
# Main classifier view
# -------------------------------------------------------
def classify_review(request):
    context = {
        "models": {
            "logreg": "Logistic Regression (BoW)",
            "mlp": "MLP (BoW)",
            "nb": "Naive Bayes (BoW)",
            "dt3": "Decision Tree (BoW)",
            "ensemble": "Soft Ensemble (BoW)",
        },
        "lime_explanation": None,
        "model_stats": None,
    }

    if request.method == "POST":
        review_text = request.POST.get("review_text", "").strip()
        model_key = request.POST.get("model_key")

        # ---- Validation ----
        if not model_key:
            context["error"] = "Please choose a model."
            return render(request, "classifier/index.html", context)

        if not review_text:
            context["error"] = "Please enter a review before submitting."
            return render(request, "classifier/index.html", context)

        # ---- Choose model based on dropdown ----
        if model_key == "logreg":
            model = logreg_bow
        elif model_key == "mlp":
            model = mlp_bow
        elif model_key == "nb":
            model = nb_bow
        elif model_key == "dt3":
            model = dt3_bow
        else:
            # default -> ensemble
            model = ensemble_bow

        # ---- Vectorize + Predict ----
        X = vectorizer_bow.transform([review_text])
        y_pred = model.predict(X)[0]

        # ---- Probabilities ----
        prob_neg = prob_pos = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prob_neg, prob_pos = float(proba[0]), float(proba[1])

        # ---- LIME Explanation ----
        def predict_proba_wrapper(texts):
            X_wrap = vectorizer_bow.transform(texts)
            return model.predict_proba(X_wrap)

        exp = explainer.explain_instance(
            review_text,
            predict_proba_wrapper,
            num_features=8
        )

        lime_list = exp.as_list()
        lime_explanation = [
            {"word": w, "weight": float(wt)}
            for (w, wt) in lime_list
        ]

        # ---- Model stats for this classifier ----
        model_stats = MODEL_PERFORMANCE.get(model_key)

        # ---- Update context ----
        context.update({
            "review": review_text,
            "pred_label": y_pred,
            "prob_neg": prob_neg,
            "prob_pos": prob_pos,
            "model_name": context["models"][model_key],
            "lime_explanation": lime_explanation,
            "model_stats": model_stats,
        })

    return render(request, "classifier/index.html", context)


# -------------------------------------------------------
# Summary page view
# -------------------------------------------------------
def model_summary(request):
    """
    Renders a page summarising performance of all models:
    validation/test accuracy and training time.
    """
    context = {
        "stats": MODEL_PERFORMANCE
    }
    return render(request, "classifier/summary.html", context)
