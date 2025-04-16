# Model Card

## Model Description

**Input:**  
The model takes a **15-dimensional vector** composed of:

- **User-defined preference weights:**

  - `budget_weight`: how sensitive the user is to affordability
  - `safety_weight`: how important safety (low crime) is to the user
  - `youth_weight`: preference for areas with younger population
  - `centrality_weight`: how much the user values proximity to central London

- **One-hot encoded user context:**

  - `current_situation`: one-hot encoded as:
    - `situation_student`
    - `situation_young_professional`
    - `situation_professional`
    - `situation_other`
  - `stay_duration`: one-hot encoded as:
    - `stay_short_term`
    - `stay_medium_term`
    - `stay_long_term`

- **Borough-specific features:**
  - `norm_rent`: normalized rent value for the borough
  - `norm_crime`: normalized crime rate
  - `norm_youth`: normalized proportion of young residents
  - `norm_centrality`: normalized centrality (proximity to London center)

**Output:**  
A single float between 0 and 1 representing how suitable a given borough is for the user based on their preferences.

**Model Architecture:**  
A feedforward neural network implemented in PyTorch with the following structure:

- Input layer: 15 features
- Hidden layer 1: 64 units (ReLU)
- Hidden layer 2: 32 units (ReLU)
- Output layer: 1 unit (Sigmoid activation)

Optimized using Binary Cross Entropy Loss and the Adam optimizer. Final hyperparameters were selected via Bayesian Optimization.

---

## Performance

- **Validation loss:** `0.009997` (Binary Cross Entropy)
- **Validation split:** 20% of a synthetic dataset with binary feedback scores
- **Optimization method:** Bayesian Optimization (`skopt`)
- **Best hyperparameters:**
  - Learning rate: `0.018660708832743716`
  - Hidden layer sizes: `64`, `32`
- **Model version:** `v3`
- **Filename:** `score_model_v3_2025-04-16.pth`

---

## Limitations

- The model is trained on synthetic user feedback, which approximates how different user types might rank boroughs based on general assumptions. Actual user preferences may vary.
- The model currently includes only a limited set of borough features. Important factors like access to transport, schools, healthcare, and green spaces are not yet included.
- Scores may slightly shift if new borough-level data is added or normalization parameters are recalculated.
- The **City of London** is missing crime data, which may affect prediction quality for that borough.

---

## Trade-offs

- **Interpretability vs. flexibility:** A neural network was chosen over simpler models to better capture complex relationships between preferences and borough features. However, this comes at the cost of being less interpretable.
- **Tuning depth vs. efficiency:** Hyperparameters were optimized using 20 iterations of Bayesian Optimization for practical runtime. A deeper search could yield better results, but would require significantly more compute.
