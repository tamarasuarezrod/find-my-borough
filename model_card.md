# Model Card

## Model Description

**Input:**  
The model takes an 8-dimensional vector composed of:

- **User-defined preference weights:**

  - `budget_weight`: how sensitive the user is to affordability
  - `safety_weight`: how important safety (low crime) is to the user
  - `youth_weight`: preference for areas with younger population
  - `centrality_weight`: how much the user values proximity to central London

- **Borough-specific features:**
  - `norm_rent`: normalized rent value for the borough
  - `norm_crime`: normalized crime rate
  - `norm_youth`: normalized proportion of young residents
  - `norm_centrality`: normalized centrality (proximity to London center)

**Output:**  
A single float between 0 and 1 representing how suitable a given borough is for the user based on their preferences.

**Model Architecture:**  
A feedforward neural network implemented in PyTorch with the following structure:

- Input layer: 8 features
- Hidden layer 1: 15 units (ReLU)
- Hidden layer 2: 32 units (ReLU)
- Output layer: 1 unit (Sigmoid activation)

Optimized using Binary Cross Entropy Loss and Adam optimizer. Final hyperparameters selected through Bayesian Optimization.

---

## Performance

- **Validation loss:** 0.1672 (Binary Cross Entropy)
- **Validation split:** 20% of a synthetic dataset with labeled scores
- **Optimization method:** Bayesian Optimization (`skopt`)
- **Best hyperparameters:**
  - Learning rate: `0.00806`
  - Hidden layer sizes: `15`, `32`

The model was retrained for 30 epochs using these hyperparameters before final evaluation and saving.

---

## Limitations

- The model was trained using real borough-level data combined with synthetic user preferences and feedback scores, meaning the area features are factual, but the training labels simulate expected behavior. The goal is for the platform to collect enough real user feedback over time to retrain and improve the model based on actual preferences and behaviors.
- The model uses only a selected set of borough features (rent, crime rate, youth population, centrality). While these are relevant, it does not yet incorporate other potentially influential data like access to transport, schools, or green spaces.
- Feature values are normalized to ensure consistent training and prediction behavior. However, predictions may slightly change if the dataset is updated and normalization parameters are recalculated.
- City of London is the only borough with missing crime rate data, as it is policed by the City of London Police instead of the Metropolitan Police, which provided the crime data for all other boroughs. As a result, the model may produce less reliable recommendations for City of London. Integrating external data sources could improve coverage, but care must be taken to ensure the data is comparable and consistent with the existing dataset.

---

## Trade-offs

- **Interpretability vs. flexibility:** A neural network was chosen over simpler models to better capture complex relationships between preferences and borough features. However, this comes at the cost of being less interpretable.
- **Tuning depth vs. efficiency:** Hyperparameters were optimized using 20 iterations of Bayesian Optimization for practical runtime. A deeper search could yield better results, but would require significantly more compute.
