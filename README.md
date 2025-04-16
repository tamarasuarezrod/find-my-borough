<h1 align="center">
  FindMyBorough
</h1>

<p align="center">
    <strong>FindMyBorough</strong> is a machine learning–based system for recommending London boroughs based on user-defined preferences.
    <img src="frontend/public/images/banner.png" alt="FindMyBorough banner" width="100%">
</p>

## Overview

Finding a place to live in London can be overwhelming, especially for newcomers who don’t know the city well. FindMyBorough is a tool that helps people discover areas of London that best match their lifestyle preferences—such as affordability, safety, or youthfulness—based on real data and personalized recommendations. Users answer a short questionnaire, and the system suggests boroughs that suit them best.

## Tech Stack

- Backend: Django + Django REST Framework
- ML model: PyTorch + skopt (Bayesian Optimization)
- Data tools: pandas, numpy
- Frontend: Next.js + Tailwind CSS

## Project Structure

```bash
.
├── backend # Django backend
│ ├── ml_model # Data processing, training scripts, ML logic
│ ├── recommender # Recommendation related API
│ ├── borough  # Borough related API
│ ├── accounts # User authentication
│ ├── scripts # Utility scripts
│ └── seeds # Initial data
│
├── frontend # Next.js frontend
│ └── src # App pages, components, services, styling
```

## Environments

- Production: coming soon!

- Staging: https://staging.findmyborough.uk/

## Data

Multiple public datasets were combined to describe each London borough, including:

- **Rent prices** – London Datastore
- **Crime statistics** – Metropolitan Police (2021–2025)
- **Age demographics** – Office for National Statistics (ONS)
- **Transport zones** – Custom mapping based on TfL data

These were merged and normalized into a final feature set per borough.
To simulate user preferences, synthetic user profiles were generated with weighted preferences, contextual information (current_situation and stay_duration), and synthetic feedback scores to train the model in a controlled environment.

See [data_sheet.md](./data_sheet.md) for full details.

## Model

The recommendation system uses a feedforward neural network implemented in PyTorch. It predicts how suitable each borough is for a user based on their preference weights, contextual data, and borough-level features. This model was chosen for its flexibility and ability to capture non-linear relationships between inputs.

The model takes a 15-dimensional input vector:

- 4 borough features
- 4 user-defined preference weights
- 7 one-hot encoded features for current_situation and stay_duration

It outputs a score between 0 and 1 indicating suitability.
See [model_card.md](./model_card.md) for full details, or [train_model.ipynb](./backend/ml_model/notebooks/03_train_model.ipynb) to view the training notebook.

## Hyperparameter Optimization

Bayesian Optimization was used to tune the model’s architecture and learning rate. Parameters explored included the number of hidden layers, units per layer, and learning rate. The best configuration found was:

- Architecture: (15 → 64 → 32 → 1)
- Activation functions: ReLU (hidden layers), Sigmoid (output)
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Learning rate: 0.009997

## Results

On the synthetic test set, the model achieved a validation loss of 0.009997 (Binary Cross Entropy). This indicates good consistency between predicted and actual synthetic feedback scores. While the training labels are simulated, the model shows strong potential to personalize borough recommendations based on user preferences and context.
As real users begin interacting with the platform and providing feedback, the model can be retrained to better reflect actual behavior, improving the quality and relevance of recommendations over time.

## Contact

Created by Tamara Suárez – [LinkedIn](https://www.linkedin.com/in/tamarasuarezrod/)  
You can also check out my portfolio at [tamarasuarez.dev](https://tamarasuarez.dev/)
