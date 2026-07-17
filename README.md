# Parameter-Efficient Convolutional GRU Network with Temporal Regularization for Personalized Prandial Recommendations in Glycemic Control
This repository provides the implementation of a parameter-efficient temporally regularized Conv-GRU network for personalized prandial
recommendations in Type 1 diabetes. The model recommends insulin bolus and carbohydrate intake amounts to achieve a target blood glucose level
within a 30–90 minute post-meal horizon.
## Problem

* People with Type 1 diabetes using multiple daily insulin injections lack personalized decision-support tools for insulin bolus and carbohydrate recommendations.
* Existing machine-learning research mainly focuses on blood-glucose forecasting for sensor-augmented pump systems rather than open-loop MDI settings.
* Conventional rule-based bolus advisors cannot adequately capture individual physiological and glycemic variability.
* Reinforcement-learning approaches often struggle to learn stable policies because of highly variable glucose responses.
* Existing deep-learning models can be computationally expensive and difficult to deploy on resource-constrained healthcare devices.

## Key Achievements

* Developed a temporally regularized Conv-GRU model for personalized insulin bolus and carbohydrate intake recommendations.
* Supports blood-glucose targets over prediction horizons ranging from 30 to 90 minutes at 5-minute intervals.
* Introduced temporal masking to reduce temporal overfitting and improve model generalization.
* Consistently outperformed heuristic baselines, Chained-LSTM, and N-BEATS on the OhioT1DM dataset.
* Achieved lower RMSE and MAE across multiple insulin and carbohydrate recommendation scenarios.
* Used 31.5% fewer parameters than Chained-LSTM.
* Used more than 90.5% fewer parameters than N-BEATS.
* Required a total memory footprint of approximately 80 kB.
* Demonstrated the potential for efficient and deployable AI-driven decision support for Type 1 diabetes self-care.
<br> <br>

## Model Architecture

<p align="center">
  <img
    width="700"
    alt="Conv-GRU model architecture"
    src="https://github.com/user-attachments/assets/33d2e3a2-36ae-40d1-b78c-d1845009718b"
  />
</p>


<p align="center">
  <img
    width="700"
    alt="Temporal regularization architecture"
    src="https://github.com/user-attachments/assets/fdb19ff9-3af8-4a74-9263-5e3e0e160194"
  />
</p>
<p align="center">
  <em>Overview of the proposed temporally regularized Conv-GRU architecture.</em>
</p>

<br><br>

## Experimental Results

<p align="center">
  <img
    width="800"
    alt="Experimental results"
    src="https://github.com/user-attachments/assets/3d3fcc91-5fc7-42ab-843c-998661fbfc64"
  />
</p>

<br>

<p align="center">
  <img
    width="750"
    alt="Performance comparison"
    src="https://github.com/user-attachments/assets/d66bfe07-3d08-4a2d-a4a3-ffc5e6de7776"
  />
</p>
<p align="center">
  <em>
    Comparison of recommendation errors across prediction horizons and
    recommendation scenarios.
  </em>
</p>

<br>

<p align="center">
  <img
    width="650"
    alt="Model size and parameter comparison"
    src="https://github.com/user-attachments/assets/01f9d6e1-5a5e-468b-8b0a-47732a3258a9"
  />
</p>

<br><br>

## Repository Structure

```text
Conv-GRU/
├── src/
│   ├── model.py                  # Conv-GRU model architecture
│   ├── pretrain.py               # Model pretraining
│   ├── finetune.py               # Subject-specific fine-tuning
│   ├── testing.py                # Model evaluation
│   ├── run.py                    # Main experiment runner
│   ├── baselines.py              # Baseline implementations
│   ├── load_data.py              # Data loading and preprocessing
│   ├── split_data.py             # Dataset splitting
│   └── *_hyper_parameters.json   # Experiment configurations
├── data/
│   ├── processed/                # Processed experimental data
│   └── processed_without_bolus/  # Processed no-bolus data
├── README.md
└── requirements.txt
```

<br> <br>

## How to run the code
There are two recommendation scenarios: general recommendation and without bolus recommendation. To run the general recommendation, copy the pkl files from data/processed and paste them inside src. Run the pipeline.ipynb. For the without-bolus-recommendation scenario, copy the pkl files from data/processed_without_bolus and paste them inside src, and run the pipeline.ipynb. All the hypermeter files and modules are inside src. Make sure to use it as an active folder. 
