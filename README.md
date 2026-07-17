# Parameter-Efficient Convolutional GRU Network with Temporal Regularization for Personalized Prandial Recommendations in Glycemic Control

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


<img width="3600" height="2700" alt="image" src="https://github.com/user-attachments/assets/33d2e3a2-36ae-40d1-b78c-d1845009718b" />

<img width="3600" height="2700" alt="image" src="https://github.com/user-attachments/assets/fdb19ff9-3af8-4a74-9263-5e3e0e160194" />

<img width="10880" height="6120" alt="fig3" src="https://github.com/user-attachments/assets/3d3fcc91-5fc7-42ab-843c-998661fbfc64" />

<img width="3132" height="2206" alt="image" src="https://github.com/user-attachments/assets/d66bfe07-3d08-4a2d-a4a3-ffc5e6de7776" />

<img width="994" height="350" alt="image" src="https://github.com/user-attachments/assets/01f9d6e1-5a5e-468b-8b0a-47732a3258a9" />

