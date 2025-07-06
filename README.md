<!-- Banner -->
<p align="center">
  <img src="https://raw.githubusercontent.com/hamaylzahid/exoplanet-discovery-habitability-predictor/main/docs/banner_exoplanet.png" alt="Exoplanet Discovery Banner" width="100%"/>
</p>

<h1 align="center">🌌 Exoplanet Discovery & Habitability Predictor</h1><br>

<p align="center">
  <em>Unveiling the mysteries of distant worlds</em> —  
  A Machine Learning-powered approach to identify potential life-supporting exoplanets using real astronomical datasets from NASA and observatories around the globe.
</p>


<p align="center">
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/github/languages/top/hamaylzahid/exoplanet-discovery-habitability-predictor?color=blue&logo=python" alt="Top Language">
  <img src="https://img.shields.io/github/last-commit/hamaylzahid/exoplanet-discovery-habitability-predictor?logo=github" alt="Last Commit">
  <img src="https://img.shields.io/badge/Made%20With-Scikit--Learn%2C%20Pandas%2C%20Matplotlib-blue" alt="Stack Badge">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" alt="Project Status">
</p>

<br>
<h2 align="center">📖 Table of Contents</h2><br>

- [🧠 Project Overview](#-project-overview)  
- [🎯 Objectives](#-objectives)  
- [📊 Dataset Info](#-dataset-info)  
- [🛠️ Workflow Breakdown](#️-workflow-breakdown)  
- [📈 Metrics & Results](#-metrics--results)  
- [🔥 Bias Mitigation Strategy](#-bias-mitigation-strategy)  
- [⚙️ Setup & Installation](#️-setup--installation)  
- [📚 Concepts Covered](#-key-concepts-covered)  
- [🚀 Deployment Notes](#-recommendations-for-ethical-deployment)  
- [🙏 Acknowledgments](#-acknowledgments)  
- [📚 Core Libraries Used](#-core-libraries-used)  
- [🤝 Contact & Contribution](#-contact--contribution)  
- [📜 License](#-license)

---

<br><h2 align="center">🧠 Project Overview</h2><br>

This project leverages machine learning to evaluate the **habitability potential of exoplanets** based on real observational data.  
It analyzes critical planetary attributes such as **mass, orbital period, discovery methods,** and **host star systems** to uncover patterns that may indicate the possibility of life-supporting conditions.

---

<br><h2 align="center">🎯 Objectives</h2><br>

- 🪐 **Predict Exoplanet Habitability**: Build ML models to assess the potential for life-supporting conditions using real astronomical parameters  
- 📊 **Uncover Discovery Patterns**: Visualize and analyze how detection methods and frequency have evolved across years and facilities  
- 🧠 **Compare ML Algorithms**: Train, evaluate, and fine-tune multiple models (e.g., Logistic Regression, Decision Trees) to identify the most effective predictors  
- 🔍 **Expose Scientific Biases**: Reveal limitations in observational techniques and explore how discovery methods skew the datasets  
- 🌍 **Bridge AI & Astrobiology**: Demonstrate how machine learning can support scientific research in space exploration and deepen our understanding of planetary systems  
- 🚀 **Create a Launchpad for Future Research**: Lay the foundation for advanced models that can be scaled to broader datasets, integrated with satellite telemetry, or used by researchers

---

<br><h2 align="center">📊 Dataset Info</h2><br>

> 🛰️ **Source:** NASA’s Exoplanet Archive and additional open-access astronomical datasets.

The dataset contains physical and observational features of confirmed exoplanets collected via various discovery missions. Each row represents a planet, accompanied by key measurable attributes.

| 🪐 **Feature**             | 🧾 **Description**                                  |
|----------------------------|-----------------------------------------------------|
| Planet Name                | Official name or designation of the exoplanet       |
| Planet Host                | Name of the host star or system                     |
| Number of Stars / Planets  | Count of stars and planets in the system            |
| Discovery Method & Year    | Technique used (e.g., Radial Velocity, Imaging) and year discovered |
| Orbital Period (days)      | Time the planet takes to complete a full orbit       |
| Semi-Major Axis (AU)       | Average distance from its host star (in AU)          |
| Mass (Jupiter-relative)    | Estimated mass of the planet relative to Jupiter     |


---

<br><h2 align="center">🛠️ Workflow Breakdown</h2><br>

1. 🔹 **Data Cleaning & Preprocessing**  
   - Handled missing values, standardized formats, encoded categorical fields  
2. 📊 **Exploratory Data Analysis (EDA)**  
   - Visualized distributions, relationships, and discovery trends across variables  
3. 🧠 **Model Development & Training**  
   - Built and trained models including Logistic Regression and Decision Trees  
4. 📈 **Evaluation & Validation**  
   - Assessed performance using metrics like Accuracy, F1 Score, and R² Score  
5. 🌌 **Scientific Interpretation**  
   - Interpreted ML outcomes to understand planetary traits linked to habitability  


---

<br><h2 align="center">📈 Metrics & Results</h2><br>

### Discovery Method Distribution
![Discovery Method](https://raw.githubusercontent.com/hamaylzahid/exoplanet-discovery-habitability-predictor/main/docs/discovery_method_chart.png)

### Mass vs Orbital Period
![Mass vs Period](https://raw.githubusercontent.com/hamaylzahid/exoplanet-discovery-habitability-predictor/main/docs/mass_vs_orbit.png)

> Replace with actual plots from your notebook if needed.

---

<br><h2 align="center">🔥 Bias Mitigation Strategy</h2><br>

To ensure the integrity and fairness of our predictive modeling, we implemented several bias mitigation techniques rooted in best practices for scientific data handling:

- 🧹 **Eliminated Redundancy**: Removed highly correlated and duplicate features to prevent model distortion and data leakage  
- ⚖️ **Balanced Representation**: Adjusted sample distribution across discovery methods and years to avoid overfitting to dominant categories  
- 🧪 **Stratified Sampling**: Used stratified train-test splitting to preserve class distributions and prevent sampling bias  
- 🛰️ **Facility-Based Bias Analysis**: Investigated potential overrepresentation from certain observatories and instruments to account for discovery inequalities  
- 🚨 **Avoided Overinterpretation**: Carefully interpreted results, keeping in mind that observational data may reflect technological limitations, not just scientific truths

> 🧠 *Bias isn't just a social issue—it's a statistical one too. Addressing it makes your model smarter, not just fairer.*


---

<br><h2 align="center">⚙️ Setup & Installation</h2><br>

```bash
# Step 1: Clone the repository
git clone https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor.git

# Step 2: Move into the folder
cd exoplanet-discovery-habitability-predictor

# Step 3: Install required packages
pip install -r requirements.txt

# Step 4: Run notebooks or scripts in /src
```
---

<br><h2 align="center">📚 Key Concepts Covered</h2><br>

Dive deep into each core concept with the descriptions below:

<h4>🧹 Data Wrangling & Transformation</h4>  
Clean, normalize, and encode raw exoplanet data to prepare for analysis and modeling.

<h4>🔍 Exploratory Data Analysis (EDA)</h4>  
Uncover hidden patterns and relationships through statistical summaries and visualizations.

<h4>🤖 Classification & Regression Modeling</h4>  
Apply and compare algorithms (Logistic Regression, Decision Trees, etc.) to predict habitability scores.

<h4>✨ Feature Selection</h4>  
Identify and retain the most informative features while reducing noise and dimensionality.

<h4>🔬 Scientific Data Interpretation</h4>  
Translate model outputs into meaningful astrophysical insights, respecting observational limitations.

<h4>📊 Data Visualization (Static & Comparative)</h4>  
Craft clear, publication-ready plots to communicate findings and compare planetary attributes.

---

<br><h2 align="center">🚀 Recommendations for Ethical Deployment</h2><br>

When applying machine learning to scientific domains like astrobiology and planetary research, ethical responsibility is essential. Here are some key considerations:

<h4>🔭 Acknowledge Observational Limitations</h4>  
Astronomical data is influenced by telescope sensitivity, discovery method preference, and mission-specific biases. These factors must be transparently addressed before drawing conclusions.

<h4>🧪 Avoid Over-Interpreting Predictions</h4>  
ML models offer probabilistic insights — not certainties. Treat habitability predictions as exploratory tools, not absolute scientific verdicts.

<h4>⚠️ Communicate Uncertainty Clearly</h4>  
Always include margins of error, confidence scores, and assumptions when presenting findings. Encourage peer validation and replication.

> 🌐 *Responsible AI in scientific domains means respecting the data's origin, understanding its limits, and promoting open science.*

---

<br><h2 align="center">🙏 Acknowledgments</h2><br>

- NASA Exoplanet Archive  
- Python open-source ecosystem  
- Instructors and mentors who guided the project  
- Research communities on Astrobiology & Astronomy
> 💡 *This project stands on the shoulders of open science and collaborative learning.*
---
<br><h2 align="center">📚 Core Libraries Used</h2><br>

<p align="center">
  <img src="https://img.shields.io/badge/pandas-Data%20Handling-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-Numerical%20Computing-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Toolkit-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/matplotlib-Data%20Plots-11557C?style=flat-square&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/seaborn-Statistical%20Graphics-4186B4?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/PyCharm-IDE-000000?style=flat-square&logo=pycharm&logoColor=white" />
</p>

<p align="center">

| 🔧 Purpose           | 🧰 Libraries                         |
|----------------------|-------------------------------------|
| Data Analysis        | `pandas`, `numpy`                   |
| Visualization        | `matplotlib`, `seaborn`             |
| Machine Learning     | `scikit-learn`                      |
| Notebook & IDE       | Jupyter Notebook, PyCharm           |
| Data Export & Utils  | `csv`, `os`, `glob`                 |

</p>
---

<br><h2 align="center">🤝 Contact & Contribution</h2><br>

<p align="center">
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Gmail Badge" />
  </a>
  <a href="https://www.linkedin.com/in/your-linkedin-profile">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn Badge" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor/stargazers">
    <img src="https://img.shields.io/github/stars/hamaylzahid/exoplanet-discovery-habitability-predictor?style=flat-square&logo=github&color=yellow" alt="Star Badge" />
  </a>
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-2b9348?style=flat-square&logo=github" alt="PRs Welcome Badge" />
  </a>
</p>

<p align="center">
  <strong>Have feedback, want to collaborate, or just say hello?</strong><br>
  Let’s connect and build something amazing together!
</p>

---


<br><h2 align="center">📜 License</h2><br>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor/commits/main">
    <img src="https://img.shields.io/github/last-commit/hamaylzahid/exoplanet-discovery-habitability-predictor?color=blue" alt="Last Commit">
  </a>
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor">
    <img src="https://img.shields.io/github/repo-size/hamaylzahid/exoplanet-discovery-habitability-predictor?color=lightgrey" alt="Repo Size">
  </a>
</p>

<p align="center">
  This project is licensed under the <strong>MIT License</strong> – free to use, modify, and distribute.
</p>

<p align="center">
  ✅ <strong>Project Status:</strong> Completed and ready for portfolio showcase<br>
  🧾 <strong>License:</strong> MIT – <a href="LICENSE">View License »</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Built%20with-Python-blue?style=flat-square&logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Modeling-f7931e?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualizations-11557c?style=flat-square&logo=plotly&logoColor=white" alt="Matplotlib Badge" />
</p>

<p align="center">
  <b>Crafted with cosmic curiosity & scientific precision</b> 🪐  
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •  
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email Badge" />
  </a>
  •  
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/exoplanet-discovery-habitability-predictor/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Explore%20the%20Universe-2ea44f?style=flat-square&logo=github" alt="Fork Project Badge" />
  </a>
</p>

<p align="center">
  <sub><i>Inspired by the stars. Driven by data. Designed to discover the unknown.</i></sub>
</p>

<p align="center">
  🔭 <b>Use this project to showcase your passion for AI-driven space exploration</b>  
  <br>
  🧪 Clone it, analyze it, expand it — and bring cosmic data to life with machine learning.
</p>
