Forest Fire Prediction
Data‑driven wildfire risk estimation with a Streamlit UI and a reproducible ML pipeline grounded in Fire Weather Index (FWI) components.

Table of Contents
Overview.

Features.

Technologies Used.

Installation.

Usage.

Deployment.

Live Demo.

Examples.

Contributing.

Acknowledgements.

Overview
Forest Fire Prediction estimates the probability of a fire event using spatial cues, seasonality, meteorology, and core FWI components such as FFMC, DMC, DC, and ISI, derived from standard practice and the UCI Forest Fires dataset schema.
The application provides an interactive Streamlit interface for real‑time scoring and threshold selection, enabling practitioners to tune operational sensitivity while inspecting model quality via threshold‑independent metrics.
Core goals: build an interpretable, portable pipeline, leverage FWI‑aligned features, and expose a fast UI that supports repository or uploaded datasets.

Features
Data‑source selector in the sidebar to load a repository CSV or upload a CSV at runtime without code changes.

Automated preprocessing for tabular fire data: categorical handling for month/day, cyclical encodings if used, and numeric scaling aligned to model assumptions.

Incorporation of FWI components and drivers: FFMC (fine‑fuel moisture), DMC (duff moisture), DC (drought), and ISI (initial spread), all influenced by temperature, humidity, wind, and rain.

Threshold‑independent evaluation with ROC‑AUC and PR‑AUC, plus an operator threshold slider for practical alert tuning.

Clean Streamlit layout: characteristics on the left and a concise prediction panel on the right for decision support.

Technologies Used
Python for the end‑to‑end modeling and app runtime in a single repository.

scikit‑learn for preprocessing (encoders/scalers), model training, and evaluation (ROC/PR metrics).

Streamlit for the interactive web interface and simple local/cloud execution via CLI.

Fire Weather Index references for domain‑aligned features and interpretation.

Installation
Clone the repository.

text
git clone <your-repo-url>
cd forest-fire-prediction
Create a virtual environment and install dependencies.

text
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
Usage
Place the dataset (for example, the UCI forest fires CSV) in the repository or keep a CSV ready for upload from the sidebar.

Run the Streamlit app locally.

text
streamlit run app.py
In the sidebar, choose “Repository file” or “Upload CSV,” set model/threshold controls, enter characteristics, and view the predicted fire probability and risk status.

Deployment
Run the same CLI on a server or a managed platform; Streamlit’s run command starts a local server and opens the app in the browser, with identical behavior in containerized or cloud environments.
Keep dependency versions pinned and use a repo‑relative default dataset path to ensure reproducible builds and predictable startup behavior.

Live Demo
Add the hosted Streamlit app URL here after deploying; the same entry point used locally (streamlit run app.py) applies on most hosting targets.

Examples
Input: spatial coordinates (X, Y), calendar (month, day), FFMC/DMC/DC/ISI, temperature, relative humidity, wind, and rain, matching the UCI schema.

Output: fire probability and an “in danger/safe” label at the chosen decision threshold, supported by ROC‑AUC and PR‑AUC for model quality at all cutoffs.

Contributing
Fork the repository and create a feature branch.

text
git checkout -b feature/your-feature
Commit and push your changes.

text
git commit -m "Add feature: your-feature"
git push origin feature/your-feature
Open a pull request with a clear description and rationale for review.

Acknowledgements
Fire Weather Index documentation and guidance on FFMC, DMC, DC, ISI, BUI, and FWI usage for operational insights.

UCI Forest Fires dataset and derivative descriptions used for feature schema and experimentation.

Streamlit and scikit‑learn for rapid development of reproducible web ML workflows.

