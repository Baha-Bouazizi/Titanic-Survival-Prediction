# Titanic Survival Prediction 🛳️

A **Streamlit** web app that predicts whether a passenger aboard the RMS Titanic would have survived, based on classic Kaggle Titanic features.


## Features

* Sleek glassmorphism UI with blue gradient background
* Sidebar inputs for all relevant passenger attributes
* Probability bar chart (Altair) showing survival likelihood
* Instant prediction using a pre-trained model (`best_model.pkl3`)
* Works offline – no external API calls

## Quick Start

```bash
# 1. Clone (or download) this repository

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows


# 4. Launch the Streamlit app
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`.

## Requirements

* Python 3.8+
(core: streamlit, pandas, altair, scikit-learn, pickle-5)

A minimal `requirements.txt` is provided:

```
streamlit>=1.24
pandas>=2.0
altair>=5.0
scikit-learn>=1.3
pickle5; python_version<'3.8'
```

## Model Details

`best_model.pkl3` is a scikit-learn pipeline trained on the Titanic dataset with engineered features:

| Feature      | Description                               |
|--------------|-------------------------------------------|
| `pclass`     | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)|
| `sex`        | Biological sex                            |
| `age`        | Age in years                              |
| `sibsp`      | # siblings / spouses aboard              |
| `parch`      | # parents / children aboard              |
| `fare`       | Ticket fare (£)                           |
| `class`      | Cabin class name (First / Second / Third) |
| `who`        | Man / Woman / Child                       |
| `adult_male` | Boolean: adult male?                      |
| `alone`      | Boolean: travelling alone?                |

The model outputs:

* **Prediction** – 1 = Survived, 0 = Did not survive
* **Probability** – Confidence (%) of survival

> Note: If you wish to retrain the model, ensure you keep the same feature order and data preprocessing to remain compatible with `main.py`.

## File Structure

```
├─ main.py                # Streamlit application
├─ best_model.pkl3        # Pre-trained scikit-learn model
├─ Practice Project.ipynb # Original notebook (EDA / training)
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation (this file)
```

## License

MIT – free to use, modify, and distribute.

---

Built with ❤ using Streamlit and scikit-learn. Enjoy predicting!
