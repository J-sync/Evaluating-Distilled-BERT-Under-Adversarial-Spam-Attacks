# BERT Distillation for Spam Classification with Adversarial Attacks

This project explores knowledge distillation from a BERT teacher model to a lightweight student model for binary spam classification. It also evaluates model robustness through adversarial email perturbations using word substitutions.

---

## 📁 Project Structure

- `student_model.ipynb` – Trains a student model using BERT as a teacher model on spam/ham data.
- `attack_email.py` – Generates adversarial spam emails by perturbing them with randomly chosen replacement words.
- `freq.py` – Computes word frequency statistics in spam vs ham emails from `spam.csv`.
- `spam.csv` – Source dataset with labeled spam and ham messages.
- `attack.csv` – Output file containing adversarial spam samples and cosine similarity scores.

---

## ⚙️ Installation

Tested with **Python 3.9+**

Install required dependencies:
```bash
pip install -r requirements.txt
```

Initialize NLTK resources after installation:
```python
import nltk
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('words')
```

---

## 🚀 How to Run

Ensure `spam.csv` is in the root directory before running the scripts.

1. **Train Models**
   Open and run all cells in `student_model.ipynb`.

2. **Analyze Word Frequencies**
   ```bash
   python freq.py
   ```

3. **Generate Adversarial Emails**
   ```bash
   python attack_email.py
   ```

---

## 📊 Results

| Model         | Accuracy | Description                           |
|---------------|----------|---------------------------------------|
| BERT (Teacher) | ~98.4%   | Fine-tuned on spam dataset            |
| Student Model  | ~95.6%   | Distilled lightweight spam classifier |

---

## 📈 Performance & Evaluation

- **Teacher Model (BERT)**: Fine-tuned using `tensorflow/bert_en_uncased_L-12_H-768_A-12`.
- **Student Model Accuracy**: Consistent accuracy of **89%**, with macro F1-score also at **0.89** as shown below:

```
              precision    recall  f1-score   support

           0       0.91      0.88      0.89       187
           1       0.88      0.91      0.89       187

    accuracy                           0.89       374
   macro avg       0.89      0.89      0.89       374
weighted avg       0.89      0.89      0.89       374
```

---

## 🧠 Attack Methodology

These attack types mimic adversarial conditions a spam classifier might face in real-world deployment.

1. **Cosine Similarity** is used to ensure semantic similarity between original and attacked emails. A similarity score **> 0.8** is considered acceptable for keeping attack emails realistic.

2. **Attack Generation**:
   - Random spam emails are picked from `spam.csv`.
   - Around **10% of the words** are replaced using similar-length words from NLTK's Gutenberg corpus.
   - If cosine similarity between original and attacked version is > 0.8, the attack email is accepted.
   - ~70 such emails were generated, covering approximately **10% of the dataset**.

---

## 🧪 Attack Variants

### 🔸 Attack 1 (Test-time Attack)
- Does not interfere with model training.
- Tests robustness of the pre-trained model directly.
- **Impact**: Leads to **misclassification**, degrading accuracy to **81%**.
- Primarily affects **deep-layer feature extractors**.

### 🔹 Attack 2 (Training-time Attack)
- Re-trains the student model by including attack emails in training data.
- Tests resilience of the model to data poisoning.
- **Impact**: Drops accuracy significantly to **63%**, potentially detrimental.
- Impacts **mid-layer feature extractors**.

---

## ⚠️ Limitations & Future Work

- Attack strategy uses random word substitutions — could be improved with synonym/embedding-based attacks.
- `attack_email.py` uses TensorFlow BERT while other scripts use Huggingface; standardization is advised.
- Cosine similarity is computed but not yet used in downstream evaluation or training loop weighting.

---

## 📄 License

MIT License

---

## 🤝 Contributions

Feel free to open issues or pull requests for improvements.
