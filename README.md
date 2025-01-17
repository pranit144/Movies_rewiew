# Sentiment Classifier with BERT

This project is a sentiment classification application built using **Streamlit**, **Hugging Face Transformers**, and **TensorFlow**. The model is based on BERT (Bidirectional Encoder Representations from Transformers) and is fine-tuned for sentiment analysis. The application predicts whether a given text has a positive or negative sentiment.

---

## Features

- Interactive user interface with **Streamlit**.
- Fine-tuned BERT model for accurate sentiment analysis.
- Real-time text classification.
- Displays word count, character count, and sentiment scores.
- Customizable UI with enhanced CSS styling.

---

## File Structure

```
Sentiment_Classifier_with_BERT/
│
├── Model/                       # Directory containing the fine-tuned BERT model
│   └── (saved model files)
├── Tokenizer/                   # Directory containing the tokenizer files
│   └── (saved tokenizer files)
├── app.py                       # Main Streamlit application script
├── requirements.txt             # Dependencies required to run the project
├── README.md                    # Documentation file
```

---

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Dependencies

- streamlit
- tensorflow
- transformers
- numpy

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-classifier.git
```

2. Navigate to the project directory:

```bash
cd sentiment-classifier
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the URL provided by Streamlit in your browser (typically `http://localhost:8501`).

---

## How It Works

1. The user enters a sentence in the input text area.
2. The input is tokenized and passed through the fine-tuned BERT model.
3. The model predicts the sentiment:
   - **Positive** if the label is `1`
   - **Negative** if the label is `0`
4. The app displays:
   - Predicted sentiment
   - Word count
   - Character count
   - Sentiment scores
5. A bar chart visualizes the sentiment scores.

---

## Model and Tokenizer

The project uses a pre-trained BERT model and tokenizer that are fine-tuned for sentiment classification. Update the `Model` and `Tokenizer` paths in the code as per your local directory structure.

---

## UI Design

Custom CSS enhances the user interface for:

- Background color
- Text area styling
- Button appearance
- Sentiment results display

---

## Example

1. Enter a sentence:
   - *"I love this product, it's amazing!"*
2. Predicted Sentiment: **Positive**
3. Scores:
   - Positive: `0.95`
   - Negative: `0.05`

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
