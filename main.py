from transformers import pipeline


def main(input_text):
    # Initialize the text classification pipeline
    classifier = pipeline('text-classification', model='distilbert-base-cased')

    # Perform text classification on the input text
    result = classifier(input_text, labels=["positive", "negative"])

    # Print the predicted label and the associated confidence
    print(f"Predicted label: {result[0]['label']}")
    print(f"Confidence score: {result[0]['score']:.2f}")


if __name__ == '__main__':
    text = "This is a positive review of a product."
    main(text)
