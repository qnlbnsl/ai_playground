from transformers import pipeline


def classify(input_text):
    # classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # results = classifier("Fred drives a car.")
    #
    # # Use a breakpoint in the code line below to debug your script.
    #
    # print(f'Classifier: {results}')

    # Initialize the text classification pipeline
    classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
    # Perform text classification on the input text
    result = classifier(input_text)
    # Print the predicted label and the associated confidence
    print(f"Predicted label: {result[0]['label']}")
    print(f"Confidence score: {result[0]['score']:.2f}")
