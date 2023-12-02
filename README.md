# Question-Answering-from-PDF-Documents-Using-Speech-Input
This repository contains the code for a PDF-based speech-to-answer system that allows users to ask questions about a PDF document using speech input. The system utilizes the jonatasgrosman/wav2vec2-large-xlsr-53-english automatic speech recognition (ASR) model from Hugging Face Transformers to transcribe spoken questions and then leverages the deepset/roberta-base-squad2 question-answering model to answer those questions based on the provided PDF document.



Clone the repository:
git clone 


Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate


Install the required dependencies:
pip install -r requirements.txt
Usage
To run the code, follow these steps:

copy the path to the  PDF document you want to use:
//example.com/example.pdf


Place the selected PDF document in the same directory as the code.

Run the code:

python3 main.py
Follow the prompts to record your question and view the answer.

Improving the System
The system can be improved in several ways:

Add support for multiple languages.


Improve the accuracy of the ASR and question-answering models.


Add support for more complex questions.


Create a graphical user interface (GUI) for the system.
