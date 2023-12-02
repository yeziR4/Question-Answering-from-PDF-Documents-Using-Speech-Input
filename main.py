import fitz
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from base64 import b64decode
from IPython.display import display, Javascript
from google.colab import output
import ipywidgets as widgets
from IPython.display import display

# Load ASR model using pipeline
asr_pipe = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Load the question-answering pipeline
nlp = pipeline('question-answering', model="deepset/roberta-base-squad2")

# Path to the PDF file
pdf_path = "/content/Cfg.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf_doc:
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to record audio
def record_audio(sec=3):
    print("Speak Now...")
    output.eval_js('''
    const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
    const b2text = blob => new Promise(resolve => {
      const reader = new FileReader()
      reader.onloadend = e => resolve(e.srcElement.result)
      reader.readAsDataURL(blob)
    })
    var record = time => new Promise(async resolve => {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      recorder = new MediaRecorder(stream)
      chunks = []
      recorder.ondataavailable = e => chunks.push(e.data)
      recorder.start()
      await sleep(time)
      recorder.onstop = async ()=>{
        blob = new Blob(chunks)
        text = await b2text(blob)
        resolve(text)
      }
      recorder.stop()
    })
    ''')
    audio_data_b64 = output.eval_js(f'record({sec*1000})')
    audio_data = b64decode(audio_data_b64.split(',')[1])
    print("Done Recording !")
    return audio_data

# Function to recognize speech
def recognize_speech(audio_data):
    # Using pipeline
    result_pipe = asr_pipe(audio_data)
    print("Pipeline Result:", result_pipe)  # Print the entire result to examine its structure

    if result_pipe and 'text' in result_pipe:
        transcription_pipe = result_pipe['text']
        print("Using Pipeline:", transcription_pipe)
        return transcription_pipe
    else:
        print("Unable to extract transcription from the pipeline result.")
        return ""

# Function to answer question from PDF
def answer_question_interface(btn):
    # Record audio and convert it to text
    oral_question_audio = record_audio(sec=5)
    oral_question_text = recognize_speech(oral_question_audio)

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Set up the question and context for the question-answering model
    QA_input = {
        'question': oral_question_text,
        'context': pdf_text
    }

    # Get predictions
    result = nlp(QA_input)

    # Access the answer
    answer = result['answer']

    # Display the answer
    answer_output.value = f"Answer: {answer}"

# Create a button for starting the process
record_button = widgets.Button(description="Record and Answer")
record_button.on_click(answer_question_interface)

# Create an output widget to display the answer
answer_output = widgets.Output()

# Display the button and output widgets
display(record_button)
display(answer_output)
