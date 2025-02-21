from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a lightweight and fast QA model
chatbot = pipeline("question-answering", model="deepset/roberta-base-squad2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")

    try:
        # Provide a static context to make the model answer health questions better
        context = (
            "Toothaches can be relieved using warm salt water rinses, clove oil, or over-the-counter pain relievers. "
            "For fever, rest, hydration, and paracetamol can help. "
            "For stomach aches, ginger tea and hydration are good remedies. "
            "If symptoms persist, consult a doctor."
        )

        response = chatbot(question=user_message, context=context)
        return jsonify({"response": response['answer']})
    except Exception as e:
        return jsonify({"response": "I'm sorry, I couldn't process that. Please try again!"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
