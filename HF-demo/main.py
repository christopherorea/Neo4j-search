from transformers import AutoTokenizer, TFAutoModel
from flask import Flask, request

model_id = 'mixedbread-ai/mxbai-embed-large-v1'
print('loading model')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = TFAutoModel.from_pretrained(model_id)
print('model loaded')

app = Flask(__name__)

@app.route("/", methods=['GET'])
def query():
    query= request.args.get('query')
    if query:
        inputs = tokenizer(query, return_tensors="tf")
        model_inputs = model(inputs)
        return {"embedding": model_inputs[1:][0].numpy()[0].tolist()}, 200
    else:
       return "No query found", 404
    

@app.route("/check", methods=['GET'])
def health_check():
    return '', 200

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)