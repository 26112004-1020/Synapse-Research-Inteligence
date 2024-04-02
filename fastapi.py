from flask import Flask, request, jsonify
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer

quantized_model_dir = "AVMLegend/Quantised-Phi-2" 
model = AutoModelForCausalLM.from_pretrained(quantized_model_dir, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)


def search_papers(term, num_papers=10):
    url = f"https://api.semanticscholar.org/v1/paper/search?query={term}&limit={num_papers}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print("Failed to fetch papers")
        return []  


def generate_summaries(papers):
    summaries = []
    for paper in papers:
        title = paper["title"]
        abstract = paper.get("abstract", "")  # Handle potential absence of abstract
        prompt = f"{title}. {abstract}"
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_new_tokens=150)
        summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        summaries.append((title, summary))
    return summaries


app = Flask(__name__)


@app.route("/search", methods=["GET"])
def search_and_summarize():
    search_term = request.args.get("term")  

    if not search_term:
        return jsonify({"error": "Missing search term"}), 400 

    papers = search_papers(search_term)

    if not papers:
        return jsonify({"message": "No papers found for the given term."}), 200 

    paper_summaries = generate_summaries(papers)
    response = {"summaries": paper_summaries}

    return jsonify(response), 200 


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")  
