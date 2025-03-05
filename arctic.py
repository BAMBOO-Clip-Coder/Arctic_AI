
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import random
import sympy
from sympy import cos, sin, tan, acos, asin, atan, pi, degree

from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

# Install required libraries:
# pip install langchain duckduckgo-search sentence-transformers faiss-cpu huggingface_hub

# Set your Hugging Face API token (required)
huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not huggingfacehub_api_token:
    huggingfacehub_api_token = input("Please enter your Hugging Face API token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token

def create_retriever(query):
    """Creates a retriever from DuckDuckGo search results."""
    search = DuckDuckGoSearchAPIWrapper()
    results = search.results(query, max_results=5)

    if not results:
        return None

    documents = []
    for result in results:
        if 'body' in result and 'link' in result and result['body']:
            documents.append(Document(page_content=result["body"], metadata={"source": result["link"]}))
        elif 'snippet' in result and 'link' in result:
            documents.append(Document(page_content=result['snippet'], metadata={'source': result['link']}))

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()

def run_conversation(query, chat_history, output_text, link_text):
    """Runs a conversational retrieval chain."""
    try:
        retriever = create_retriever(query)
        if not retriever:
            messagebox.showinfo("Info", "No relevant information found.")
            return

        repo_id = "google/flan-t5-large"
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1024})

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

        result = qa({"question": query})

        answer = result["answer"]
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, answer + "\n")

        # Extract and display links (if applicable)
        if retriever:
            retrieved_docs = retriever.get_relevant_documents(query)
            links = [doc.metadata["source"] for doc in retrieved_docs]
            link_text.delete(1.0, tk.END)
            for link in links:
                link_text.insert(tk.END, link + "\n")
        else:
            link_text.delete(1.0, tk.END)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def extract_math_expression(text):
    """Extracts a math expression from a given text."""
    math_patterns = [
        r"(\d+(\.\d+)?)\s*[\+\-\*/\^]\s*(\d+(\.\d+)?)",
        r"sqrt\(\s*(\d+(\.\d+)?)\s*\)",
        r"(\d+(\.\d+)?)\s*to the power of\s*(\d+(\.\d+)?)",
        r"what is (\S+)",
        r"calculate (\S+)",
        r"solve (\S+)",
        r"find (\S+)",
        r"sin\(\s*(\S+)\s*\)",
        r"cos\(\s*(\S+)\s*\)",
        r"tan\(\s*(\S+)\s*\)",
        r"arcsin\(\s*(\S+)\s*\)",
        r"arccos\(\s*(\S+)\s*\)",
        r"arctan\(\s*(\S+)\s*\)",
        r"(\d+)\s*degrees",
    ]

    for pattern in math_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    return None

def solve_word_problem(text):
    """Attempts to solve word problems by extracting and solving math expressions."""
    math_expressions = re.findall(r"(\d+(\.\d+)?)\s*[\+\-\*/\^]\s*(\d+(\.\d+)?)", text)
    if math_expressions:
        try:
            expression_str = " ".join([f"{x[0]} {x[2]}" for x in math_expressions])
            result = sympy.sympify(expression_str)
            if isinstance(result, sympy.Expr):
                result = sympy.simplify(result)
            return str(result)
        except:
            return None
    else:
        return None

def conversational_math_bot(query, conversation_history=None):
    """A more talkative conversational math bot using SymPy."""

    if conversation_history is None:
        conversation_history = []

    math_expression = extract_math_expression(query)
    word_problem_result = solve_word_problem(query)

    if math_expression:
        try:
            # Handle degrees conversion
            query_degrees = re.search(r"(\d+)\s*degrees", query, re.IGNORECASE)
            if query_degrees:
                degrees = float(query_degrees.group(1))
                query = query.replace(query_degrees.group(0), str(pi * degrees / 180))

            cleaned_expression = re.sub(r'[^\w\s\+\-\*/\(\)\.\^]', '', math_expression)
            result = sympy.sympify(cleaned_expression).evalf()

            if isinstance(result, sympy.Expr):
                result = sympy.simplify(result)

            # Convert float to fraction
            response = str(result.as_numer_denom())
            conversation_history.append((query, response))

            math_phrases = [
                f"The answer to {math_expression} is {response}!",
                f"For {math_expression}, the result is {response}.",
                f"After calculating {math_expression}, I got {response}.",
            ]
            return random.choice(math_phrases), conversation_history

        except (sympy.SympifyError, TypeError, SyntaxError, ZeroDivisionError):
            error_phrases = [
                "Hmm, I'm having trouble with that calculation.",
                "I couldn't quite understand that math expression.",
                "There seems to be an error in the mathematical part of your query.",
            ]
            conversation_history.append((query, "I'm sorry, I couldn't understand or solve that."))
            return random.choice(error_phrases), conversation_history
        except Exception as e:
            conversation_history.append((query, f"An unexpected error occurred: {e}"))
            return f"An unexpected error occurred: {e}", conversation_history
    elif word_problem_result:
        conversation_history.append((query, word_problem_result))
        return f"The answer to your word problem is: {word_problem_result}", conversation_history
    else:
        non_math_responses = [
            "I'm a math bot, so I'm better at numbers. Can you ask me a math question?",
            "I'm not sure how to answer that. Could you ask me a math question?",
            "I can only answer math questions. Please provide a mathematical expression."
        ]
        return random.choice(non_math_responses), conversation_history

def run_math_bot(query, chat_history, output_text):
    """Runs the math bot."""
    response, chat_history = conversational_math_bot(query, chat_history)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, response + "\n")
    return chat_history

def run_search():
    query = query_entry.get()
    if math_mode:
        global chat_history_math
        chat_history_math = run_math_bot(query, chat_history_math, output_text)
    else:
        global chat_history
        run_conversation(query, chat_history, output_text, link_text)
    query_entry.delete(0, tk.END)

def toggle_mode():
    global math_mode
    math_mode = not math_mode
    search_button.config(text="Math" if math_mode else "Search")

# Create the main window
window = tk.Tk()
window.title("Conversational Bot")
window.configure(bg="#333333")  # Set background to dark gray

# Create a label and entry for the query
query_label = tk.Label(window, text="Enter your message:", bg="#333333", fg="white")
query_label.pack(pady=10)

query_entry = tk.Entry(window, width=50, bg="#444444", fg="white", highlightbackground="#666666", highlightthickness=1)
query_entry.pack(pady=10)

# Create a button to run the search
search_button = tk.Button(window, text="Search", command=run_search, bg="#444444", fg="white", highlightbackground="#666666", highlightthickness=1)
search_button.pack(pady=10)

# Create a button to toggle mode
mode_button = tk.Button(window, text="Math / Search", command=toggle_mode, bg="#444444", fg="white", highlightbackground="#666666", highlightthickness=1)
mode_button.pack(pady=10)

# Create a scrolled text widget for the answer
answer_frame = tk.Frame(window, bg="#333333")
answer_frame.pack(pady=10)
answer_label = tk.Label(answer_frame, text="Response:", bg="#333333", fg="white")
answer_label.pack()
output_text = scrolledtext.ScrolledText(answer_frame, width=80, height=15, bg="#444444", fg="white", highlightbackground="#666666", highlightthickness=1)
output_text.pack()

# Create a scrolled text widget for the links
link_frame = tk.Frame(window, bg="#333333")
link_frame.pack(pady=10)
link_label = tk.Label(link_frame, text="Links:", bg="#333333", fg="white")
link_label.pack()
link_text = scrolledtext.ScrolledText(link_frame, width=80, height=5, bg="#444444", fg="white", highlightbackground="#666666", highlightthickness=1)
link_text.pack()

chat_history = []
chat_history_math = []
math_mode = False

window.mainloop()

#pip install langchain duckduckgo-search sentence-transformers faiss-cpu huggingface_hub
#pip install -U langchain-community
#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
