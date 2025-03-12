import os
import random
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext, font
from tkinter import ttk
import sympy
from sympy import simplify, pi
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import DuckDuckGoSearchAPIWrapper, GoogleSearchAPIWrapper
import webbrowser
import threading
import groq

huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not huggingfacehub_api_token:
    huggingfacehub_api_token = input("Please enter your Hugging Face API token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token

google_api_key = os.environ.get("GOOGLE_API_KEY")
google_cse_id = os.environ.get("GOOGLE_CSE_ID")

if not google_api_key or not google_cse_id:
    print("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
    google_api_key = input("Please enter your google api key: ")
    google_cse_id = input("Please enter your google cse id: ")
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_CSE_ID"] = google_cse_id

def create_retriever(query):
    duckduckgo_search = DuckDuckGoSearchAPIWrapper()
    google_search = GoogleSearchAPIWrapper()
    duckduckgo_results = duckduckgo_search.results(query, max_results=3)
    google_results = google_search.results(query, 2)
    results = duckduckgo_results + google_results

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

def detect_code_query(query):
    """Check if the query is asking for code examples in specific languages."""
    code_keywords = [
        "code", "example", "script", "program", "function", "class", 
        "implement", "develop", "create", "build", "write"
    ]
    
    languages = {
        "python": ["python", "py", "django", "flask", "numpy", "pandas"],
        "javascript": ["javascript", "js", "node", "react", "angular", "vue", "jquery"],
        "html": ["html", "markup", "web page", "webpage"],
        "css": ["css", "style", "styling", "stylesheet"],
        "java": ["java", "spring", "maven", "gradle"],
        "csharp": ["c#", "csharp", ".net", "dotnet", "asp.net", "unity"],
        "cpp": ["c++", "cpp", "opencv", "qt"]
    }
    
    # Check if query contains code-related keywords
    has_code_keyword = any(keyword in query.lower() for keyword in code_keywords)
    
    if not has_code_keyword:
        return False, []
        
    # Detect which languages are mentioned
    detected_languages = []
    for lang, keywords in languages.items():
        if any(keyword in query.lower() for keyword in keywords):
            detected_languages.append(lang)
            
    return has_code_keyword, detected_languages

def run_conversation(query, chat_history, output_text, link_text):
    try:
        is_code_query, languages = detect_code_query(query)
        
        retriever = create_retriever(query)
        if not retriever:
            messagebox.showinfo("Info", "No relevant information found.")
            return

        try:
            # Enhance the query with specific instructions for longer responses if it's not a code query
            enhanced_query = query
            if not is_code_query:
                enhanced_query = f"""
                Please provide a detailed and comprehensive response to the following query: 
                {query}
                
                Your response should be lengthy, informative and well-structured with multiple paragraphs.
                Include examples, explanations, and relevant details.
                """
            elif languages:
                language_str = ", ".join(languages)
                enhanced_query = f"""
                Please provide a detailed explanation AND code examples in the following languages: {language_str}
                for this query: {query}
                
                For each language, include well-commented code with explanations.
                Format code blocks properly with ```language and ``` syntax.
                """
            
            client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": enhanced_query}], 
                model="mixtral-8x7b-32768"
            )
            answer = chat_completion.choices[0].message.content
        except Exception as groq_error:
            print(f"Groq API error: {groq_error}")
            repo_id = "google/flan-t5-large"
            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 4096})
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
            result = qa({"question": enhanced_query})
            answer = result["answer"]

        # Format the answer to be more verbose and structured
        formatted_answer = format_answer(answer, is_code_query)

        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, formatted_answer + "\n")
        
        # Add syntax highlighting for code blocks
        highlight_code_blocks(output_text)
        
        output_text.config(state=tk.DISABLED)

        if retriever:
            retrieved_docs = retriever.get_relevant_documents(query)
            links = [doc.metadata["source"] for doc in retrieved_docs]
            link_text.config(state=tk.NORMAL)
            link_text.delete(1.0, tk.END)
            for link in links:
                link_text.insert(tk.END, link + "\n", "hyper")
            link_text.config(state=tk.DISABLED)
        else:
            link_text.config(state=tk.NORMAL)
            link_text.delete(1.0, tk.END)
            link_text.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def highlight_code_blocks(text_widget):
    """Apply syntax highlighting to code blocks."""
    content = text_widget.get(1.0, tk.END)
    
    # Find all code blocks with ```language and ``` syntax
    code_blocks = re.finditer(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
    
    # Define tag configurations for different languages
    text_widget.tag_configure("code_block", background="#f5f5f5", font=("Courier", 10))
    text_widget.tag_configure("python", foreground="#306998")
    text_widget.tag_configure("javascript", foreground="#f0db4f")
    text_widget.tag_configure("html", foreground="#e34c26")
    text_widget.tag_configure("css", foreground="#2965f1")
    text_widget.tag_configure("java", foreground="#b07219")
    text_widget.tag_configure("csharp", foreground="#178600")
    text_widget.tag_configure("cpp", foreground="#f34b7d")
    
    # Apply tags to each code block
    for match in code_blocks:
        start_index = f"1.0 + {match.start()} chars"
        end_index = f"1.0 + {match.end()} chars"
        
        # Apply general code block formatting
        text_widget.tag_add("code_block", start_index, end_index)
        
        # Apply language-specific formatting if a language is specified
        language = match.group(1)
        if language and language.lower() in ["python", "javascript", "html", "css", "java", "csharp", "cpp"]:
            text_widget.tag_add(language.lower(), start_index, end_index)

def format_answer(answer, is_code_query=False):
    """Format the answer to be longer and preserve code blocks."""
    # Don't modify the structure of code responses
    if is_code_query:
        return answer
    
    # Check if the answer already contains code blocks
    if re.search(r'```\w*\n', answer):
        return answer
        
    # Split the answer into sentences and create paragraphs
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)
        if len(current_paragraph) >= 3:  # Smaller paragraphs for better readability
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    # Return all paragraphs, not just the first 3
    return '\n\n'.join(paragraphs)

def solve_word_problem(text):
    try:
        symbols = re.findall(r'[a-zA-Z]+', text)
        if symbols:
            sympy_symbols = ' '.join(list(set(symbols)))
            sympy_symbols = sympy.symbols(sympy_symbols)
            result = sympy.solve(text, sympy_symbols)

            if result:
                if isinstance(result, list) and len(result) == 1:
                    return str(result[0])
                else:
                    return str(result)
        else:
            # Handle simple math expressions without variables
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
            return None
    except (sympy.SympifyError, TypeError, SyntaxError, NotImplementedError):
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

def extract_math_expression(query):
    """Extracts a math expression from the query using regular expressions."""
    # This regex pattern captures a basic math expression
    # It looks for numbers, operators (+, -, *, /, ^), parentheses, and decimal points
    pattern = r"[\d\+\-\*/\^\(\)\.]+"
    match = re.search(pattern, query)
    if match:
        return match.group(0)
    return None

def conversational_math_bot(query, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    math_expression = extract_math_expression(query)
    word_problem_result = solve_word_problem(query)

    if math_expression:
        try:
            query_degrees = re.search(r"(\d+)\s*degrees", query, re.IGNORECASE)
            if query_degrees:
                degrees = float(query_degrees.group(1))
                query = query.replace(query_degrees.group(0), str(pi * degrees / 180))

            cleaned_expression = re.sub(r'[^\w\s\+\-\*/\(\)\.\^]', '', math_expression)
            
            # Try to directly evaluate the expression first
            try:
                result = sympy.sympify(cleaned_expression)
                
                if isinstance(result, sympy.Expr):
                    result = sympy.simplify(result)

                    if isinstance(result, sympy.logic.boolalg.Boolean):
                        response = str(result)
                    else:
                        response = str(result.evalf(n=5))

                    conversation_history.append((query, response))

                    math_phrases = [
                        f"The answer to {math_expression} is {response}!",
                        f"For {math_expression}, the result is {response}.",
                        f"After calculating {math_expression}, I got {response}.",
                    ]
                    return random.choice(math_phrases), conversation_history

                else:
                    response = str(result)
                    conversation_history.append((query, response))
                    return f"The answer is {response}", conversation_history

            except (sympy.SympifyError, TypeError, SyntaxError) as e:
                print(f"Error in direct evaluation: {e}")
                # If direct evaluation fails, fall back to word problem solution
                if word_problem_result:
                    conversation_history.append((query, word_problem_result))
                    return f"The answer to your expression is: {word_problem_result}", conversation_history
                else:
                    raise e  # Re-raise if word problem solution also failed

        except (sympy.SympifyError, TypeError, SyntaxError, ZeroDivisionError) as e:
            error_phrases = [
                "Hmm, I'm having trouble with that calculation.",
                "I couldn't quite understand that math expression.",
                "There seems to be an error in the mathematical part of your query.",
                f"Error details: {e}"
            ]
            conversation_history.append((query, "I'm sorry, I couldn't understand or solve that."))
            return random.choice(error_phrases), conversation_history
        except Exception as e:
            print(f"Unexpected error in math bot: {e}")
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
    print("run_math_bot called with query:", query)
    response, updated_chat_history = conversational_math_bot(query, chat_history)
    print("Response from conversational_math_bot:", response)
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, response + "\n")
    output_text.config(state=tk.DISABLED)
    return updated_chat_history  # Return the updated chat history

def run_search():
    query = query_entry.get()
    
    if not query or query.strip() == "" or query == "Explore Something":
        return
    
    # Reposition UI elements after first query
    if initial_state:
        transition_to_conversation_layout()
    
    if math_mode:
        global chat_history_math 
        if not chat_history_math:  # Initialize if it's empty
            chat_history_math = []
        chat_history_math = run_math_bot(query, chat_history_math, answer_text)
    else:
        global chat_history
        run_conversation(query, chat_history, answer_text, link_text)
    
    query_entry.delete(0, tk.END)
    query_entry.insert(0, "Explore Something")

def transition_to_conversation_layout():
    global initial_state
    initial_state = False
    
    # Hide the initial centered input
    initial_frame.pack_forget()
    
    # Show the main layout
    main_content_frame.pack(fill=tk.BOTH, expand=True)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

def toggle_mode(mode):
    global math_mode
    math_mode = mode
    info_button.config(style="TButton" if not math_mode else "Active.TButton")
    math_button.config(style="TButton" if math_mode else "Active.TButton")

def clear_text():
    query_entry.delete(0, tk.END)
    query_entry.insert(0, "Explore Something")
    answer_text.config(state=tk.NORMAL)
    answer_text.delete(1.0, tk.END)
    answer_text.config(state=tk.DISABLED)
    link_text.config(state=tk.NORMAL)
    link_text.delete(1.0, tk.END)
    link_text.config(state=tk.DISABLED)

def copy_answer():
    answer = answer_text.get(1.0, tk.END)
    window.clipboard_clear()
    window.clipboard_append(answer)
    window.update()

def open_link(event):
    link_text.config(state=tk.NORMAL)
    index = link_text.tag_ranges("sel")
    if index:
        start_index = index[0]
        end_index = index[1]
        link = link_text.get(start_index, end_index).strip()
        webbrowser.open_new_tab(link)
    link_text.config(state=tk.DISABLED)

def clear_placeholder(event):
    if query_entry.get() == "Explore Something":
        query_entry.delete(0, tk.END)

def restore_placeholder(event):
    if not query_entry.get():
        query_entry.insert(0, "Explore Something")

# --- UI Setup ---
window = tk.Tk()
window.title("Conversational Bot")
window.configure(bg="white")
window.geometry("900x700")  # Set initial window size

style = ttk.Style()
style.theme_use("clam")

style.configure("TButton", background="white", foreground="black", borderwidth=1, 
                relief="solid", padding=8, font=("Arial", 10))
style.configure("Active.TButton", background="#f0f0f0", foreground="black", font=("Arial", 10, "bold"))
style.map("TButton", background=[("active", "#e0e0e0")])

# Create frames for different UI states
initial_frame = tk.Frame(window, bg="white")  # Initial state with centered input
main_content_frame = tk.Frame(window, bg="white")  # Main content area
bottom_frame = tk.Frame(window, bg="white", height=60)  # Bottom bar for input

# Create the response display areas
response_frame = tk.Frame(main_content_frame, bg="white")
response_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

answer_label = tk.Label(response_frame, text="Response", bg="white", font=("Arial", 12, "bold"))
answer_label.pack(anchor="w", pady=(0, 5))

answer_text = scrolledtext.ScrolledText(response_frame, width=60, height=20, bg="white", fg="black", 
                                      borderwidth=1, highlightthickness=0, relief="solid", font=("Arial", 10))
answer_text.pack(fill=tk.BOTH, expand=True)
answer_text.config(state=tk.DISABLED)

link_frame = tk.Frame(main_content_frame, bg="white", width=200)
link_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
link_frame.pack_propagate(False)  # Prevent frame from shrinking

link_label = tk.Label(link_frame, text="Sources", bg="white", font=("Arial", 12, "bold"))
link_label.pack(anchor="w", pady=(0, 5))

link_text = scrolledtext.ScrolledText(link_frame, width=25, height=20, bg="white", fg="black", 
                                    borderwidth=1, highlightthickness=0, relief="solid", font=("Arial", 10))
link_text.pack(fill=tk.BOTH, expand=True)
link_text.config(state=tk.DISABLED)

link_text.tag_config("hyper", foreground="blue", underline=1)
link_text.tag_bind("hyper", "<Button-1>", open_link)

# Create the bottom input bar
input_frame = tk.Frame(bottom_frame, bg="white")
input_frame.pack(fill=tk.X, padx=10, pady=10)

query_entry = tk.Entry(input_frame, width=50, bg="white", fg="black", insertbackground="black",
                    borderwidth=1, highlightthickness=1, relief="solid", font=("Arial", 12),
                    highlightbackground="#cccccc", highlightcolor="#999999")
query_entry.insert(0, "Explore Something")
query_entry.bind("<FocusIn>", clear_placeholder)
query_entry.bind("<FocusOut>", restore_placeholder)
query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)  # Add internal padding

enter_button = ttk.Button(input_frame, text="Enter", command=run_search)
enter_button.pack(side=tk.LEFT, padx=5)

clear_button = ttk.Button(input_frame, text="Clear", command=clear_text)
clear_button.pack(side=tk.LEFT)

# Mode toggle buttons
buttons_frame = tk.Frame(bottom_frame, bg="white")
buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

mode_label = tk.Label(buttons_frame, text="Mode:", bg="white", font=("Arial", 10))
mode_label.pack(side=tk.LEFT, padx=(0, 5))

info_button = ttk.Button(buttons_frame, text="Informational", command=lambda: toggle_mode(False), style="Active.TButton")
math_button = ttk.Button(buttons_frame, text="Mathematical", command=lambda: toggle_mode(True), style="TButton")
info_button.pack(side=tk.LEFT, padx=5)
math_button.pack(side=tk.LEFT)

copy_button = ttk.Button(buttons_frame, text="Copy Answer", command=copy_answer)
copy_button.pack(side=tk.RIGHT)

# Create the initial centered input
initial_query_frame = tk.Frame(initial_frame, bg="white")
initial_query_frame.pack(expand=True)

initial_query_entry = tk.Entry(initial_query_frame, width=50, bg="white", fg="black", insertbackground="black",
                           borderwidth=1, highlightthickness=1, relief="solid", font=("Arial", 12),
                           highlightbackground="#cccccc", highlightcolor="#999999")
initial_query_entry.insert(0, "Explore Something")
initial_query_entry.bind("<FocusIn>", clear_placeholder)
initial_query_entry.bind("<FocusOut>", restore_placeholder)
initial_query_entry.pack(side=tk.LEFT, ipady=8)  # Add internal padding

# Connect the initial entry to the main entry for synchronization
def sync_entry(*args):
    if initial_state:
        query_entry.delete(0, tk.END)
        query_entry.insert(0, initial_query_entry.get())
    else:
        initial_query_entry.delete(0, tk.END)
        initial_query_entry.insert(0, query_entry.get())

initial_query_entry.bind("<KeyRelease>", sync_entry)
query_entry.bind("<KeyRelease>", sync_entry)

# Connect Enter key to run_search
initial_query_entry.bind("<Return>", lambda event: run_search())
query_entry.bind("<Return>", lambda event: run_search())

initial_enter_button = ttk.Button(initial_query_frame, text="Enter", command=run_search)
initial_enter_button.pack(side=tk.LEFT, padx=5)

initial_clear_button = ttk.Button(initial_query_frame, text="Clear", command=clear_text)
initial_clear_button.pack(side=tk.LEFT)

# Start with the initial layout
initial_frame.pack(fill=tk.BOTH, expand=True)

# Track UI state
initial_state = True
chat_history = []
chat_history_math = []
math_mode = False

window.mainloop()
#pip install langchain duckduckgo-search sentence-transformers faiss-cpu huggingface_hub google-api-python-client sympy groq -U langchain-community
#python file_path_here
#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
#AIzaSyCa4_KLTmH6jiFMzBrpv2JIk73CB6IMJ3w
#a624aee9e518143df
