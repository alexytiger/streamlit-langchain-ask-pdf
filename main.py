
#
# pip install python-dotenv
# pip install streamlit
# pip install langchain
# pip install openai
# pip install PyPDF2
# pip install tiktoken
# pip install faiss-cpu

# Standard libraries
import os
import io
import base64

# Third-party libraries
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# LangChain-related imports (assuming they're third-party)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI

# Your own modules
from htmlTemplates import css, bot_template

 
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image_path = 'img/ai-robot-2.jpg'
image_base64 = get_image_base64(image_path)

bot_message = bot_template.replace("{{IMAGE}}", f'data:image/png;base64,{image_base64}')


# Create a function to split the text into chunks
def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
     # Remove empty chunks
     # to understand this code her eis the explanation
     # numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
     # squared_evens = [x**2 for x in numbers if x % 2 == 0]
# Explanation:

# x**2: This is the expression that will be evaluated for each item that meets the condition. 
# It squares the number.
# for x in numbers: This iterates over each item in the numbers list and assigns it to the variable x.
# if x % 2 == 0: This is the condition. It checks if x is even. If the condition is true, 
# the expression (x**2) will be evaluated and its result will be added to the new list. 
# If the condition is false, the item is skipped.
# result:
# [4, 16, 36, 64, 100]
    return [chunk for chunk in chunks if chunk]


# Function to load the data from the pdf
def get_pdf_text(uploaded_pdf):
    text = ""

    try:
        # Set up the pdf reader
        pdf_reader = PdfReader(uploaded_pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if the text extraction was successful for this page
                text += page_text
            else:
                # Handle pages where text extraction failed
                print("Failed to extract text from one of the pages. The output might be incomplete.")
    except Exception as e:
        st.error(f"An error occurred while extracting text from the PDF: {str(e)}")
        return None

    return text


def get_vector_database(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Debug: Check the embeddings
    print("Embeddings:", embeddings)
    
    # Debug: Check the length of each embedding
    embeddings_lengths = [len(embed) for embed in embeddings]
    print("Embedding Lengths:", embeddings_lengths)
    
    try:
        vector_database = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"An error occurred while initializing the vector database: {str(e)}")
        return None
    
    return vector_database

def is_question_meaningful (question):
    # Check for minimum length
    if len(question) < 3:
        return False
    
    # Check if the input is not just whitespace or special characters
    if question.strip() == "" or all(not char.isalnum() for char in question):
        return False

    return True


def main():
    # load our environment to read secrets
    load_dotenv()

# This function sets the configuration options of the Streamlit's page.
# Here, page_title="Ask your CSV" changes the default page title from "Streamlit" to "Ask your CSV".
# The browser tab reflects this change.
# It's the first thing that runs when a Streamlit app starts up.
    st.set_page_config(page_title="Ask your PDF", page_icon="🐶")
    st.write(css, unsafe_allow_html=True)

    st.header("Ask your PDF")
    
    # Load the OpenAI API key from the environment variable   
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("The OpenAI API key is not set. Please set it before proceeding.")
        return # exit(1)
    else:
        st.toast("OpenAI API key is set and validated!")

    
    # Sidebar contents
    with st.sidebar:
        st.title('LLM PDF Chat App')
        
        st.subheader("About")
   
        # Markdown Hyperlinks: [Link Text](URL) is the syntax to create a hyperlink in Markdown.
        st.write("""
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models)
        - [QA over Document](https://python.langchain.com/docs/use_cases/question_answering/)
        """)
        
        robot_tiger = Image.open('img/ai-tiger-robot.jpeg')
        st.image(robot_tiger)
      
        
        for _ in range(4):
            st.write("\n")
        st.markdown('<b>Made by @Bad Tiger</b>', unsafe_allow_html=True)
    
    
    pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if pdf:
        # Check if the uploaded file is truly a PDF by inspecting the first few bytes
        header = pdf.read(5).decode('latin-1')  # Read the first 5 bytes
        if header != "%PDF-":
            st.error("The uploaded file doesn't appear to be a valid PDF. Please upload a proper PDF file.")
            return
    
        pdf.seek(0)  # Reset file pointer after reading
        
        st.success("PDF file uploaded successfully!")
    else:
        st.error("No file uploaded!") 
        return
    
#important notice:
# Yes, the Streamlit app will continue to run even if `return` is used in any function, 
 # including your `main()` function.

#The important thing to understand here is that the `return` statement in Python 
# just exits the current function and hands the program execution back to its caller. 

#In the context of Streamlit, when a user interacts with an input widget (like your file uploader), 
# the entire script gets rerun from top to bottom. 
# So after returning from `main()`, Streamlit will immediately start over again 
# from the top of your script, waiting for the next interaction.

#Here's a simple way to visualize it:

#'''
#def main():
    # some logic here
#    return

# This part keeps running despite return in main()
#if __name__ == "__main__":
#    while True:
#        main()

#'''


#So, while the `return` statement does end the `main()` function prematurely, 
# the Streamlit application will not stop running due to the nature of the framework's execution model. 
# It will still be active, waiting for further user interactions.
    
    
   # read data from the file and put them into a variable called text
   # get pdf text
    st.info("Start processing the uploaded PDF...")
    with st.spinner(text="Wait for it..."):
        text = get_pdf_text(pdf)

    # This will catch both None and an empty string, as both are considered "falsy" in Python.
    if not text:
        st.error("Please try to upload a different PDF file. This one seems has a problems")
        return
    else:
        st.success("PDF processed successfully!")
        
    # Splitting up the text into smaller chunks for indexing
    chunks = get_text_chunks(text, 1000, 200)
    
    # get vector database
    knowledge_base =  get_vector_database(chunks)
    if knowledge_base is None:
        st.error("Failed to initialize the knowledge base.")
        return
    
    user_question = st.text_input("Ask a question about your PDF: ")

    if user_question is not None and is_question_meaningful(user_question):
        with st.spinner(text="In progress..."):
            docs = knowledge_base.similarity_search(user_question)
    
            # yes, openai = OpenAI(model_name="text-davinci-003") is the same as openai = OpenAI(engine="davinci"). 
            # which is the default
            # The model_name parameter specifies the name of the LLM engine to use. And the engine parameter is just a shortcut for the model_name parameter.
            # In this case, text-davinci-003 is the name of the LLM engine that is used by OpenAI. 
            # It is the third version of the davinci engine, which is the most powerful engine available with the OpenAI class.
            # So, if you want to use the davinci engine, you can use either model_name="text-davinci-003" or engine="davinci".
            
            llm = OpenAI(temperature=0)
            
            # The chain_type parameter in the load_qa_chain() function specifies the type of chain to use. 
            # The stuff chain type is a new chain type that was introduced in LangChain 0.10.0. 
            # It is a more efficient way to train and use chains for question answering tasks.
            # The stuff chain type works by breaking the text of the documents into smaller chunks, 
            # and then using the LLM to generate a response for each chunk. 
            # The responses for the chunks are then stitched together to create a final response to the question.
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # The get_openai_callback() method returns a callback object 
            # that can be used to track the progress of the run() method. 
            # The callback object will be called periodically during the execution of the run() method 
            # and will provide information about the progress of the method.

            # The callback object has the following methods:

            # on_begin(): This method is called when the run() method starts.
            # on_progress(): This method is called periodically during the execution of the run() method.
            # on_done(): This method is called when the run() method finishes.
            with get_openai_callback() as cb:
                try:
                    response = chain.run(input_documents=docs, question=user_question, callback=cb)
                except Exception as e:
                    st.error("An error occurred: {}".format(str(e)))
                    
            st.write(bot_message.replace(
                        "{{MSG}}", response), unsafe_allow_html=True)
    else:
        st.warning("Please enter a meaningful question.")
            
  
# def main():
#     print("Hello world!")

# # Other function definitions
# def foo():
#     print("foo!")

# # ...

# if __name__ == "__main__":
#     main()  # This is called only when the script is run directly.

# In this case, if main.py is executed directly like so: python main.py,
# then Hello world! will be printed onto the console because the __name__ variable
# for the script will equal to "__main__", thus calling the main() function.

# But if you import main.py in another Python script using import main,
# no text is printed. However, the functions main() and foo() are now available for use in that script.



if __name__ == "__main__":
    main()
