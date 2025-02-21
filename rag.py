from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
import os
from dotenv import load_dotenv

class RAG:
    def __init__(self, input, openai_api_key):
        """Initializes the RAG pipeline, including document processing, retrieval, and LLM setup."""
        self.input = input
        self.openai_api_key = openai_api_key
        
        # Load and split PDF documents
        self.loader = PyPDFDirectoryLoader("data/")
        self.pages = self.loader.load_and_split()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.documents = self.text_splitter.split_documents(self.pages)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize FAISS vector store
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            print("Loading existing FAISS index...")
            self.vector = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS index...")
            self.vector = FAISS.from_documents(
                self.documents,
                self.embeddings
            )
            # Save the FAISS index
            self.vector.save_local(index_path)
        
        # Set up the retriever
        self.retriever = self.vector.as_retriever()
        
        # Set up the LLM
        self.llm = ChatOpenAI(openai_api_key=openai_api_key)
        
        # Set up the output parser
        self.output_parser = StrOutputParser()
        
        # Set up the prompts
        self._setup_prompts()
        
        # Create the chains
        self._setup_chains()

    def _setup_prompts(self):
        """Defines the question reformulation and QA prompts."""

        self.instruction_to_system = """
        Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question.
        DO NOT answer the question, just reformulate it if needed and otherwise return it as it is"""
        
        self.question_maker_prompt = ChatPromptTemplate.from_messages([
            ("system", self.instruction_to_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Main QA prompt
        self.qa_system_prompt = """You are an intelligent and highly precise Software Engineering Course Assistant for COMP3401 at SQU. 
            Your primary role is to analyze multiple text-based PDFs (including important tables and occasional images) and extract 
            **only** the exact answers based on user queries.

            ### 🔹 Rules & Behavior:
            1️⃣ **Strict Answer Extraction**:  
            - Respond **exclusively** using the content from the provided PDFs.  
            - **Do NOT generate** any information beyond what is explicitly stated in the documents.  

            2️⃣ **Smart PDF Selection**:  
            - Analyze the question carefully.  
            - Choose the **most relevant PDF(s)** that contain the correct answer.  

            3️⃣ **Accurate Data Handling**:  
            - Extract **tables accurately** and format them properly.  
            - Ensure important tabular data is **clear and well-structured** in your response.  

            4️⃣ **Strict Relevance**:  
            - If the answer **cannot** be found in the PDFs, clearly state:  
                **"I cannot find this information in the provided documents."**  

            5️⃣ **Citations & Transparency**:  
            - Always include the **exact source** of your response in this format:  
                **"Answer found in [Document Name], Page X."**  

            6️⃣ **Handling Conflicting Information**:  
            - If multiple PDFs provide **different answers**, list **all conflicting responses**.  
            - Mention discrepancies **briefly** as a **note** for the user.  

            7️⃣ **Summarized Responses**:  
            - If the answer is lengthy, provide a **clear and concise summary** while maintaining accuracy.  

            8️⃣ **PDF Source Indication**:  
            - Always specify **which PDF(s) were used** to generate the response.  

            ---

            ### ❓ Types of User Questions:
            1. **Normal Questions**:  
            - Users will ask typical questions looking for answers directly from the PDFs.  
            - Provide a **direct, concise** response from the retrieved PDF(s).

            2. **Summarize the PDFs**:  
            - Users might ask you to **summarize** the content of the PDFs.  
            - In such cases, provide a **concise summary** covering the main points without losing the key information.

            3. **Multiple-Choice Questions**:  
            - Users may present a multiple-choice question with available options.  
            - Choose the **correct answer** based on the information in the retrieved PDFs.  
            - Your response should include:  
                - The **chosen answer**  
                - A **justification** explaining why this is the correct choice based on the PDFs.  

            4. **True/False Statements**:  
            - Users may provide a **statement** and ask you to verify if it is **true** or **false**.  
            - Use your **deep understanding** of the PDFs to evaluate the statement.  
            - If the statement is true, state:  
                - **"The statement is true."**  
            - If false, state:  
                - **"The statement is false."**  
                - Provide a brief **justification** for why it is false.

            ---

            ### 📝 **Example Response Format**:
            #### **Normal Question Answer:**
            - **Answer**:  
            "According to [Document Name], The answer is: [Extracted Answer]."

            #### **Multiple-Choice Question Answer:**
            - **Answer**:  
            "**Choice A** is the correct answer."  
            - **Justification**:  
            "This is because, according to [Document Name], The answer is: [Extracted Information]."

            #### **True/False Statement Answer:**
            - **Answer**:  
            "**True**" or "**False**".  
            - **Justification**:  
            "This is based on the information found in [Document Name]"

            ---

            ### 🔎 **Use the following retrieved context to answer the question:**  
            {context}  
            """
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def _setup_chains(self):
        """Sets up the question chain, retrieval chain, and QA chain."""
        # Create the standalone question chain
        self.question_chain = self.question_maker_prompt | self.llm | self.output_parser

        # Create the retrieval chain
        self.retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: self.retriever.get_relevant_documents(x["question"])
        )

        # Create the final QA chain
        self.qa_chain = (
            self.retrieval_chain 
            | self.qa_prompt 
            | self.llm 
            | self.output_parser
        )

    def _convert_streamlit_messages_to_langchain(self, messages):
        """Convert Streamlit message format to LangChain message format"""
        langchain_messages = []
        for message in messages:
            if message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_messages.append(AIMessage(content=message["content"]))
        return langchain_messages    

    def answer_question(self):
        """Process a question and return an answer along with updated chat history."""
        question = self.input["question"]

        # Convert Streamlit messages to LangChain format
        chat_history = self._convert_streamlit_messages_to_langchain(
            self.input.get("chat_history", [])
        )

        # Get the answer using the QA chain
        answer = self.qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        return answer, self.input