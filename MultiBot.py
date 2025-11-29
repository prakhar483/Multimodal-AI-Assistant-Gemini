import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import json

# --- Page config ---
st.set_page_config(
    page_title="🤖 Multimodal AI Chatbot",
    page_icon="🧠",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Multimodal Chatbot (Gemini Model)")
    st.markdown("""
📝 **Problem Statement**  
Build an AI chatbot capable of answering across 7 knowledge domains using **Gemini**, **LangChain**, and **Streamlit**.

---

💬 **Features:**  
- Accepts **natural language queries**  
- Intelligent, context-aware responses  
- Covers **7 key domains**  
- Maintains **conversational history**  
- Works in **real time**

---

📚 **Domains Supported:**  
- 📊 Data Science / AI  
- ✈️ Travel  
- 🎬 Movies  
- 🐍 Python  
- 💼 Career Guidance  
- 🌍 General Knowledge  
- 🧘‍♀️ Health & Wellness  

---

❤️ Built by **PRAKHAR DWIVEDI**  
🔗 Powered by Gemini + LangChain + Streamlit  
""")

# --- Header ---
st.markdown("""
<div style="text-align:center; margin-bottom: 20px;">
    <h1>🤖 Multimodal Knowledge Assistant</h1>
    <h4>💡 Ask me about Data Science, Travel, Movies, Python, Careers & More!</h4>
    <hr style="margin-top: 10px; margin-bottom: 20px;">
</div>
""", unsafe_allow_html=True)

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your Gemini API Key
)

domain_instructions = """
You are a Multi-Domain Knowledge Assistant.
You can respond about the following domains:
1. Data Science / ML / AI
2. Travel (trip planning, cost estimation, travel advice)
3. Movies / Entertainment (recommendations, genres, summaries)
4. Python / Programming Help
5. Career / Interview Prep
6. General Knowledge / Facts
7. Health & Lifestyle Advice

Instructions:
- Identify the domain based on user input.
- Respond concisely and clearly.
- Keep answers structured if applicable.
- Use examples, code, or lists where helpful.
- Maintain multi-turn conversation context.
- Provide JSON structured output if the domain involves lists (like movies or travel recommendations).
"""

chat_template = ChatPromptTemplate.from_messages([
    ("system", domain_instructions),
    ("human", "{user_input}\n\nPrevious conversation:\n{history}")
])

chain = LLMChain(llm=llm, prompt=chat_template)

# --- Session State for memory ---
if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = {"history": []}

# --- User Input UI ---
user_input = st.text_area(
    "💬 Ask your question:",
    height=100,
    placeholder="E.g., 'Suggest 5 Bollywood thrillers' or 'Travel tips for Manali in October'"
)

submit = st.button("🚀 Ask AI")

# --- On submit ---
if submit:
    if not user_input.strip():
        st.warning("⚠️ Please enter your question!")
    else:
        with st.spinner("🤖 Thinking..."):
            # Build chat history
            chat_history = "\n".join([
                f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
                for m in st.session_state.memory_buffer["history"]
            ])

            # Get LLM response
            response = chain.invoke({
                "user_input": user_input,
                "history": chat_history
            })

            raw_text = response.get("text", "")
            if "```" in raw_text:
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            # Save to memory
            st.session_state.memory_buffer["history"].append(HumanMessage(content=user_input))
            st.session_state.memory_buffer["history"].append(AIMessage(content=raw_text))

            # Try to parse JSON
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed = None

            # --- Response Output ---
            st.markdown("### 🤖 AI Response")

            if isinstance(parsed, dict) and "recommendations" in parsed:
                domain = parsed.get("domain", "").lower()
                recs = parsed["recommendations"]

                if "movie" in domain:
                    st.markdown("### 🎬 Movie Recommendations")
                    for idx, m in enumerate(recs, 1):
                        st.markdown(f"""
**{idx}. 🎥 {m.get("title")} ({m.get("year")})**  
👨‍🎬 Director: *{m.get("director", "Unknown")}*  
📝 Summary: {m.get("plot_summary", "N/A")}  
---
""")
                elif "travel" in domain:
                    st.markdown("### ✈️ Travel Suggestions")
                    for section in recs:
                        for k, v in section.items():
                            st.markdown(f"🔹 **{k}**: {v}")
                else:
                    st.json(parsed)

            elif isinstance(parsed, list):
                for idx, item in enumerate(parsed, 1):
                    title = item.get("title") or item.get("name", f"Item {idx}")
                    st.markdown(f"**{idx}. {title}**")
                    for key, value in item.items():
                        if key != "title" and key != "name":
                            st.markdown(f" - 🔹 {key.capitalize()}: {value}")
                    st.markdown("---")
            elif parsed:
                st.json(parsed)
            else:
                st.markdown(raw_text)

# --- Conversation History ---
if st.session_state.memory_buffer["history"]:
    st.markdown("---")
    st.markdown("### 🧵 Conversation History")
    for i, msg in enumerate(st.session_state.memory_buffer["history"]):
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🧑 You:** {msg.content}")
        else:
            with st.expander(f"🤖 AI Response #{(i+1)//2}"):
                st.markdown(msg.content)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:14px;">
    Made with ❤️ by <b>PRAKHAR DWIVEDI</b> | ⚙️ Powered by <b>Gemini</b> + <b>LangChain</b> + <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)

