# Chiron Learning Agent - LangGraph Framework

A sophisticated AI-powered personalized learning tutor built with LangGraph that guides learners through structured checkpoints, verifies understanding, and provides Feynman-style explanations when needed.

## 📚 Overview

Chiron is an adaptive learning system that provides 1:1 tutoring experiences through AI. It combines:
- **Sequential Checkpoint System**: Structured learning milestones with clear success criteria
- **Web Search Integration**: Dynamically retrieves relevant learning materials using Tavily
- **Semantic Context Processing**: Intelligent chunking and embedding-based retrieval
- **Understanding Verification**: Evaluates learner comprehension with a 70% threshold
- **Feynman Teaching Method**: Simplifies complex concepts when understanding is insufficient

## 🎯 Key Features

- ✅ **24/7 Personalized Tutoring**: Individualized attention and feedback
- ✅ **Own Notes + Web Content**: Uses student-provided materials or retrieves relevant content
- ✅ **Adaptive Learning Path**: Adjusts based on understanding level
- ✅ **Human-in-the-Loop**: Interactive editing of checkpoints and answer submission
- ✅ **Structured Learning**: Clear progression from foundation to mastery
- ✅ **Semantic Chunking**: Preserves context meaning better than fixed-size splits

## 🏗️ Architecture

### Technology Stack

- **LLM**: Google Gemini Flash 2.5 (`gemini-2.5-flash`)
- **Embeddings**: Google Generative AI (`models/text-embedding-004`)
- **Chunking**: HuggingFace Encoder (`sentence-transformers/all-MiniLM-L6-v2`)
- **Framework**: LangGraph for stateful agent orchestration
- **Search**: Tavily API for web content retrieval
- **State Management**: LangGraph MemorySaver for persistence

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   LangGraph State Graph                  │
├─────────────────────────────────────────────────────────┤
│  State: LearningtState (TypedDict)                      │
│    ├─ topic, goals, context                             │
│    ├─ checkpoints, verifications, teachings             │
│    └─ current_checkpoint, current_question, answer     │
├─────────────────────────────────────────────────────────┤
│  Nodes (Functions):                                      │
│    ├─ generate_checkpoints                              │
│    ├─ chunk_context / generate_query / search_web      │
│    ├─ context_validation                                │
│    ├─ generate_question                                 │
│    ├─ verify_answer                                     │
│    └─ teach_concept                                     │
├─────────────────────────────────────────────────────────┤
│  Routing Functions:                                      │
│    ├─ route_context      (context vs search)            │
│    ├─ route_verification (progress vs teach)            │
│    ├─ route_search       (search vs question)           │
│    └─ route_teaching     (next vs end)                  │
├─────────────────────────────────────────────────────────┤
│  Supporting Systems:                                     │
│    ├─ ContextStore (embedding storage)                  │
│    ├─ MemorySaver (state persistence)                   │
│    └─ Semantic Chunking (HuggingFace)                   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- API Keys:
  - Google API Key (for Gemini Flash and embeddings)
  - Tavily API Key (for web search)

### Installation Steps

1. **Install Dependencies**

```bash
pip install langchain-community langchain-google-genai langgraph pydantic python-dotenv semantic-chunkers semantic-router tavily-python ipywidgets torch transformers sentence-transformers
```

Or run Cell 2 in the notebook which contains the pip install command.

2. **Set Up Environment Variables**

Create a `.env` file in the same directory:

```env
GOOGLE_API_KEY=your-google-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

Get your keys:
- **Google API Key**: [Google AI Studio](https://aistudio.google.com/apikey)
- **Tavily API Key**: [Tavily API](https://tavily.com)

3. **Run the Notebook**

Open `chiron_learning_agent_langraph.ipynb` in Jupyter and run cells sequentially.

## 🔄 Understanding the Flow

### High-Level Learning Cycle

```
1. Define Topic & Goals
   ↓
2. Generate Learning Checkpoints (3 milestones)
   ↓ [INTERRUPT - Review/Edit Checkpoints]
3. Process Context (chunking or web search)
   ↓
4. Validate Context Coverage
   ↓
5. Generate Question for Current Checkpoint
   ↓ [INTERRUPT - User Provides Answer]
6. Verify Answer (70% threshold)
   ↓
7a. Understanding ≥ 70% → Next Checkpoint
7b. Understanding < 70% → Feynman Teaching → Retry
   ↓
8. Repeat for all checkpoints
   ↓
9. Complete!
```

### Detailed Execution Flow

```
START
  │
  ├─→ generate_checkpoints
  │    ├─ Input: topic, goals
  │    ├─ Output: 3 checkpoints (foundation → mastery)
  │    └─ [INTERRUPT_AFTER] - Review checkpoints
  │
  ├─→ route_context
  │    ├─ Has context? → chunk_context
  │    │   ├─ Semantic chunking (128-512 tokens)
  │    │   ├─ Generate embeddings
  │    │   └─ Store in ContextStore
  │    │
  │    └─ No context? → generate_query
  │        ├─ Generate search queries from checkpoints
  │        └─→ search_web
  │            ├─ Retrieve content via Tavily
  │            ├─ Generate embeddings
  │            └─ Store in ContextStore
  │
  ├─→ context_validation
  │    ├─ For each checkpoint:
  │    │   ├─ Embed checkpoint verification
  │    │   ├─ Find top 3 similar chunks
  │    │   └─ LLM: Can criteria be answered?
  │    └─ If insufficient → generate new queries
  │
  ├─→ route_search
  │    ├─ Need search? → search_web (loop back)
  │    └─ Ready? → generate_question
  │
  ├─→ generate_question
  │    ├─ Based on checkpoint description & criteria
  │    └─ [INTERRUPT_BEFORE] - Wait for user
  │
  ├─→ user_answer (manual input)
  │    └─ Update state with answer
  │
  ├─→ verify_answer
  │    ├─ Retrieve top 3 relevant chunks
  │    ├─ LLM assesses against criteria
  │    └─ Returns: understanding_level, feedback, suggestions
  │
  ├─→ route_verification
  │    ├─ understanding_level < 0.7? → teach_concept
  │    │   ├─ Feynman-style explanation
  │    │   └─→ route_teaching → next_checkpoint or END
  │    │
  │    ├─ More checkpoints? → next_checkpoint
  │    │   └─→ generate_question (loop)
  │    │
  │    └─ All done? → END
```

## 📊 State Management

### LearningtState Structure

```python
class LearningtState(TypedDict):
    # Inputs
    topic: str                    # Learning topic
    goals: List[Goals]           # Learning objectives
    context: str                 # Optional: initial content
    
    # Processing
    context_chunks: Annotated[list, operator.add]  # Accumulated chunks
    context_key: str             # Key for embedding storage
    search_queries: SearchQuery  # Generated queries
    
    # Learning Structure
    checkpoints: Checkpoints     # Generated milestones
    current_checkpoint: int      # Progress index (0, 1, 2)
    
    # Interaction
    current_question: QuestionOutput  # Current question
    current_answer: str          # Student's answer
    
    # Assessment
    verifications: LearningVerification  # Answer evaluation
    teachings: FeynmanTeaching   # Teaching explanations
```

### State Accumulation Pattern

The `context_chunks` field uses `Annotated[list, operator.add]` to automatically accumulate chunks from multiple nodes:

```python
# Node A returns: {"context_chunks": [chunk1, chunk2]}
# Node B returns: {"context_chunks": [chunk3]}
# Final state: context_chunks = [chunk1, chunk2, chunk3]  # Merged!
```

## 🔀 Routing Logic

### Conditional Edges

The graph uses routing functions to make dynamic decisions:

#### 1. `route_context` - Initial Path Decision
```python
def route_context(state):
    if state.get("context"):
        return 'chunk_context'  # User provided context
    return 'generate_query'     # Need to search
```

#### 2. `route_verification` - Progress Decision
```python
def route_verification(state):
    if state['verifications'].understanding_level < 0.7:
        return 'teach_concept'  # Need help
    if more_checkpoints_exist:
        return 'next_checkpoint'  # Progress
    return END  # Complete
```

#### 3. `route_search` - Validation Loop
```python
def route_search(state):
    if state['search_queries'] is None:
        return "generate_question"  # Context sufficient
    return "search_web"  # Need more context
```

## 🧠 Key Components Explained

### 1. Checkpoint Generation

**Function**: `generate_checkpoints`

- Takes topic and goals
- Uses Gemini Flash with structured output
- Generates exactly 3 checkpoints:
  1. Foundation level
  2. Application level
  3. Mastery level

**Output Structure**:
```python
Checkpoints(
    checkpoints=[
        LearningCheckpoint(
            description="Understand basic concepts...",
            criteria=["Define X", "Identify Y", "Explain Z"],
            verification="Explain X to a peer..."
        ),
        # ... 2 more checkpoints
    ]
)
```

### 2. Semantic Chunking

**Function**: `chunk_context`

- Uses HuggingFace encoder (`all-MiniLM-L6-v2`) for semantic boundary detection
- Chunk size: 128-512 tokens
- Preserves semantic meaning (not just fixed-size splits)
- Generates embeddings for each chunk using Google embeddings
- Stores in `ContextStore` for efficient retrieval

### 3. Context Validation

**Function**: `context_validation`

**Process**:
1. For each checkpoint:
   - Embed the checkpoint's verification method as a query
   - Use cosine similarity to find top 3 relevant chunks
   - Ask LLM: "Can this checkpoint be answered with these chunks?"
2. If any checkpoint fails validation:
   - Generate new search queries
   - Trigger additional web search

**Why This Matters**: Ensures context quality before asking questions.

### 4. Question Generation

**Function**: `generate_question`

- Reads current checkpoint from state
- Uses checkpoint description, criteria, and verification method
- LLM generates an appropriate question aligned with checkpoint goals

### 5. Answer Verification

**Function**: `verify_answer`

**Process**:
1. Retrieve top 3 relevant chunks using embeddings
2. LLM evaluates answer against:
   - Checkpoint description
   - Success criteria
   - Verification method
   - Relevant context chunks
3. Returns structured assessment:
   - `understanding_level`: 0.0 to 1.0 (70% = threshold)
   - `feedback`: Detailed explanation
   - `suggestions`: Improvement recommendations
   - `context_alignment`: Whether answer aligns with context

### 6. Feynman Teaching

**Function**: `teach_concept`

Triggered when `understanding_level < 0.7`

**Output**:
- `simplified_explanation`: Jargon-free explanation
- `key_concepts`: Essential points list
- `analogies`: Concrete comparisons

**Feynman Technique**: Explain concepts as if teaching a child - use simple language and analogies.

## 💾 Memory & Persistence

### MemorySaver Checkpointer

```python
memory = MemorySaver()
graph = searcher.compile(checkpointer=memory)
```

**What Gets Saved**:
- Complete state after each node execution
- Node execution history
- Allows resuming from any point

### ContextStore

```python
class ContextStore:
    def save_context(chunks, embeddings, key)
    def get_context(context_key)
```

**Purpose**: 
- Stores text chunks and their embeddings
- Enables efficient similarity search
- Avoids re-embedding same content

## 🔌 Human-in-the-Loop Pattern

### Interrupt Points

```python
graph = searcher.compile(
    interrupt_after=["generate_checkpoints"],  # Pause after checkpoints
    interrupt_before=["user_answer"],         # Pause before answer input
    checkpointer=memory
)
```

### Usage Pattern

```python
# 1. Initial run - pauses after checkpoints
thread = {"configurable": {"thread_id": "20"}}
for event in graph.stream(initial_input, thread):
    print_checkpoints(event)  # Review checkpoints

# 2. Edit checkpoints (optional)
updated_model = editor.get_model()
graph.update_state(thread, {"checkpoints": updated_model}, 
                   as_node="generate_checkpoints")

# 3. Resume - pauses before user_answer
for event in graph.stream(None, thread):
    print(event['current_question'])

# 4. Provide answer
answer = input("Answer: ")
graph.update_state(thread, {"current_answer": answer}, 
                   as_node="user_answer")

# 5. Resume to verify
for event in graph.stream(None, thread):
    print_verification_results(event)
```

## 📖 Usage Example

### Basic Usage

```python
# 1. Define learning topic and goals
initial_input = {
    "topic": "Anemia",
    "goals": [Goals(goals="I am a medical student, I want to master the diagnosis of Anemia")],
    "context": note,  # Optional: your own notes
    "current_checkpoint": 0
}

# 2. Run graph (first interrupt after checkpoints)
thread = {"configurable": {"thread_id": "20"}}
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print_checkpoints(event)

# 3. Review/Edit checkpoints (optional)
# Use the interactive widget or manually update

# 4. Continue execution
for event in graph.stream(None, thread, stream_mode="values"):
    current_question = event.get('current_question', '')
    if current_question:
        print(current_question)

# 5. Provide answer
answer_question = input("Answer the question above: ")

# 6. Update state and continue
graph.update_state(thread, {"current_answer": answer_question}, 
                   as_node="user_answer")

# 7. Get verification results
for event in graph.stream(None, thread, stream_mode="values"):
    print_verification_results(event)
    print_teaching_results(event)
```

## 🔧 Configuration

### Model Configuration

In the setup cell (Cell 6), you can adjust:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # LLM model
    temperature=0,              # Lower = more deterministic
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # Embedding model
    google_api_key=os.getenv('GOOGLE_API_KEY')
)
```

### Understanding Threshold

In `route_verification` function, adjust the threshold:

```python
if state['verifications'].understanding_level < 0.7:  # Change 0.7 to your preference
    return 'teach_concept'
```

### Chunking Parameters

In `chunk_context` function:

```python
chunker = StatisticalChunker(
    encoder=encoder,
    min_split_tokens=128,  # Minimum chunk size
    max_split_tokens=512   # Maximum chunk size
)
```

## 🎨 Interactive Features

### Checkpoint Editor Widget

The notebook includes an interactive widget (ipywidgets) for editing checkpoints:

- Edit descriptions
- Modify success criteria
- Update verification methods
- Add/remove criteria
- Accept/reject checkpoints

### Pretty Printing Functions

- `print_checkpoints()`: Formatted checkpoint display
- `print_verification_results()`: Visual understanding level with bar chart
- `print_teaching_results()`: Formatted Feynman teaching output

## 🔍 Embedding & Retrieval System

### How It Works

1. **Embedding Generation**:
   - Text chunks are embedded using Google Generative AI embeddings
   - Embeddings are vectors in high-dimensional space (typically 768 dimensions)

2. **Similarity Search**:
   ```python
   query_embedding = embeddings.embed_query("verification method")
   similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
   top_indices = sorted(range(len(similarities)), 
                       key=lambda i: similarities[i], 
                       reverse=True)[:3]
   ```

3. **Cosine Similarity**:
   - Measures angle between vectors (not distance)
   - Range: -1 to 1 (typically 0 to 1 for normalized embeddings)
   - Higher = more similar content

### Why This Matters

- **Efficiency**: Only retrieve relevant chunks (top 3) instead of all context
- **Accuracy**: Semantic similarity finds contextually relevant content
- **Token Savings**: Reduce token usage by focusing on relevant information

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError for langchain_community**
   - Solution: Run Cell 2 (pip install) in the notebook

2. **ImportError for torch/transformers**
   - Solution: Install with `!pip install torch transformers sentence-transformers`

3. **ValueError: contents are required**
   - Solution: Ensure GOOGLE_API_KEY is set in your .env file

4. **Graph execution pauses indefinitely**
   - Solution: Check for interrupt points - you need to provide input or resume manually

### Debug Tips

- Check state at any point: `graph.get_state(thread).values`
- Inspect next nodes: `graph.get_state(thread).next`
- View graph structure: `graph.get_graph().draw_mermaid_png()`

## 📝 Code Structure

```
chiron_learning_agent_langraph.ipynb
├── Cell 0-2: Introduction & Requirements
├── Cell 3-4: Imports
├── Cell 5-6: Setup (LLM, embeddings, API keys)
├── Cell 7-8: Data Models (Pydantic)
├── Cell 9-10: State Definition
├── Cell 11-12: Helper Functions
├── Cell 13-14: Prompt Configuration
├── Cell 15-16: Context Storage
├── Cell 17-18: Core Functions
├── Cell 19-20: State Management Functions
├── Cell 21-22: Routing Functions
├── Cell 23-24: Graph Construction
└── Cell 25+: Examples & Interactive Widgets
```

## 🚀 Extending the Framework

### Adding New Nodes

```python
def my_custom_node(state: LearningtState):
    # Read from state
    data = state['some_field']
    
    # Process
    result = process(data)
    
    # Return state update
    return {"some_field": result}

# Add to graph
searcher.add_node("my_custom_node", my_custom_node)
searcher.add_edge("previous_node", "my_custom_node")
```

### Adding New Routing Logic

```python
def my_route_function(state: LearningtState):
    if some_condition(state):
        return "node_a"
    return "node_b"

searcher.add_conditional_edges(
    "source_node",
    my_route_function,
    {"node_a": "node_a", "node_b": "node_b"}
)
```

### Custom Embeddings

Replace Google embeddings with another provider:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## 📚 Key Concepts

### LangGraph Fundamentals

1. **State Graph**: Maintains shared state across all nodes
2. **Nodes**: Functions that take state, return state updates
3. **Edges**: Define flow between nodes
4. **Conditional Edges**: Dynamic routing based on state
5. **Checkpointer**: Persists state for resumability
6. **Interrupts**: Pause execution for human input

### Learning System Design

1. **Structured Progression**: Checkpoints ensure logical learning flow
2. **Adaptive Feedback**: Adjusts based on understanding level
3. **Context-Aware**: Uses embeddings to find relevant information
4. **Validation Loop**: Ensures quality before proceeding
5. **Feynman Technique**: Simplifies complex concepts when needed

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Add your contact information here]

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Google Gemini](https://deepmind.google/technologies/gemini/) models
- Semantic chunking via [semantic-chunkers](https://github.com/rlancemartin/semantic-chunkers)
- Web search via [Tavily](https://tavily.com/)

---

**Note**: This is an educational AI tutoring system. For actual medical or professional advice, consult qualified professionals.

