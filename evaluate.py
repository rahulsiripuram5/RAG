from ask_rag import answer_question # We'll import your main function
from collections import Counter
import re

# --- 1. YOUR EVALUATION DATASET ---
# (Create 10-15 of these based on your data)
eval_dataset = [
    {
        "question": "How can extensions list all keys in the VS Code secret store?",
        "ground_truth_answer": "By using a context.secret.keys() function, extensions can return all keys in the secret store."
    },
    {
        "question": "What issue was reported with autoDetectColorScheme in VS Code?",
        "ground_truth_answer": "autoDetectColorScheme was suddenly not working as expected in a recent VS Code version."
    },
    {
        "question": "How can extensions contribute new modes, prompts, or instructions to Copilot?",
        "ground_truth_answer": "Extensions can contribute modes, prompts, and instructions, such as a 'Testing' mode, to Copilot, allowing for custom chat modes and tool recommendations."
    },
    {
        "question": "What was the request regarding comments or newlines in .prompt/.chatmode files?",
        "ground_truth_answer": "The request was to support comments and newlines in .prompt/.chatmode files, and the parser was updated to support comments in the front matter header."
    },
    {
        "question": "What git issue appeared after updating VS Code from 1.89.1 to 1.90.0?",
        "ground_truth_answer": "Files appeared as deleted or moved in git after updating VS Code, even though no manual file operations were performed."
    },
    {
        "question": "What feature was requested for terminal profiles in agent mode?",
        "ground_truth_answer": "A feature was requested to allow configuring a specific terminal profile for agent mode, such as 'github.copilot.chat.agent.terminalProfile'."
    },
    {
        "question": "What bug was reported regarding snippet workflows and Copilot completions?",
        "ground_truth_answer": "Copilot completions were interfering with snippet workflows, disrespecting settings and keybindings that should prevent unwanted acceptance."
    },
    {
        "question": "What improvement was suggested for the 'Keep Changes' button during agent loops?",
        "ground_truth_answer": "It was suggested to allow users to 'Keep Changes' at any time during an agent loop, not just after it finishes."
    },
    {
        "question": "What new feature was proposed for Git conflict resolution in Copilot?",
        "ground_truth_answer": "A feature was proposed to allow users to delegate resolving merge conflicts to Copilot."
    },
    {
        "question": "What was the concern about settings auto-refetch compatibility in experimental configuration?",
        "ground_truth_answer": "The concern was about ensuring settings with experimental configuration values handle automatic changes when experiment values are updated, especially with mode: 'startup' or 'auto'."
    },
    {
        "question": "What usability improvement was requested for copying redirect URIs in dialogues?",
        "ground_truth_answer": "It was requested to provide affordance, such as linkifying, to easily copy redirect URIs from dialogues."
    },
    {
        "question": "What feature was requested for agent mode regarding skipping tool calls?",
        "ground_truth_answer": "A 'Skip and Continue' button was requested in agent mode to allow skipping a tool call and continuing the agent flow."
    },
    {
        "question": "What notification feature was requested for chat response completion?",
        "ground_truth_answer": "A setting was requested to show OS notifications when agent/edit mode execution finishes, such as 'chat.notifyWindowOnCompletion'."
    },
    {
        "question": "What was the issue with TODOs in agent mode?",
        "ground_truth_answer": "TODOs were not being done in order, and some remained in progress or were skipped, raising concerns about order and token consumption."
    },
    {
        "question": "What was the request regarding safe usage of 'git grep' in VS Code?",
        "ground_truth_answer": "A request was made to allow safe usage of 'git grep' in VS Code."
    }
]

# --- 2. EVALUATION METRIC FUNCTIONS ---
def normalize_text(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s) # Remove punctuation
    return s

def calculate_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_em(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

# --- 3. RUN THE EVALUATION ---
total_f1 = 0
total_em = 0

print("Starting evaluation...")

for item in eval_dataset:
    question = item["question"]
    ground_truth = item["ground_truth_answer"]
    
    # Get the answer from your RAG pipeline
    predicted_answer = answer_question(question)
    
    f1 = calculate_f1(predicted_answer, ground_truth)
    em = calculate_em(predicted_answer, ground_truth)
    
    total_f1 += f1
    total_em += em
    
    print(f"\nQ: {question}")
    print(f"A (True): {ground_truth}")
    print(f"A (Pred): {predicted_answer}")
    print(f"F1: {f1:.4f} | EM: {em}")
    print("-" * 20)

# Calculate and print average scores
avg_f1 = total_f1 / len(eval_dataset)
avg_em = total_em / len(eval_dataset)

print("\n--- EVALUATION COMPLETE ---")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"Average Exact Match: {avg_em:.4f}")
print("---------------------------")