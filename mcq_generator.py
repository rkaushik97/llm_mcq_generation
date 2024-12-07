import os
from dotenv import load_dotenv
import json
import logging
import random
from datetime import datetime
from openai import OpenAI


logging.basicConfig(
    filename="mcq_generator.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load env variables from the .env file
load_dotenv()
logging.info("Loaded API key and base URL from environment.")

deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
logging.debug(f"API Key: {deepinfra_api_key[:4]}... (truncated)")

deepinfra_base_url = os.getenv("DEEPINFRA_BASE_URL")
logging.debug(f"Base URL: {deepinfra_base_url}")

if not deepinfra_api_key:
    raise ValueError("API key not found. Please set DEEPINFRA_API_KEY in the .env file.")
if not deepinfra_base_url:
    raise ValueError("Base URL not found. Please set DEEPINFRA_BASE_URL in the .env file.")

openai = OpenAI(api_key=deepinfra_api_key,
                base_url=deepinfra_base_url)

# 1. Base prompt template (Simple MCQ generation)
BASE_PROMPT_TEMPLATE = """
Using the text below, create a multiple-choice question (MCQ) with four answer options (a, b, c, d) and provide the correct answer. Indicate the correct answer explicitly.

Text:
{text}

MCQ Question with Answer: 
"""

# 2. Chain of Thought Prompt Template (with reasoning and answer)
CHAIN_OF_THOUGHT_PROMPT_TEMPLATE = """
Using the text below, answer the following question step by step with reasoning. Base your reasoning only on the given text.

Text:
{text}

Question:
{base_mcq_question}

Answer with reasoning:
"""

# 3. Self Consistency in Chain of Thought Prompt Template (with multiple rounds and answer consistency)
SELF_CONSISTENCY_PROMPT_TEMPLATE = """
Using the text below, select the most probable answer to the question. Your answer should be based on reasoning within the context of the text. Repeat the process multiple times to ensure consistency.

Text:
{text}

Question:
{base_mcq_question}

Answer with reasoning:
"""

NONE_OF_THE_ABOVE_PROMPT_TEMPLATE = """
Using the text below, create an MCQ question where the correct answer is replaced by "None of the above." Generate plausible incorrect options.

Text:
{text}

Base Question:
{base_mcq_question}

MCQ with 'None of the above':
"""

TRUE_FALSE_PROMPT_TEMPLATE = """
Using the text below, create True-or-False questions based on the options from the following MCQ.

Text:
{text}

MCQ:
{base_mcq_question}

True-or-False Questions:
"""

# MCQ generator class
class MCQGenerator:
    def __init__(self, model="meta-llama/Meta-Llama-3.1-405B-Instruct", max_tokens=1500):
        self.model = model
        self.max_token = max_tokens
        logging.info(f"Initialized MCQGenerator with model: {model}, max_tokens: {max_tokens}")

    def generate_mcq_with_answer(self, text, prompt_template):
        prompt = prompt_template.format(text=text)
        logging.info("Generating Base MCQ with answer.")
        return self._query_openai(prompt, max_tokens=500)
    
    def generate_mcq_variants(self, text, base_mcq, selected_variants=None):
        if not selected_variants:
            selected_variants = ["Re-ordered Options", "Varying Options", "None of the Above", "True or False"]

        logging.info(f"Generating MCQ variants: {selected_variants}")
        variants = {}
        if "Re-ordered Options" in selected_variants:
            variants["Re-ordered Options"] = self.reorder_options(base_mcq)
        if "Varying Options" in selected_variants:
            variants["Varying Options"] = self.generate_mcq_with_varying_options(base_mcq)
        if "None of the Above" in selected_variants:
            variants["None of the Above"] = self.replace_with_none_of_the_above(text, base_mcq)
        if "True or False" in selected_variants:
            variants["True or False"] = self.generate_true_false_questions(text, base_mcq)
        return variants
    
    def reorder_options(self, base_mcq):
        lines = base_mcq.split("\n")
        options = [line for line in lines if line.startswith(("a)", "b)", "c)", "d)"))]
        random.shuffle(options)
        logging.info("Reordered options for the MCQ.")
        return "\n".join(options)

    def generate_mcq_with_varying_options(self, base_mcq, num_options=[2, 3, 6]):
        varying_options = []
        for n in num_options:
            prompt = f"""
            Transform the following MCQ into one with {n} answer options. Ensure options are plausible and one is correct.

            Original MCQ:
            {base_mcq}

            MCQ with {n} options:
            """
            response = self._query_openai(prompt)
            varying_options.append(response)
            logging.info(f"Generated MCQ with {n} options.")
        return varying_options

    def replace_with_none_of_the_above(self, text, base_mcq):
        prompt = NONE_OF_THE_ABOVE_PROMPT_TEMPLATE.format(text=text, base_mcq_question=base_mcq)
        logging.info("Replaced correct answer with 'None of the above'.")
        return self._query_openai(prompt)

    def generate_true_false_questions(self, text, base_mcq):
        prompt = TRUE_FALSE_PROMPT_TEMPLATE.format(text=text, base_mcq_question=base_mcq)
        logging.info("Generated True-or-False questions from MCQ options.")
        return self._query_openai(prompt)

    
    def evaluate_with_cot(self, text, question):
        prompt = CHAIN_OF_THOUGHT_PROMPT_TEMPLATE.format(text=text, base_mcq_question=question)
        logging.info("Evaluating question using Chain of Thought reasoning.")
        return self._query_openai(prompt, max_tokens=1500)

    def evaluate_with_self_consistency(self, text, question, n_samples=5):
        # Use Self Consistency template on the given question
        results = []
        logging.info("Evaluating question using Self Consistency.")
        for _ in range(n_samples):
            prompt = SELF_CONSISTENCY_PROMPT_TEMPLATE.format(text=text, base_mcq_question=question)
            results.append(self._query_openai(prompt, max_tokens=1000))    
        return results
    
    def _query_openai(self, prompt, max_tokens=1500):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            logging.debug(f"Prompt Tokens: {response.usage.prompt_tokens}, Completion Tokens: {response.usage.completion_tokens}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during OpenAI query: {e}")
            return None
        
def save_results_to_file(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {filename}")

def run_experiment(text, max_tokens=1500, selected_variants=None):
    mcq_generator = MCQGenerator(max_tokens=max_tokens)

    base_mcq_with_answer = mcq_generator.generate_mcq_with_answer(text, BASE_PROMPT_TEMPLATE)
    base_mcq_question = extract_question(base_mcq_with_answer)
    mcqa_variants = mcq_generator.generate_mcq_variants(text, base_mcq_question, selected_variants)

    evaluations = {
        "Base MCQ": {
            "CoT": mcq_generator.evaluate_with_cot(text, base_mcq_question),
            "Self Consistency": mcq_generator.evaluate_with_self_consistency(text, base_mcq_question)
        }
    }
    for name, variant in mcqa_variants.items():
        evaluations[name] = {
            "CoT": mcq_generator.evaluate_with_cot(text, variant),
            "Self Consistency": mcq_generator.evaluate_with_self_consistency(text, variant)
        }

    results = {
        "Base MCQ with Answer": base_mcq_with_answer,
        "Variants": mcqa_variants,
        "Evaluations": evaluations
    }

    filename = f"mcq_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results_to_file(results, filename)
    return results

def extract_question(mcq_with_answer):
    """
    Extracts the question and options from the MCQ output, removing the 'Correct Answer' line.
    """
    lines = mcq_with_answer.split("\n")
    cleaned_lines = [line for line in lines if not line.lower().startswith("**correct answer:**")]
    return "\n".join(cleaned_lines).strip()


if __name__ == "__main__":
    sample_text = """
1.1 Purpose
Bolivian cities’ exposure to natural hazards – mainly floods, flash floods and landslides – combined with rapid
unplanned urbanization, is a huge issue: Bolivia has the second highest economic risk exposure to multiple
hazards worldwide. To improve resilience of its cities, the government has identified key challenges needed to be
addressed.
Accordingly, the government is improving the policy frame and scaling up investment approaches and tools to
increase resilience. The project Resilient Bolivian Cities will support the two municipalities of Santa Cruz and La
Paz and the central Government in this quest. This project is financed by SECO to the amount of 4.9 million Swiss
francs and is part of a larger World Bank project with funding of US$ 75 million.
The present Instructions to Bidders constitute the core of the tender for the mandate of project design and
management plan for the Kantuntani Ecological Park, located in the city of La Paz, Bolivia. The proposed
assignment is a direct support to the Municipal Autonomous Government of La Paz (GAMLP) and will be provided
in the framework of the “Swiss Accompanying Measures (SAM)” - mechanism established by the Swiss State
Secretariat for Economic Affairs SECO.
In summary, the mandate entails the conception of the Kantuntani Ecological Park (PEK) project and will be
responsible for providing the Government of the Municipality of La Paz with (i) a diagnosis of the entire PEK area,
including all the complementary base studies necessary for its implementation; (ii) the programm, the preliminary
project and the PEK detailed project, including the Pre-investment Technical Design Study for its subsequent
construction; (iii) the development of the management and maintenance plan for the PEK. Details about the
mandate of the SAM and the required expertise and experience are contained in the Terms of Reference (ToR).
The Contracting Entity for this mandate is the Consortium Urbaplan-ENCO (SAM Coordinator), on behalf of the
Swiss State Secretariat for Economic Affairs (SECO).
1.2 Eligible organizations/persons
The award procedure chosen is an open international tender.
The tendering process is open to institutions and companies able to implement assignments in the field of
landscape architecture in Bolivia.
The establishment of a consortium between institutions, companies and sub-contracting of independent
consultants is possible. In the case of a consortium, the lead company and/or the team leader need to be clearly
defined.
"""
results = run_experiment(sample_text, max_tokens=1200, selected_variants=["Re-ordered Options", "None of the Above"])

# Print Results
"""
for key, value in results.items():
    if isinstance(value, list):
        print(f"{key}:\n" + "\n".join(value))
    else:
        print(f"{key}:\n{value}")
"""