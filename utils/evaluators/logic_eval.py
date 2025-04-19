import re
import spacy
from sentence_transformers import SentenceTransformer, util

from models.llm_agent import LLM_Agent, OpenAiModel


def extract_tag_content(response: str, tag_name: str) -> tuple[list[str], bool]:
    """Extract the content within a specific XML-style tag from a response."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    return (match.group(1).strip(), True) if match else ([], False)


def extract_all_tag_contents(response: str, tag_name: str) -> tuple[list[str], bool]:
    """Extract the content within all occurrences of a specific XML-style tag from a response."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, response, re.DOTALL)
    return ([match.strip() for match in matches], True) if matches else ([], False)


def extract_values(response: str, tag_name: str) -> tuple[list[float], bool]:
    """Extract a list of float values from a specific tag."""
    raw, found = extract_tag_content(response, tag_name)
    if not found:
        return [], False
    try:
        values = [float(v.strip()) for v in raw.split(',') if v.strip()]
        return values, True
    except ValueError:
        return [], False

   
def infer_observations_one_by_one(llm_agent: LLM_Agent, generated_response: str, print_results: bool = False):
    """
    Given the response from the LLM, infer the observation.
    
    Args:
        llm_agent (LLM_Agent): The LLM agent to use for inference.
        response (str): The response from the LLM.
        config (dict): Configuration dictionary containing system and user prompts.
    
    Returns:
        str: The inferred observation.
    """

    # Extracting the observations and the deductions from the response
    observations, _ = extract_all_tag_contents(generated_response, "observation")
    deductions, _ = extract_all_tag_contents(generated_response, "deduction")

    obs_similarities = []
    indication = ["our speed (in m/s)", "headway (in m)", "leader speed (in m/s)"]
    for i, observation in enumerate(observations):
        prompt = ""
        for j, obs in enumerate(observations):
            if i != j:
                prompt += f"<observation>{obs}</observation>\n"
        prompt += f"<masked observation> </masked observation> (indication: the observation is about '{indication[i]}'\n"
        for deduction in deductions:
            prompt += f"<deduction>{deduction}</deduction>\n"

        config = {
            'system_prompt': """
            You are an expert in finding missing observations from the deductions. This deductions are steps of a reasoning process about the current and future traffic situation.
            You are given a list of deductions and the number of missing observations and you have to guess the content of the missing observations.
            Return only each missing observation between <masked observation>. 
            """,
            'user_prompt': """
            {0}
            Guess the masked observations.
            """
        }

        # Getting the response from the LLM 
        response = llm_agent.get_response(
            config["system_prompt"], 
            config['user_prompt'].format(prompt), 
            temperature=0, 
            num_samples=1
        )

        masked_observations, _ = extract_all_tag_contents(response, "masked observation")

        obs_similarities.append(calculate_cosine_similarity(observation, masked_observations[0]))

        if print_results:
            print(f"<original observation> {observation} </original observation>")
            print(f"<masked observation> {masked_observations[0]} </masked observation>")
            print(f"{calculate_cosine_similarity(observation, masked_observations[0])}")
            print("--------------------------------------------------")

    return obs_similarities


def infer_the_observation(llm_agent: LLM_Agent, generated_response: str, print_results: bool = False):
    """
    Given the response from the LLM, infer the observation.
    
    Args:
        llm_agent (LLM_Agent): The LLM agent to use for inference.
        response (str): The response from the LLM.
        config (dict): Configuration dictionary containing system and user prompts.
    
    Returns:
        str: The inferred observation.
    """    
    ## masked all the observations and ask the model to guess them.

    # in the observation, there is number of the exact speed etc xhich is impossible to guess.
    # so it is not relevant to ask the model to guess the observation.

    # Extracting the observations and the deductions from the response
    observations, _ = extract_all_tag_contents(generated_response, "observation")
    deductions, _ = extract_all_tag_contents(generated_response, "deduction")

    nlp = spacy.load("en_core_web_sm")
    masked_obs_str = ""
    indication = ["our speed", "headway", "leader speed"]
    for i, observation in enumerate(observations):
        masked_obs_str += f"<masked observation> </masked observation> (indication: the observation is about '{indication[i]}'\n"

    deductions_str = ""
    for deduction in deductions:
        deductions_str += f"<deduction>{deduction}</deduction>\n"

    config = {
        'system_prompt': """
        You are an expert in finding missing observations from the deductions. This deductions are steps of a reasoning process about the current and future traffic situation.
        You are given a list of deductions and the number of missing observations and you have to guess the content of the missing observations.
        Return only each missing observation between <masked observation>. 
        """,
        'user_prompt': """
        {0}
        {1}
        Guess the masked observations.
        """
    }

    # Getting the response from the LLM 
    response = llm_agent.get_response(
        config["system_prompt"], 
        config['user_prompt'].format(masked_obs_str, deductions_str), 
        temperature=1, 
        num_samples=1
    )

    masked_observations, _ = extract_all_tag_contents(response, "masked observation")

    obs_similarities = []

    for mask_obs, obs in zip(masked_observations, observations):
        obs_similarities.append(calculate_cosine_similarity(obs, mask_obs))

        if print_results:
            print(f"<original observation> {obs} </original observation>")
            print(f"<masked observation> {mask_obs} </masked observation>")
            print(f"{calculate_cosine_similarity(obs, mask_obs)}")
            print("--------------------------------------------------")

    return obs_similarities


def infer_deductions(llm_agent: LLM_Agent, generated_response: str, print_results: bool = False):
    """
    For each deduction,
    we extract the subject of the sentence 
    we mask the rest of the sentence
    with the previous observations/deductions and the next deductions, we ask the model to infer the rest of the sentence.
    Then we compare the new sentence with the original one.
    """
    observations, _ = extract_all_tag_contents(generated_response, "observation")
    deductions, _ = extract_all_tag_contents(generated_response, "deduction")

    nlp = spacy.load("en_core_web_sm")
    llm_agent = OpenAiModel()

    deduc_similarities = []

    for deduction in deductions:

        subject, next_word = extract_subject_and_next_word(deduction, nlp)
        # Get response from LLM
        response = llm_agent.get_response(
            system_prompt="you are an expert in finding missing deductions from the observations and the deductions. \n return only the missing deduction between <masked deduction>.\n", 
            user_prompt=user_prompt(observations, deductions, deduction, subject + next_word), 
            temperature=1, 
            num_samples=1
        )

        deduc_similarities.append(calculate_cosine_similarity(deduction, response))

        if print_results:
            print(f"<original deduction> {deduction} </original deduction>")
            print(response)
            print(f"{calculate_cosine_similarity(deduction, response)}")
            print("--------------------------------------------------")

    return deduc_similarities


def extract_subject(sentence: str, nlp: spacy) -> str:
    """
    Extract the subject of the sentence using spaCy.
    Args:
        sentence (str): The input sentence.
        nlp (spacy.lang.en.English): The spaCy NLP model.
    Returns:
        str: The subject of the sentence.
    """
    # Process the sentence using spaCy
    doc = nlp(sentence)

    for token in doc:
        if "subj" in token.dep_:
            return token.text
    return None


def extract_subject_and_next_word(sentence: str, nlp: spacy) -> tuple[str, str]:
    """
    Extract the subject and the next word in the sentence. So we now what is the deduction about.
    Having the next word (verb or auxiliaire) is useful to know if the deduction is about the current situation or what is likely to happen in the future.
    Args:
        sentence (str): The input sentence.
        nlp (spacy.lang.en.English): The spaCy NLP model.
    Returns:
        tuple[str, str]: The subject and the next word in the sentence.
    """
    # Process the sentence using spaCy
    doc = nlp(sentence)

    for i, token in enumerate(doc):
        # Check if the token is the subject of the sentence
        if "subj" in token.dep_:
            subject = token.text
            # Get the next word if it exists
            if i + 1 < len(doc):
                next_word = doc[i + 1].text
            else:
                next_word = None
            return subject, next_word
    return None, None


def user_prompt(observations: list[str], deductions: list[str], current_deduction: str, indication: str) -> str:
    """
    Args:
        observations (list[str]): List of observations.
        deductions (list[str]): List of deductions.
        current_deduction (str): The current deduction being masked.
        indication (str): The indication for the masked deduction (the subject of the sentence (ex: speed, headway, etc.) and the next word (most of the time the tense of the verb, so deduction about the current situation or what is likely to happen in the future)).
    Returns:
        str: The user prompt for the LLM.
        example:
        <observation>...</observation>
        <observation>...</observation>
        <deduction>...</deduction>
        <masked deduction> </masked deduction> (indication: the deduction is talking about 'speed will')
        <deduction>...</deduction>
        <deduction>...</deduction>
        Guess the masked deduction.
    """
    prompt = ""
    for observation in observations:
        prompt += f"<observation>{observation}</observation>\n"
    for deduction in deductions:
        if deduction != current_deduction:
            prompt += f"<deduction>{deduction}</deduction>\n"
        else:
            prompt += f"<masked deduction> </masked deduction> (indication: the deduction is talking about '{indication}'\n"
    prompt += "Guess the masked deduction.\n"
    return prompt


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Rate the similarity between two texts from 0 to 1 using cosine similarity.
    """
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings for both texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarity = util.cos_sim(embedding1, embedding2)

    return cosine_similarity.item()


def get_logic_score(llm_agent, generated_response, print_results: bool = False):
    """
    Get the logic score of the response.
    Args:
        llm_agent (LLM_Agent): The LLM agent to use for inference.
        generated_response (str): The response from the LLM.
        print_results (bool): Whether to print the results or not.
    Returns:
        float: The logic score of the response.
    """
    # obs_similarities = infer_the_observation(llm_agent, generated_response, print_results) # Hide all the observations
    obs_similarities = infer_observations_one_by_one(llm_agent, generated_response, print_results) # Hide the observations one by one

    deduc_similarities = infer_deductions(llm_agent, generated_response, print_results)

    avg_similarity_score = (sum(obs_similarities) + sum(deduc_similarities)) / (len(obs_similarities) + len(deduc_similarities))

    return avg_similarity_score
    

if __name__ == "__main__":
    generated_response = "<observation>The current speed is 31.4153 m\/s.</observation> \n<observation>The headway distance is 34.9655 meters, indicating a safe following distance given the speed.</observation>\n<observation>The speed of the vehicle ahead (leader speed) is 31.8674 m\/s, which is slightly higher than my current speed.</observation>\n<deduction>The headway is adequate for the current difference in speed, so I do not need to accelerate or decelerate aggressively at this moment.</deduction>\n<deduction>However, the headway will decrease by 30% over the next 5 seconds, prompting a potential safety concern if my speed and following behavior remain unchanged.</deduction>\n<deduction>To avoid the risk of a collision with the vehicle ahead as the headway decreases, a gentle deceleration will be necessary.</deduction>\n<prediction>In the next 5 seconds, if I decelerate moderately, it will allow the gap to diminish less drastically, enabling a smoother transition without abrupt changes that could create stop-and-go waves.</prediction>\n<prediction>After 10 seconds, if I maintain a smooth deceleration, I will stay at a safe distance from the leader vehicle and ensure traffic remains flowing smoothly.</prediction>\n<action>-0.5</action>"

    llm_agent = OpenAiModel()
    print_results = True # To print what the judge LLM generated to fill the missing observations and deductions. 

    avg_similarity_score = get_logic_score(llm_agent, generated_response, print_results)

    print("--------------------------------------------------")
    print(f"Logic score: {avg_similarity_score}")
    print("--------------------------------------------------")
