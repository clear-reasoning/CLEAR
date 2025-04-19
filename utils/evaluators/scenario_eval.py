import re
from models.llm_agent import LLM_Agent, OpenAiModel


def extract_tag_content(response: str, tag_name: str) -> tuple[list[str], bool]:
    """Extract the content within a specific XML-style tag from a response."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    return (match.group(1).strip(), True) if match else ([], False)


def evaluate_scenario(
    current_situation: dict, hypothetical_situation: str, llm_agent
) -> float:
    """
    Evaluate the likelihood of a hypothetical scenario given the current situation using an LLM.

    Args:
        current_situation (dict): The current state of the vehicle.
        hypothetical_situation (str): The hypothetical scenario to evaluate.
        llm_model: The LLM model to use for evaluation.
        system_prompt (str): The system prompt for the LLM.
        user_prompt (str): The user prompt for the LLM.
        temperature (float): Temperature setting for the LLM.

    Returns:
        float: Normalized score from 0 to 1 indicating the likelihood of the scenario.
    """

    system_prompt = """
    You are an expert in evaluating the likelihood of hypothetical traffic scenarios based on current conditions. You will be provided with the following details about the current situation:
    - Your vehicle's speed (in meters per second)
    - The headway distance to the vehicle in front (in meters)
    - The speed of the vehicle in front (in meters per second)

    Additionally, you will be given a hypothetical scenario. Your task is to evaluate how likely this scenario is to occur, given the current conditions. Rate the likelihood on a scale from 0 to 5, where:
    - 0 indicates extremely unlikely
    - 5 indicates extremely likely

    For example, if the current state describes a smooth traffic flow and the hypothetical scenario mentions constant congestion with frequent acceleration and braking, you should assign a low score.

    Return the score enclosed in `<score>` tags.
    """

    user_prompt = """
    current_situation: {0}
    hypothetical_situation: {1}
    """

    text_current_situation = ", ".join(
        f"{key}: {value}" for key, value in current_situation.items()
    )
    user_prompt = user_prompt.format(text_current_situation, hypothetical_situation)

    # Get response from LLM
    response = llm_agent.get_response(
        system_prompt, user_prompt, temperature=1, num_samples=1
    )

    # Extract score from response
    score = extract_tag_content(response, "score")[0]

    # Normalize score to be between 0 and 1
    normalized_score = int(score) / 5

    return normalized_score


if __name__ == "__main__":

    current_situation = {"speed": 71.0, "headway": 35.0, "leader_speed": 70.5}

    hypothetical_situation = "What happens if the headway suddenly decreases from 18m to 10m over the next 3 seconds?"

    llm_agent = OpenAiModel()

    scenario_score = evaluate_scenario(
        current_situation, hypothetical_situation, llm_agent
    )

    print("--------------------------------------------------")
    print(f"Scenario score: {scenario_score}")
    print("--------------------------------------------------")
