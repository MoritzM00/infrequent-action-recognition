import textwrap

from PIL import Image


def get_system_prompt():
    prompt = textwrap.dedent("""
        Role:
        You are an expert Human Activity Recognition (HAR) specialist. Analyze the video clip to classify the action or posture of the main subject.

        Allowed Labels:
        * Core: walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing, other
        * Extended (Rare): kneel_down, kneeling, squat_down, squatting, crawl, jump

        Definitions & Constraints:
        * Walk vs. Other: walk includes jogging, drunk walking, and carrying small items. Pushing large objects (chairs, carts) must be labeled other.
        * Fall vs. Lie/Sit:
            * fall: Uncontrolled, rapid descent (accidental).
            * lie_down / sit_down: Intentional, controlled descent.
        * Dynamic vs. Static:
            * Dynamic (Actions): walk, fall, sit_down, lie_down, stand_up. Label starts at first frame of motion.
            * Static (States): fallen, sitting, lying, standing. Label starts when subject comes to complete rest.

        Sequence Rules:
        * Moment of Rest: If a person transitions Sit -> Lie without pausing, use only the destination label.
        * Fall Termination: fall ends when inertia stops. fallen begins only when the person is on the ground in a resting state.

        Output Format:
        Return a JSON object:
        {
          "reasoning": "Brief analysis of intent, movement type, and objects.",
          "label": "<class_label>"
        }
    """).strip()  # .strip() removes the very first and last newlines caused by the triple quotes

    return prompt


def build_prompts(
    dataset, n_samples=5
):  # loads the omnifall test set and creates prompts with video samples
    prompt = get_system_prompt()
    messages = []
    samples = []
    for i in range(n_samples):
        sample = dataset[i]
        video_sample = sample.pop("video")  # list of frames (numpy arrays)
        samples.append(sample)

        # convert to PIL image
        video_sample = [Image.fromarray(frame) for frame in video_sample]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_sample,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

    return messages, samples


def run_inference():
    pass


def evaluate_predictions():
    pass
