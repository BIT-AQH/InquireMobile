import io
import base64
from openai import OpenAI
from PIL import Image, ImageDraw
from qwen_vl_utils import smart_resize

def draw_circle(image, point, radius = 30, line_width = 5, color = 'blue'):
    draw = ImageDraw.Draw(image)
    width, height = image.width, image.height
    x, y = point[0], point[1]
    draw.circle((width * x, height *y), radius, width=line_width, outline=color, fill = None)
    return image


def draw_cross(image, point, radius = 50 * 0.71, line_width = 5, color = 'blue'):
    draw = ImageDraw.Draw(image)
    width, height = image.width, image.height
    x, y = point[0], point[1]
    draw.line([width * x - radius, height * y - radius, width * x + radius, height * y + radius], fill=color, width = line_width)
    draw.line([width * x - radius, height * y + radius, width * x + radius, height * y - radius], fill=color, width = line_width)
    return image


def check_click_match(image_path, point1, point2, min_pixels=32*28*28, max_pixels=1280*28*28):
    if isinstance(image_path, str):
        dummy_image = Image.open(image_path)
    else:
        dummy_image = image_path

    resized_height, resized_width  = smart_resize(dummy_image.height, dummy_image.width,
        factor=28,
        min_pixels=min_pixels, max_pixels=max_pixels)
    dummy_image.resize((resized_width, resized_height))
    
    dummy_image = draw_circle(dummy_image, point1, color='blue', line_width=5)
    dummy_image = draw_cross(dummy_image, point2, color='black', line_width=5)
    
    buffered = io.BytesIO()
    dummy_image.convert('RGB').save(buffered, format='png')
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # dummy_image = Image.open(BytesIO(base64.b64decode(base64_image.encode('utf-8'))))

    system_prompt = {'role': 'system',  'content': [{"type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. We will provide you a mobile phone screenshot and two cordinates, verify whether the two cordinates locate on a same UI element."}]}
    
    instruction = """
    The attached image is a screenshot showing the state of the phone.
    There are two marked coordinates: ({0}, {1}), indicated by a blue unfilled circle, and ({2}, {3}), indicated by a black cross. 
    Determine if both coordinates are within the same UI element. If so, answer 'yes'; otherwise, answer 'no'.
    """.format(int(point1[0]*resized_width), int(point1[1]*resized_height), int(point2[0]*resized_width), int(point2[1]*resized_height))
    
    user_query = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": instruction
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            }
        ]
    }
    
    messages = [system_prompt, user_query]
    
    
    client = OpenAI(
        api_key="sk-9414016927214a0ea62ed61e3137898e",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
                            model = "qwen2.5-vl-72b-instruct",
                            # model = "qwen2.5-vl-7b-instruct",
                            # model = "qwen2.5-vl-3b-instruct",
                            messages = messages
    )
    
    action_text = completion.choices[0].message.content
    
    if action_text.lower().startswith("yes"):
        return 1
    if action_text.lower().startswith("no"):
        return 0
    return None


def check_back_match(image_path, point, min_pixels=32*28*28, max_pixels=1280*28*28):
    if isinstance(image_path, str):
        dummy_image = Image.open(image_path)
    else:
        dummy_image = image_path

    resized_height, resized_width  = smart_resize(dummy_image.height, dummy_image.width,
        factor=28,
        min_pixels=min_pixels, max_pixels=max_pixels)
    dummy_image.resize((resized_width, resized_height))

    dummy_image = draw_circle(dummy_image, point, color='blue', line_width=5)
    
    buffered = io.BytesIO()
    dummy_image.convert('RGB').save(buffered, format='png')
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    

    system_prompt = {'role': 'system',  'content': [{"type": "text",
                    "text": "You are a helpful AI assistant for operating mobile phones. We will provide you a mobile phone screenshot and a cordinate, determine whether the element corresponding to this coordinate is a back button."}]}
    
    instruction = '''
    The attached image is a screenshot showing the state of the phone.
    The coordinate ({0}, {1}) is indicated by a blue unfilled circle.
    Determine whether the element corresponding to this coordinate is a back button. If so, answer 'yes'; otherwise, answer 'no'.
    '''.format(int(point[0]*resized_width), int(point[1]*resized_height))


    user_query = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": instruction
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            }
        ]
    }
    
    messages = [system_prompt, user_query]
    
    
    client = OpenAI(
        api_key="sk-9414016927214a0ea62ed61e3137898e",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
                            model = "qwen2.5-vl-72b-instruct",
                            # model = "qwen2.5-vl-7b-instruct",
                            # model = "qwen2.5-vl-3b-instruct",
                            messages = messages
    )
    
    action_text = completion.choices[0].message.content
    
    if action_text.lower().startswith("yes"):
        return 1
    if action_text.lower().startswith("no"):
        return 0
    return None
