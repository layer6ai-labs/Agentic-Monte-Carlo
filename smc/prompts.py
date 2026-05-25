WEBSHOP_REACT_PROMPT = """You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
Your response should use the following format:

Thought:
I think ... 

Action: 
click[something]"""

TEXTCRAFT_REACT_PROMPT = """You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".
Every round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.
Your output must strictly follow this format:
"Thought: 
your thoughts.

Action:
your next action"

Reminder:
1. Always specify the quantity when using "get" and "craft" commands. - Example of get: get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot
2. When using "get" command, do not specify whether the item comes from the inventory or the environment.\n3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.
"""

SCIWORLD_REACT_PROMPT = """You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. Here are the actions you may take: 
[{"action": "open <<obj>", "description": "open a container"},
{"action": "close <<obj>", "description": "close a container"} 
{"action": "activate <obj>", "description": "activate a device"},
{"action": "deactivate <obj>", "description": "deactivate a device"},  
{"action": "connect <obj> to <obj>", "description": "connect electrical components"}, 
{"action": "disconnect <obj>", "description": "disconnect electrical components"}, 
{"action": "use <obj> [on <obj>]", "description": "use a device/item"}, 
{"action": "look around", "description": "describe the current room"}, 
{"action": "look at <obj>", "description": "describe an object in detail"}, 
{"action": "look in <obj>", "description": "describe a container\'s contents"}, 
{"action": "read <obj>", "description": "read a note or book"}, 
{"action": "move <obj> to <obj>", "description": "move an object to a container"}, 
{"action": "pick up <obj>", "description": "move an object to the inventory"}, 
{"action": "put down <obj>", "description": "drop an inventory item"}, 
{"action": "pour <obj> into <obj>", "description": "pour a liquid into a container"}, 
{"action": "dunk <obj> into <obj>", "description": "dunk a container into a liquid"}, 
{"action": "mix <obj>", "description": "chemically mix a container"}, 
{"action": "go to <location>", "description": "move to a new location"}, 
{"action": "eat <obj>", "description": "eat a food"}, 
{"action": "flush <obj>", "description": "flush a toilet"}, 
{"action": "focus on <obj>", "description": "signal intent on a task object"}, 
{"action": "wait", "description": "take no action for 10 iterations"}, 
{"action": "wait1", "description": "take no action for 1 iteration"}, 
{"action":"examine <obj>","description":"provides a description of the objects present on or in a receptacle."}, 
{"action": "task", "description": "describe current task"}, 
{"action": "inventory", "description": "list your inventory"}]

Your response should use the following format:
Thought:
your thoughts.

Action:
your next action
"""

SCIWORLD_REFLACT_PROMPT = """You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. Here are the actions you may take: [{"action": "open/close OBJ", "description": "open/close a container"}, {"action": "de/activate OBJ", "description": "activate/deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"}, {"action": "look in OBJ", "description": "describe a container\'s contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"}, {"action":"examine OBJ","description":"provides a description of the objects present on or in a receptacle."}, {"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"}]
You should first reflect in one sentence on the agent’s state in relation to the instruction, and then output the action for this turn.
Your response should use the following format:
Reflection:
your thoughts.

Action:
your next action"""


VALUE_FUNCTION_PROMPT_TEMPLATE_TEXTCRAFT = """You are a value estimator for a Minecraft crafting task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original crafting request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on the crafting material, intermediate crafting results and quantities relative to the INSTRUCTION.
Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}"""

VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_TEXTCRAFT = """You are a value estimator for a Minecraft crafting task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original crafting request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on the crafting material, intermediate crafting results and quantities relative to the INSTRUCTION.
Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}

Your response MUST be a single floating-point number between 0.0 and 1.0, and NOTHING ELSE.

Score:"""

VALUE_FUNCTION_PROMPT_TEMPLATE_WEBSHOP = """You are a value estimator for a web shopping task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original shopping request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on attributes, type, options, and price of shown products relative to the INSTRUCTION.
Based on your analysis, provide a score between 0.0 and 1.0.    

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}"""

VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION = """You are a value estimator for a web shopping task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original shopping request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on attributes, type, options, and price of shown products relative to the INSTRUCTION.
Based on your analysis, provide a score between 0.0 and 1.0.

First, analyze the real task data provided below.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}

Your response MUST be a single floating-point number between 0.0 and 1.0, and NOTHING ELSE.

Score:"""

VALUE_FUNCTION_FOA_TEMPLATE = """Given an item to purchase and a trajectory that aims to buy an item that exactly matches the specification, which corresponds to the ideal score of 10, analyze the following trajectory, then at the last line conclude "Thus the correctness score is ", where score is an integer from 1 to 10.

Here are some examples

Webshop [SEP] Instruction: [SEP] i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars [SEP] Search

Action: 
search[dairy free and apple variety pack of chips]

Observation:
Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B07HRFJWP8 [SEP] Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total) [SEP] $100.0 [SEP] B01KMHY5PG [SEP] Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars) [SEP] $21.49 [SEP] B008D2X8C4 [SEP] Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz) [SEP] $100.0

Thought:
The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.

Action: 
click[B008D2X8C4]

Observation:
Back to Search [SEP] < Prev [SEP] flavor name dill and sour cream garlic & parmesan light sea salt margherita pizza thai chili lime variety pack [SEP] size 0.8 ounce (pack of 24) 4 ounce (pack of 12) [SEP] Price: $100.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

Thought:
The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.

Action: 
click[variety pack]

Observation: 
Back to Search [SEP] < Prev [SEP] flavor name dill and sour cream garlic & parmesan light sea salt margherita pizza thai chili lime variety pack [SEP] size 0.8 ounce (pack of 24) 4 ounce (pack of 12) [SEP] Price: $100.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

Action: 
click[0.8 ounce (pack of 24)]

Observation: 
Back to Search [SEP] < Prev [SEP] flavor name dill and sour cream garlic & parmesan light sea salt margherita pizza thai chili lime variety pack [SEP] size 0.8 ounce (pack of 24) 4 ounce (pack of 12) [SEP] Price: $100.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now

Action: 
click[Buy Now]

Reflection: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.
Thus the correctness score is 5
Webshop [SEP] Instruction: [SEP] i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars [SEP] Search

Action: 
search[3 ounce bright citrus deodorant sensitive skin]

Observation:
Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B078GWRC1J [SEP] Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B078GTKVXY [SEP] Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce [SEP] $10.99 [SEP] B08KBVJ4XN [SEP] Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) [SEP] $15.95

Thought:
B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.

Reflection: Although the task is not yet complete, the first search is correct and returns relevant results. The thought is also logical and has potential to lead to the correct selection.
Thus the correctness score is 9
"""


VALUE_FUNCTION_PROMPT_TEMPLATE_SCIWORLD = """You are a value estimator for a science world task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original scientific request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on milestones, such as locating the correct room, acquiring required tools,
and following the specific steps (e.g., 'focus', 'interaction') outlined in the INSTRUCTION.
Heavily penalize states where the observation is 'No known action matches that input',
as this indicates the agent is stuck in an invalid command loop or syntax error.
Based on your analysis, provide a score between -1.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}"""


MOVIE_REACT_PROMPT = """You are an autonomous intelligent agent. You can use actions to help people solve problems.
We detail name, description, input(parameters) and output(returns) of each action as follows:
Name: get_search_movie(movie_name)
Description: Search for a movie by name and return basic details
Parameters:
- movie_name (Type: string): The name of the movie to search for.
Returns:
- id : The ID of the found movie.
- overview : The overview description of the movie.
- title : The title of the movie.

Name: get_movie_details(movie_id)
Description: Get detailed information about a movie by ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- budget : The budget of the movie.
- genres : The genres of the movie.
- revenue : The revenue of the movie.
- vote_average : The average vote score of the movie.
- release_date : The release date of the movie.

Name: get_movie_production_companies(movie_id)
Description: Get the production companies of a movie by its ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- production_companies : The production companies of the movie.

Name: get_movie_production_countries(movie_id)
Description: Get the production countries of a movie by its ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- production_countries : The production countries of the movie.

Name: get_movie_cast(movie_id)
Description: Retrieve the list of the top 10 cast members from a movie by its ID.
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- cast : List of the top 10 cast members.

Name: get_movie_crew(movie_id)
Description: Retrieve the list of crew members (limited to 10) from a movie by its ID. The list primarily includes Director, Producer, and Writer roles.
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- crew : List of the top 10 of crew members

Name: get_movie_keywords(movie_id)
Description: Get the keywords associated with a movie by ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- keywords : The keywords associated with the movie.

Name: get_search_person(person_name)
Description: Search for a person by name.
Parameters:
- person_name (Type: string): The name of the person to search for.
Returns:
- id : The ID of the found person.
- name : The name of the person.

Name: get_person_details(person_id)
Description: Get detailed information about a person by ID
Parameters:
- person_id (Type: string): The ID of the person.
Returns:
- biography : The biography of the person.
- birthday : The birthday of the person.
- place_of_birth : The place of birth of the person.

Name: get_person_cast(person_id)
Description: Retrieve the top 10 movie cast roles of a person by their ID
Parameters:
- person_id (Type: string): The ID of the person.
Returns:
- cast : A list of movies where the person has acted, limited to top 10

Name: get_person_crew(person_id)
Description: Retrieve the top 10 movie crew roles of a person by their ID
Parameters:
- person_id (Type: string): The ID of the person.
Returns:
- crew : A list of movies where the person has participated as crew, limited to top 10

Name: get_person_external_ids(person_id)
Description: Get the external ids for a person by ID
Parameters:
- person_id (Type: string): The ID of the person.
Returns:
- imdb_id : The IMDB id of the person.
- facebook_id : The Facebook id of the person.
- instagram_id : The Instagram id of the person.
- twitter_id : The Twitter id of the person.

Name: get_movie_alternative_titles(movie_id)
Description: Get the alternative titles for a movie by ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- titles : The alternative titles of the movie.
- id : The ID of the movie.

Name: get_movie_translation(movie_id)
Description: Get the description translation for a movie by ID
Parameters:
- movie_id (Type: string): The ID of the movie.
Returns:
- translations : The description translation of the movie.
- id : The ID of the movie.

Name: check_valid_actions()
Description: Get supported actions for current tool.
Returns:
- actions (Type: array): Supported actions for current tool.

Name: finish(answer)
Description: Return an answer and finish the task
Parameters:
- answer (Type: ['string', 'number', 'array']): The answer to be returned

If you want to get the movie_id or person_id, Please call "get_search_person", "get_search_movie"! Do not generate it by yourself which maybe wrong. If you are finished, you will call "finish" action.
Please refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!

Your response must be in the format of:
Thought: [your thought]

Action: [your action] with Action Input: [your action input]

Here is an example:

Goal: When did the movie Scream 6 come out?
Thought: I need to know the ID of the movie Scream 6 first.

Action: get_search_movie with Action Input: {"movie_name": "Scream 6"}
Observation: {'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}
Thought: I can get the release date from get_movie_details know.

Action: get_movie_details with Action Input: {"movie_id": "934433"}
Observation: {'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}
Thought: The release date is 2023-03-08.

Action: finish with Action Input: {"answer": "2023-03-08"}
Observation: 2023-03-08
"""

WEATHER_REACT_PROMPT = """You are an autonomous intelligent agent. You can use actions to help people solve problems.
We detail name, description, input(parameters) and output(returns) of each action as follows:
Name: get_user_current_date()
Description: Get the user's current date.
Returns:
The current date in 'YYYY-MM-DD' format.

Name: get_user_current_location()
Description: Get the user's current city.
Returns:
The user's current city.

Name: get_historical_temp(latitude, longitude, start_date, end_date)
Description: Get historical temperature data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).
- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).
Returns:
Historical temperature data.

Name: get_historical_rain(latitude, longitude, start_date, end_date)
Description: Get historical rainfall data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).
- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).
Returns:
Historical rainfall data.

Name: get_historical_snow(latitude, longitude, start_date, end_date)
Description: Get historical snowfall data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).
- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).
Returns:
Historical snowfall data.

Name: get_snow_forecast(latitude, longitude, start_date, end_date)
Description: Get snowfall forecast data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).
- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).
Returns:
Snowfall forecast data.

Name: get_current_snow(latitude, longitude, current_date)
Description: Get current snowfall data for a specified location and date.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- current_date (Type: string): The current date to retrieve snowfall data (YYYY-MM-DD).
Returns:
Current snowfall data.

Name: get_current_temp(latitude, longitude, current_date)
Description: Get current temperature data for a specified location and date.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- current_date (Type: string): The current date to retrieve temperature data (YYYY-MM-DD).
Returns:
Current temperature data.

Name: get_latitude_longitude(name)
Description: Get latitude and longitude information for a specified location name.
Parameters:
- name (Type: string): The name of the location. (e.g., city name)
Returns:
latitude and longitude information for the specified location.

Name: get_elevation(latitude, longitude)
Description: Get elevation data for a specified location.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
Returns:
Elevation data for the specified location.

Name: get_temp_forecast(latitude, longitude, start_date, end_date)
Description: Get temperature forecast data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).
- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).
Returns:
Temperature forecast data.

Name: get_rain_forecast(latitude, longitude, start_date, end_date)
Description: Get rainfall forecast data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).
- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).
Returns:
Rainfall forecast data.

Name: get_current_rain(latitude, longitude, current_date)
Description: Get current rainfall data for a specified location and date.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- current_date (Type: string): The current date to retrieve rainfall data (YYYY-MM-DD).
Returns:
Current rainfall data.

Name: get_distance(latitude1, longitude1, latitude2, longitude2)
Description: Calculate the distance between two sets of latitude and longitude coordinates.
Parameters:
- latitude1 (Type: number): The latitude of the first location.
- longitude1 (Type: number): The longitude of the first location.
- latitude2 (Type: number): The latitude of the second location.
- longitude2 (Type: number): The longitude of the second location.
Returns:
The distance between the two sets of coordinates in kilometers.

Name: get_historical_air_quality_index(latitude, longitude, start_date, end_date)
Description: Get historical air quality index data for a specified location and date range.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).
- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).
Returns:
Historical air quality index (PM2.5) data.

Name: get_current_air_quality_index(latitude, longitude, current_date)
Description: Get current air quality index data for a specified location and date.
Parameters:
- latitude (Type: number): The latitude of the location.
- longitude (Type: number): The longitude of the location.
- current_date (Type: string): The current date to retrieve air quality index data (YYYY-MM-DD).
Returns:
Current air quality index (PM2.5) data.

Name: get_air_quality_level(air_quality_index)
Description: Determine the air quality level based on the air quality index (AQI).
Parameters:
- air_quality_index (Type: number): The air quality index (AQI) value.
Returns:
The air quality level, which can be 'good', 'fair', 'moderate', 'poor', 'very poor', or 'extremely poor'.

Name: check_valid_actions()
Description: Get supported actions for current tool.
Returns:
- actions (Type: array): Supported actions for current tool.

Name: finish(answer)
Description: Return an answer and finish the task
Parameters:
- answer (Type: ['string', 'number', 'array']): The answer to be returned

If you want to get the latitude and longitude information of a city, you must call "get_latitude_longitude"! Do not generate it by yourself which maybe wrong. If you are finished, you will call "finish" action.
Please refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!

Your response must be in the format of:
Thought: [your thought]

Action: [your action] with Action Input: [your action input]

Here is an example:

Goal: What is the lowest temperature yesterday?
Thought: This question is about the lowest temperature yesterday, I should first get the location information of the user.

Action: get_user_current_location with Action Input: {}
Observation: Shanghai
Thought: The user is currently in Shanghai. I should first get the latitude and longitude information of Shanghai.

Action: get_latitude_longitude with Action Input: {"name": "Shanghai"}
Observation: {'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}
Thought: I have got the latitude and longitude information of Shanghai, I should get the current date to get the date of yesterday.

Action: get_user_current_date with Action Input: {}
Observation: 2015-01-02
Thought: Current date in 2015-01-02, so yesterday is 2015-01-01. Now, I can get the temperature data of Shanghai in 2015-01-01.

Action: get_historical_temp with Action Input: {"latitude": 31.22222, "longitude": 121.45806, "start_date": "2015-01-01", "end_date": "2015-01-01"}
Observation: {'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'temperature_2m_mean': '°C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}
Thought: The average temperature is -0.1, I will call finish to end the task.

Action: finish with Action Input: {"answer": -0.1}
Observation: -0.1"""

VALUE_FUNCTION_PROMPT_TEMPLATE_MOVIE = """You are a value estimator for an autonomous tool-use and API-calling agent.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES and ACTIONS provided as context.

Format of the input you will receive:

- INSTRUCTION: The original user request or problem to solve.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE (usually the output of the most recent tool call or an error message).

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on logical milestones, such as:

- Gathering necessary prerequisite context (e.g., properly using `get_search_movie` or `get_search_person` to map string names to their respective numeric IDs).
- Successfully navigating relational data structures (e.g., isolating a specific `person_id` from a `get_movie_crew` payload to subsequently look up their biography).
- Making valid tool calls that return successful, populated data payloads rather than empty lists or errors.

Heavily penalize states where:

- The agent hallucinates or guesses a `movie_id` or `person_id` instead of successfully retrieving it via a search action first.
- The agent passes a string name (e.g., "Kyle Balda") into an endpoint that strictly requires an ID (e.g., `get_person_details`).
- The STATE returns an API error, missing parameter warning, or "invalid action".
- The agent is stuck in an endless loop of repeating the exact same failed action.
- The agent calls the finish action with a clearly incorrect answer or an answer that is not strictly grounded in the observation history.

Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}"""

VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_MOVIE = """You are a value estimator for an autonomous tool-use and API-calling agent.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES and ACTIONS provided as context.

Format of the input you will receive:

- INSTRUCTION: The original user request or problem to solve.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE (usually the output of the most recent tool call or an error message).

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on logical milestones, such as:

- Gathering necessary prerequisite context (e.g., properly using `get_search_movie` or `get_search_person` to map string names to their respective numeric IDs).
- Successfully navigating relational data structures (e.g., isolating a specific `person_id` from a `get_movie_crew` payload to subsequently look up their biography).
- Making valid tool calls that return successful, populated data payloads rather than empty lists or errors.

Heavily penalize states where:

- The agent hallucinates or guesses a `movie_id` or `person_id` instead of successfully retrieving it via a search action first.
- The agent passes a string name (e.g., "Kyle Balda") into an endpoint that strictly requires an ID (e.g., `get_person_details`).
- The STATE returns an API error, missing parameter warning, or "invalid action".
- The agent is stuck in an endless loop of repeating the exact same failed action.
- The agent calls the finish action with a clearly incorrect answer or an answer that is not strictly grounded in the observation history.

Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}

Your response MUST be a single floating-point number between 0.0 and 1.0, and NOTHING ELSE.

Score:"""

VALUE_FUNCTION_PROMPT_TEMPLATE_WEATHER = """You are a value estimator for an autonomous tool-use and API-calling agent.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES and ACTIONS provided as context.

Format of the input you will receive:

- INSTRUCTION: The original user request or problem to solve.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE (usually the output of the most recent tool call or an error message).

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on logical milestones, such as:

- Gathering necessary prerequisite context (e.g., fetching the current date or user location).
- Successfully mapping named entities to required API parameters (e.g., converting a city name to latitude and longitude).
- Making valid tool calls that return successful data payloads rather than errors.

Heavily penalize states where:

- The STATE returns an API error, missing parameter warning, or "invalid action".
- The agent hallucinated a tool name that does not exist in the system prompt.
- The agent is stuck in an endless loop of repeating the exact same failed action.
- The agent calls the finish action with a clearly incorrect or ungrounded answer.

Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}"""

VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_WEATHER = """You are a value estimator for an autonomous tool-use and API-calling agent.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES and ACTIONS provided as context.

Format of the input you will receive:

- INSTRUCTION: The original user request or problem to solve.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE (usually the output of the most recent tool call or an error message).

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on logical milestones, such as:

- Gathering necessary prerequisite context (e.g., fetching the current date or user location).
- Successfully mapping named entities to required API parameters (e.g., converting a city name to latitude and longitude).
- Making valid tool calls that return successful data payloads rather than errors.

Heavily penalize states where:

- The STATE returns an API error, missing parameter warning, or "invalid action".
- The agent hallucinated a tool name that does not exist in the system prompt.
- The agent is stuck in an endless loop of repeating the exact same failed action.
- The agent calls the finish action with a clearly incorrect or ungrounded answer.

Based on your analysis, provide a score between 0.0 and 1.0.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}

Your response MUST be a single floating-point number between 0.0 and 1.0, and NOTHING ELSE.

Score:"""

VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_SCIWORLD = """You are a value estimator for a science world task.
Your job is to estimate the expected future reward (return) from the current STATE,
given the INSTRUCTION and the previous STATES, ACTIONS provided as context.

Format of the input you will receive:
- INSTRUCTION: The original scientific request to evaluate progress against.
- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.
- NOW: Current STATE.

Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.
Focus on milestones, such as locating the correct room, acquiring required tools,
and following the specific steps (e.g., 'focus', 'interaction') outlined in the INSTRUCTION.
Heavily penalize states where the observation is 'No known action matches that input',
as this indicates the agent is stuck in an invalid command loop or syntax error.
Based on your analysis, provide a score between -1.0 and 1.0.

First, analyze the real task data provided below.

=== INSTRUCTION ===
{instruction}

=== HISTORY ===
{history}

=== NOW ===
[STATE] {now}

Your response MUST be a single floating-point number between -1.0 and 1.0, and NOTHING ELSE.

Score:"""
