from math import ceil
import random
import gymnasium as gym
import re
from .utils import ActionFailed, ItemTag, ItemTagWithCount, Recipe, item_id_to_str
from .crafting_tree import CraftingTree
from typing import List


class TextCraftEnv(gym.Env[str, str]):
    def __init__(self, crafting_tree, commands, goal):
        self.inventory = {}
        self.action_regexes = {
            "craft": r"craft (.*) using (.*)",
            "get": r"get ([0-9]+) (.*)",
            "inventory": r"inventory",
        }
        self.count_regex = r"([0-9]+) (.*)"
        self.crafting_tree: CraftingTree = crafting_tree
        self.commands = commands
        self.goal = goal
        self.max_reward_unnormalized = 0

    def get_inventory_value(self) -> float:
        if self.max_reward_unnormalized == 0:
            return 0.0

        # recursively computes score by checking all min depths and seeing what the score is
        def recursive_score(item_name, needed_count, current_inv):
            score = 0
            inv_snapshot = current_inv.copy()

            available = inv_snapshot.get(item_name, 0)
            used = min(available, needed_count)
            
            if used > 0:
                score += used * self.crafting_tree.get_max_reward(item_name)
                inv_snapshot[item_name] -= used
            
            # for tags instead of items
            if used < needed_count and self.crafting_tree.is_tag(item_name):
                # Get all items that belong to this tag (e.g. oak, birch, spruce...)
                possible_items = self.crafting_tree.get_items_with_tags(item_name)
                
                # Intersect with what we actually have in inventory
                candidates = [i for i in possible_items if i in inv_snapshot]
                
                for cand in candidates:
                    if used >= needed_count:
                        break
                    
                    cand_available = inv_snapshot[cand]
                    needed_now = needed_count - used
                    take = min(cand_available, needed_now)
                    
                    used += take
                    inv_snapshot[cand] -= take
                    
                    # Reward based on the specific item's value
                    score += take * self.crafting_tree.get_max_reward(cand)
            
            still_needed = needed_count - used
            if still_needed <= 0:
                return score, inv_snapshot

            recipes = (self.crafting_tree.itemid_recipes.get(item_name) or 
                       self.crafting_tree.tag_recipes.get(item_name))

            if recipes:
                # get min recipes
                min_depth = min(self.crafting_tree.get_min_depth_recipes([r]) for r in recipes)
                
                best_recipes = [
                    r for r in recipes 
                    if self.crafting_tree.get_min_depth_recipes([r]) == min_depth
                ]

                # evaluate all recipes for best score
                best_branch_score = 0
                best_branch_inv = inv_snapshot
                
                for recipe in best_recipes:
                    branch_score = 0
                    branch_inv = inv_snapshot.copy()
                    
                    output_qty = recipe.output_item.count
                    batches_needed = ceil(still_needed / output_qty)
                    
                    for ingredient in recipe.input_items:
                        s, updated_inv = recursive_score( # recursively go thru each item
                            ingredient.item_tag.name, 
                            batches_needed * ingredient.count,
                            branch_inv
                        )
                        branch_score += s
                        branch_inv = updated_inv
                    
                    if branch_score > best_branch_score:
                        best_branch_score = branch_score
                        best_branch_inv = branch_inv
                
                
                score += best_branch_score
                return score, best_branch_inv

            # No recipes found (dead end)
            return score, inv_snapshot

        # Start recursion with a copy of the main inventory
        total_score, _ = recursive_score(self.goal, 1, self.inventory.copy())
        
        return total_score / self.max_reward_unnormalized

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}

        prev_inventory_value = self.get_inventory_value()

        try:
            for action_type, regex in self.action_regexes.items():
                match = re.match(regex, action)
                if match:
                    if action_type == "craft":
                        recipe = self.extract_recipe(match.group(1), match.group(2))
                        if recipe is None:
                            raise ActionFailed(
                                "Could not extract recipe from {}".format(action)
                            )
                        if not self.has_items(recipe.input_items):
                            raise ActionFailed(
                                "Could not find enough items to craft {}".format(
                                    recipe.output_item.item_tag.item_id
                                )
                            )
                        output_itemtag_count = self.crafting_tree.craft(recipe)
                        if output_itemtag_count is None:
                            raise ActionFailed(
                                "Could not find a valid recipe for {}".format(
                                    recipe.output_item
                                )
                            )
                        self.remove_items(recipe.input_items)
                        self.add_item(
                            output_itemtag_count.item_tag, output_itemtag_count.count
                        )
                        current_inventory_value = self.get_inventory_value()
                        reward = current_inventory_value - prev_inventory_value
                        observation = "Crafted {} {}".format(
                            output_itemtag_count.count,
                            output_itemtag_count.item_tag.item_id,
                        )
                        if output_itemtag_count.item_tag.item_id == self.goal:
                            terminated = True
                    elif action_type == "get":
                        (item, amt) = match.group(2), int(match.group(1))
                        item_obj = self.item_str_to_obj(item)
                        if self.crafting_tree.is_craftable(item_obj.name):
                            raise ActionFailed("Could not find {}".format(item))
                        if (
                            self.crafting_tree.is_tag(item_obj.item_id)
                            or item_obj.item_id is None
                        ):
                            raise ActionFailed("Could not find {}".format(item))
                        if not self.crafting_tree.is_valid_item(item_obj.item_id):
                            raise ActionFailed("Could not find {}".format(item))
                        self.add_item(item_obj, amt)
                        current_inventory_value = self.get_inventory_value()
                        reward = current_inventory_value - prev_inventory_value
                        observation = "Got {} {}".format(amt, item)
                        
                        if item_obj.item_id == self.goal:
                            terminated = True
                    elif action_type == "inventory":
                        observation = "Inventory: "
                        if not len(self.inventory.items()):
                            observation += "You are not carrying anything."
                        for item, amt in self.inventory.items():
                            observation += "[{}] ({}) ".format(
                                item_id_to_str(item), amt
                            )
                    else:
                        raise NotImplementedError(
                            "Action type {} not implemented".format(action_type)
                        )
            if observation is None:
                raise ActionFailed("Could not execute {}".format(action))

        except ActionFailed as e:
            observation = "{}".format(e.args[0])
            reward = 0
            info = {}

        return (observation, reward, terminated, truncated, info)

    def has_items(self, items: List[ItemTagWithCount]):
        for itemtag_count in items:
            if (
                itemtag_count.item_tag.item_id not in self.inventory
                or self.inventory[itemtag_count.item_tag.item_id] < itemtag_count.count
            ):
                return False
        return True

    def add_item(self, item_tag: ItemTag, amt: int):
        if item_tag.item_id not in self.inventory:
            self.inventory[item_tag.item_id] = 0
        self.inventory[item_tag.item_id] += amt

    def remove_items(self, items: List[ItemTagWithCount]):
        for itemtag_amts in items:
            self.inventory[itemtag_amts.item_tag.item_id] -= itemtag_amts.count
            if self.inventory[itemtag_amts.item_tag.item_id] == 0:
                del self.inventory[itemtag_amts.item_tag.item_id]

    def extract_recipe(self, output_item_str, input_items_str) -> Recipe:
        # check if there is a number in the output item
        m = re.match("([0-9]+) (.*)", output_item_str)
        if m:
            output_item = self.item_str_to_obj(m.group(2))
            output_item_count = int(m.group(1))
        else:
            output_item = self.item_str_to_obj(output_item_str)
            output_item_count = 1
        output_item_count = ItemTagWithCount(output_item, output_item_count)
        input_items = []
        for input_item_count in input_items_str.split(","):
            match = re.match(self.count_regex, input_item_count.strip())
            if match:
                count = int(match.group(1))
                item_str = match.group(2)
                input_item_obj = self.item_str_to_obj(item_str)
                input_items.append(ItemTagWithCount(input_item_obj, count))
            else:
                raise ActionFailed(
                    "Wrong item format: {}".format(input_item_count.strip())
                )
        return Recipe(input_items=input_items, output_item=output_item_count)

    def item_str_to_obj(self, item):
        item_id = "minecraft:" + item.replace(" ", "_")
        if self.crafting_tree.is_tag(item_id):
            return ItemTag(tag=item_id)
        else:
            return ItemTag(item_id=item_id)


    def reset(self, seed=42, data_idx=0, commands=None, goal=None):
        super().reset(seed=seed)
        # clean inventory
        self.inventory = {}
        if commands is not None and goal is not None:
            self.commands = commands
            self.goal = goal
            self.max_reward_unnormalized = self.crafting_tree.get_max_reward(self.goal)
            return (
                "Crafting commands:\n{}\n\nGoal: craft {}.".format(
                    self.commands, item_id_to_str(self.goal)
                ),
                {},
            )
        random.seed(seed)
        item_depth_list = list(self.crafting_tree.item_recipes_min_depth(1))
        # use idx to deterministically select goal
        sorted_item_depth_list = sorted(item_depth_list, key=lambda x: x[1])
        goal_depth = sorted_item_depth_list[data_idx % len(item_depth_list)]
        # example: self.goal = "minecraft:dark_oak_sign"
        self.goal = goal_depth[0]
        self.max_reward_unnormalized = self.crafting_tree.get_max_reward(self.goal)
        recipes_set = set()
        distractor_set = set()
        max_distractor = 10
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal)
        for recipe in recipes:
            recipes_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipes_set:
                distractor_set.add(distractor.recipe_str)

        recipes_list = list(recipes_set) + random.sample(
            list(distractor_set), min(len(distractor_set), max_distractor)
        )
        random.shuffle(recipes_list)
        self.commands = "\n".join(recipes_list)
        return (
            "Crafting commands:\n{}\n\nGoal: craft {}.".format(
                self.commands, item_id_to_str(self.goal)
            ),
            {},
        )

    def render(self, mode="human"):
        pass

    def close(self):
        super().close()
        for attr in ['inventory', 'action_regexes', 'count_regex', 'commands', 'goal']:
            if hasattr(self, attr):
                if isinstance(getattr(self, attr), (dict, list, set)):
                    getattr(self, attr).clear()
                delattr(self, attr)
