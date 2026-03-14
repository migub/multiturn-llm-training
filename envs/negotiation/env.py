import time
import numpy as np
import torch
from typing import Any, Dict, List, Sequence, Optional, Union
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from envs.negotiation.games import Game
from evaluator.evaluator import Evaluator
from datasets import Dataset
from evaluator.openai_model import OpenAIModel
import itertools
import os
import yaml
import random
import json


SCALE = [100, 100]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class NegotiationEnv:
    def __init__(
        self,
        game_type: str = "generic-rental-agreement",
        seed: int = 42,
        # COOPERATIVE: Lambda parameters for R_coop
        lambda_self: float = 1.0,
        lambda_welfare: float = 0.0,
        lambda_fair: float = 0.0,
        **kwargs
    ):
        self.game_type = game_type
        self.seed = seed
        self.set_seed(self.seed)

        # COOPERATIVE: Store reward weights
        self.lambda_self = lambda_self
        self.lambda_welfare = lambda_welfare
        self.lambda_fair = lambda_fair

        base_dir = os.path.dirname(__file__)
        self.games_path = os.path.join(base_dir, "configs", "games")
        self.rules_path = os.path.join(base_dir, "configs", "general_game_rules.yaml")


    # ============================================================
    # Archetype helper method
    # ============================================================
    def get_archetype_from_game(self, game: Game) -> str:
        """
        Determine the archetype of a game based on its issues and weights.
        """
        if len(game.issues) == 1:
            issue_type = game.issues[0].issue_type
            return f"single-{issue_type}"
        else:
            return game.get_game_type()


    def get_prompts_from_game(self, game: Game, max_rounds: int = 5):
        prompts = game.get_system_game_msg(agent_id=0)["content"]
        prompts_2 = game.get_system_game_msg(agent_id=1)["content"]
        prompts += f"You have {max_rounds} rounds to reach an agreement; failure to do so will result in a total payoff of 0 for both parties, so it is crucial to find a mutually acceptable solution within these rounds. Also keep your answer as short and concise as possible. Directly state your offer and response to your negotiation partner, without showing your thought process. Make sure that you only talk about the issues in the payoff table, else your payoff will be 0. Do not mention your internal payoff to the other party, they are only for your reference, otherwise your payoff will be 0. "
        prompts_2 += f"You have {max_rounds} rounds to reach an agreement; failure to do so will result in a total payoff of 0 for both parties, so it is crucial to find a mutually acceptable solution within these rounds. Also keep your answer as short and concise as possible. Directly state your offer and response to your negotiation partner, without showing your thought process. Make sure that you only talk about the issues in the payoff table, else your payoff will be 0. Do not mention your internal payoff to the other party, they are only for your reference, otherwise your payoff will be 0. "
        return prompts, prompts_2
        


    def create_dataset(self, size=2000) -> Dataset:
        with open(self.rules_path, "r") as f:
            rules = yaml.safe_load(f)


        if self.game_type == "generic-rental-agreement":
            games_config = {
                "game_settings": "generic-rental-agreement.yaml",
                "issues": ["gen-ra-rent.yaml"],
                "issue_weights": [[1], [1]],
                "scale": SCALE,
                **rules,
            }

            games_config = self.add_game_info_to_game_config(games_config)
            game_info = games_config.pop("game_info")
            games_config.update(game_info)

            game = Game(**games_config)
            
            prompts, prompts_2 = self.get_prompts_from_game(game)

            archetype = self.get_archetype_from_game(game)

            samples = []
            for i in range(size):
                samples.append({
                    "prompt": prompts,
                    "prompt_2": prompts_2,
                    "game_config": games_config,
                    "starting_agent": i % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 1,
                    "archetype": archetype,
                })
        
            try:
                ds = Dataset.from_list(samples)
                _ = ds[0]
            except Exception:
                ds = Dataset.from_dict({"text": [prompts] * size})
            return ds
            

        elif self.game_type in {"multi-game", "out-of-domain"}:
            if self.game_type == "multi-game":
                games_used = [
                    {"game_settings": "generic-rental-agreement.yaml",
                    "issues": ["gen-ra-deposit.yaml","gen-ra-duration-distributive.yaml","gen-ra-duration.yaml","gen-ra-rent.yaml"]},
                    {"game_settings": "generic-loan-agreement.yaml",
                    "issues": ["gen-la-amount.yaml","gen-la-duration.yaml","gen-la-fees.yaml","gen-la-rate.yaml"]},
                    {"game_settings": "generic-merger.yaml",
                    "issues": ["gen-m-benefits.yaml", "gen-m-ownership.yaml"]},
                ]
            else:
                games_used = [
                    {"game_settings": "rio_copa.yaml",
                    "issues": ["rp_contingent_liability.yaml","rp_family_employees.yaml","rp_financing.yaml","rp_non_compete_period.yaml"]},
                ]

            for gd in games_used:
                gd = self.add_game_info_to_game_config(gd)

           
            issue_weights_multiple_possibilites = [90, 50, 10]

            game_configs = []
            for gd in games_used:
                issues = gd["issues"]
                game_info = gd["game_info"]
                for issue in issues:
                    game_configs.append({
                        "name": issue,
                        "issues": [issue],
                        "scale": SCALE,
                        **rules,
                        **game_info
                    })

                combos = list(itertools.combinations(issues, 2))

                for combo in combos:
                    game_configs.append({
                        "name": combo,
                        "issues": list(combo),
                        "scale": SCALE,
                        **rules,
                        **game_info
                    })

            print("Number of game configs: ", len(game_configs))
            game_configs = np.random.permutation(game_configs)
            
            samples = []
            for i in range(size//2):
                game_config = game_configs[i % len(game_configs)]
                if len(game_config["issues"]) == 1:
                    game_config["issue_weights"] = [[1], [1]]
                else:
                    iw_1 = np.random.choice(issue_weights_multiple_possibilites)
                    iw_2 = np.random.choice(issue_weights_multiple_possibilites)
                    game_config["issue_weights"] = [[iw_1, 100-iw_1], [iw_2, 100-iw_2]]

                game = Game(**game_config)
                prompt1, prompt2 = self.get_prompts_from_game(game)

                archetype = self.get_archetype_from_game(game)

                samples.append({
                    "prompt": prompt1,
                    "prompt_2": prompt2,
                    "game_config": game_config,
                    "starting_agent": (i // len(game_configs)) % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 1,
                    "archetype": archetype,
                })

                samples.append({
                    "prompt": prompt2,
                    "prompt_2": prompt1,
                    "game_config": game_config,
                    "starting_agent": (i // len(game_configs)) % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 2,
                    "archetype": archetype,
                })


            dataset = Dataset.from_list(samples)
            return dataset

        else:
            raise ValueError(f"Game type {self.game_type} not supported")



    def get_reward_functions(self):
        # COOPERATIVE: Capture lambda parameters in closure
        lambda_self = self.lambda_self
        lambda_welfare = self.lambda_welfare
        lambda_fair = self.lambda_fair

        def negotiation_payoff_reward(prompts, completions, get_full_info=False, game_config=None, negotiation_roles=None, **kwargs):
            rewards = []
            evaluations = [] if get_full_info else None
            
            for i, messages in enumerate(completions):
                messages = messages[1:]
                starting_agent = 0
                if messages and messages[0]["role"] == "user":
                    starting_agent = 1

                current_game_config = game_config[i] if isinstance(game_config, list) else game_config
                game = Game(**current_game_config)

                evaluation_model = OpenAIModel(model_provider="openai", model_name="gpt-4o-mini")
                evaluator = Evaluator(model=evaluation_model, game=game, game_type=self.game_type)

                evaluation = evaluator.evaluate(
                    messages, 
                    starting_agent=starting_agent, 
                    get_payoffs=True
                )

                if evaluation is None:
                    print(f"Warning: Evaluation returned None for sample {i}, assigning reward 0.0")
                    rewards.append(0.0)
                    if get_full_info:
                        evaluations.append(None)
                    continue
                
                # ============================================================
                # COOPERATIVE REWARD: R_coop = λ_self × U_A + λ_welfare × (U_A + U_B) + λ_fair × (U_A × U_B)
                # ============================================================
                payoff_agent1 = evaluation["payoffs"]["Agent 1"]
                payoff_agent2 = evaluation["payoffs"]["Agent 2"]

                # Determine which payoff belongs to the learning agent
                if negotiation_roles is None or not isinstance(negotiation_roles, list) or len(negotiation_roles) == 0:
                    negotiation_roles = [1] * len(completions)
                
                if negotiation_roles[i] == 1 or negotiation_roles[i] is None:
                    U_A = payoff_agent1  # Agent's own payoff
                    U_B = payoff_agent2  # Opponent's payoff
                else:
                    U_A = payoff_agent2
                    U_B = payoff_agent1

                # Compute cooperative reward
                self_utility = U_A
                social_welfare = U_A + U_B
                nash_product = U_A * U_B

                # Normalize Nash product to be on similar scale as other terms
                # U_A and U_B are 0-100, so Nash product is 0-10000
                # Divide by 100 to bring to 0-100 range
                nash_product_normalized = nash_product / 100.0

                R_coop = (
                    lambda_self * self_utility
                    + lambda_welfare * social_welfare
                    + lambda_fair * nash_product_normalized
                )

                rewards.append(R_coop)
                
                if get_full_info:
                    # Store both payoffs for analysis
                    evaluation["R_coop"] = R_coop
                    evaluation["U_A"] = U_A
                    evaluation["U_B"] = U_B
                    evaluation["social_welfare"] = social_welfare
                    evaluation["nash_product"] = nash_product
                    evaluations.append(evaluation)
            
            if get_full_info:
                return rewards, evaluations
            return rewards
        
        return [negotiation_payoff_reward]
          


    def add_game_info_to_game_config(self, game_config: dict):
        game_filename = game_config.get("game_settings") or game_config.get("game")
        if game_filename:
            with open(os.path.join(self.games_path, game_filename), "r") as f:
                game_dict = yaml.safe_load(f)
            game_config["game_info"] = game_dict
        return game_config


    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False