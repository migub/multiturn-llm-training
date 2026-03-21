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
        logging_steps: int = 5,
        **kwargs
    ):
        self.game_type = game_type
        self.seed = seed
        self.set_seed(self.seed)

        # COOPERATIVE: Store reward weights
        self.lambda_self = lambda_self
        self.lambda_welfare = lambda_welfare
        self.lambda_fair = lambda_fair

        # Metrics accumulator for wandb logging
        self.logging_steps = logging_steps
        self._metrics_accumulator = []
        self._reward_call_count = 0

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


    def compute_max_metrics(self, game: Game, negotiation_role: int = 1) -> dict:
        """
        Compute the maximum possible values for all metrics by brute-forcing
        all possible outcome combinations.
        
        For each issue there are ~11 discrete values. For 1 issue: 11 outcomes.
        For 2 issues: 11×11 = 121 outcomes. Trivial to enumerate.
        
        Returns dict with max values for U_A, social_welfare, nash_product, and R_coop.
        Each metric is maximized independently (different outcomes may maximize different metrics).
        """
        import itertools

        issues = game.issues
        
        # For each issue, get all possible (payoff_agent1, payoff_agent2) pairs
        # issue.payoffs are already rescaled by reweigh_issues(), so use them directly
        # (same values the evaluator uses in get_payoffs())
        issue_payoff_pairs = []
        for issue_idx, issue in enumerate(issues):
            pairs = []
            for val_idx in range(len(issue.payoffs[0])):
                p0 = float(issue.payoffs[0][val_idx])
                p1 = float(issue.payoffs[1][val_idx])
                pairs.append((p0, p1))
            issue_payoff_pairs.append(pairs)
        
        # Enumerate all combinations, track max for each metric independently
        max_U_A = 0.0
        max_social_welfare = 0.0
        max_nash_product = 0.0
        max_r_coop = 0.0
        
        for combo in itertools.product(*issue_payoff_pairs):
            total_p0 = sum(p[0] for p in combo)
            total_p1 = sum(p[1] for p in combo)
            
            if negotiation_role == 1:
                U_A, U_B = total_p0, total_p1
            else:
                U_A, U_B = total_p1, total_p0
            
            social_welfare = U_A + U_B
            nash_product = U_A * U_B
            
            r_coop = (
                self.lambda_self * U_A
                + self.lambda_welfare * social_welfare
                + self.lambda_fair * nash_product / 100.0
            )
            
            max_U_A = max(max_U_A, U_A)
            max_social_welfare = max(max_social_welfare, social_welfare)
            max_nash_product = max(max_nash_product, nash_product)
            max_r_coop = max(max_r_coop, r_coop)
        
        return {
            "max_U_A": max_U_A,
            "max_social_welfare": max_social_welfare,
            "max_nash_product": max_nash_product,
            "max_r_coop": max_r_coop,
        }


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
            

        elif self.game_type in {"multi-game", "cooperative-only", "out-of-domain"}:
            if self.game_type == "cooperative-only":
                games_used = [
                    {"game_settings": "joint-venture.yaml",
                    "issues": ["jv-rd-budget.yaml", "jv-revenue-split.yaml", "jv-data-sharing.yaml", "jv-decision-authority.yaml"]},
                    {"game_settings": "employment-contract.yaml",
                    "issues": ["ec-salary.yaml", "ec-remote-work.yaml", "ec-training-budget.yaml", "ec-equity.yaml", "ec-project-scope.yaml"]},
                ]
            elif self.game_type == "multi-game":
                games_used = [
                    {"game_settings": "generic-rental-agreement.yaml",
                    "issues": ["gen-ra-deposit.yaml","gen-ra-duration-distributive.yaml","gen-ra-duration.yaml","gen-ra-rent.yaml"]},
                    {"game_settings": "generic-loan-agreement.yaml",
                    "issues": ["gen-la-amount.yaml","gen-la-duration.yaml","gen-la-fees.yaml","gen-la-rate.yaml"]},
                    {"game_settings": "generic-merger.yaml",
                    "issues": ["gen-m-benefits.yaml", "gen-m-ownership.yaml"]},
                    # NEW: Cooperative scenarios
                    {"game_settings": "joint-venture.yaml",
                    "issues": ["jv-rd-budget.yaml", "jv-revenue-split.yaml", "jv-data-sharing.yaml", "jv-decision-authority.yaml"]},
                    {"game_settings": "employment-contract.yaml",
                    "issues": ["ec-salary.yaml", "ec-remote-work.yaml", "ec-training-budget.yaml", "ec-equity.yaml", "ec-project-scope.yaml"]},
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

                # Filter out conflicting issue combinations
                excluded_combos = {
                    frozenset(("gen-ra-duration-distributive.yaml", "gen-ra-duration.yaml")),
                }

                for combo in combos:
                    if frozenset(combo) in excluded_combos:
                        continue
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


    def create_eval_dataset(self) -> Dataset:
        """Fixed, curated eval dataset — 10 samples per game type, deterministic."""
        with open(self.rules_path, "r") as f:
            rules = yaml.safe_load(f)

        eval_configs = {
            "multi-game": [
                # 1. Single distributive (rental)
                {"game_settings": "generic-rental-agreement.yaml",
                 "issues": ["gen-ra-rent.yaml"], "issue_weights": [[1], [1]]},
                # 2. Two distributive issues (loan)
                {"game_settings": "generic-loan-agreement.yaml",
                 "issues": ["gen-la-amount.yaml", "gen-la-rate.yaml"], "issue_weights": [[70, 30], [30, 70]]},
                # 3. Merger (distributive combo)
                {"game_settings": "generic-merger.yaml",
                 "issues": ["gen-m-benefits.yaml", "gen-m-ownership.yaml"], "issue_weights": [[50, 50], [50, 50]]},
                # 4. JV compatible + distributive
                {"game_settings": "joint-venture.yaml",
                 "issues": ["jv-rd-budget.yaml", "jv-revenue-split.yaml"], "issue_weights": [[50, 50], [50, 50]]},
                # 5. EC integrative + compatible
                {"game_settings": "employment-contract.yaml",
                 "issues": ["ec-remote-work.yaml", "ec-training-budget.yaml"], "issue_weights": [[50, 50], [50, 50]]},
            ],
            "cooperative-only": [
                # 1. JV single distributive
                {"game_settings": "joint-venture.yaml",
                 "issues": ["jv-revenue-split.yaml"], "issue_weights": [[1], [1]]},
                # 2. JV compatible + distributive
                {"game_settings": "joint-venture.yaml",
                 "issues": ["jv-rd-budget.yaml", "jv-revenue-split.yaml"], "issue_weights": [[50, 50], [50, 50]]},
                # 3. JV compatible + integrative
                {"game_settings": "joint-venture.yaml",
                 "issues": ["jv-data-sharing.yaml", "jv-decision-authority.yaml"], "issue_weights": [[70, 30], [30, 70]]},
                # 4. EC single distributive
                {"game_settings": "employment-contract.yaml",
                 "issues": ["ec-salary.yaml"], "issue_weights": [[1], [1]]},
                # 5. EC integrative + compatible
                {"game_settings": "employment-contract.yaml",
                 "issues": ["ec-remote-work.yaml", "ec-training-budget.yaml"], "issue_weights": [[50, 50], [50, 50]]},
            ],
            "out-of-domain": [
                # 1. Single compatible
                {"game_settings": "rio_copa.yaml",
                 "issues": ["rp_contingent_liability.yaml"], "issue_weights": [[1], [1]]},
                # 2. Single integrative
                {"game_settings": "rio_copa.yaml",
                 "issues": ["rp_family_employees.yaml"], "issue_weights": [[1], [1]]},
                # 3. Integrative + distributive
                {"game_settings": "rio_copa.yaml",
                 "issues": ["rp_financing.yaml", "rp_non_compete_period.yaml"], "issue_weights": [[50, 50], [50, 50]]},
                # 4. Compatible + integrative
                {"game_settings": "rio_copa.yaml",
                 "issues": ["rp_contingent_liability.yaml", "rp_family_employees.yaml"], "issue_weights": [[70, 30], [30, 70]]},
                # 5. Integrative + integrative
                {"game_settings": "rio_copa.yaml",
                 "issues": ["rp_financing.yaml", "rp_family_employees.yaml"], "issue_weights": [[50, 50], [50, 50]]},
            ],
        }

        if self.game_type not in eval_configs:
            # Fallback: use the same game type as key, or default to multi-game
            configs = eval_configs.get("multi-game")
        else:
            configs = eval_configs[self.game_type]

        samples = []
        for gc in configs:
            gc = {**gc, "scale": SCALE, **rules}
            gc = self.add_game_info_to_game_config(gc)
            game_info = gc.pop("game_info")
            gc.update(game_info)

            game = Game(**gc)
            prompt1, prompt2 = self.get_prompts_from_game(game)
            archetype = self.get_archetype_from_game(game)

            samples.append({
                "prompt": prompt1, "prompt_2": prompt2,
                "game_config": gc, "starting_agent": True,
                "game_type": self.game_type, "negotiation_role": 1,
                "archetype": archetype,
            })
            samples.append({
                "prompt": prompt2, "prompt_2": prompt1,
                "game_config": gc, "starting_agent": False,
                "game_type": self.game_type, "negotiation_role": 2,
                "archetype": archetype,
            })

        return Dataset.from_list(samples)


    def get_reward_functions(self):
        # COOPERATIVE: Capture lambda parameters in closure
        lambda_self = self.lambda_self
        lambda_welfare = self.lambda_welfare
        lambda_fair = self.lambda_fair
        env_self = self  # capture for accumulator access in closure

        def negotiation_payoff_reward(prompts, completions, get_full_info=False, game_config=None, negotiation_roles=None, negotiation_role=None, **kwargs):
            # Support both singular (from dataset) and plural (legacy) parameter names
            if negotiation_roles is None and negotiation_role is not None:
                negotiation_roles = negotiation_role if isinstance(negotiation_role, list) else [negotiation_role] * len(completions)
            rewards = []
            evaluations = [] if get_full_info else None

            # Collect metrics for wandb batch logging
            batch_U_A = []
            batch_U_B = []
            batch_ratio_self = []
            batch_ratio_welfare = []
            batch_ratio_nash = []
            batch_ratio_rcoop = []
            batch_max_rcoop = []
            batch_archetypes = []
            batch_agreed = []
            
            for i, messages in enumerate(completions):
                messages = messages[1:]
                starting_agent = 0
                if messages and messages[0]["role"] == "user":
                    starting_agent = 1

                current_game_config = game_config[i] if isinstance(game_config, list) else game_config
                game = Game(**current_game_config)

                # Determine negotiation role
                if negotiation_roles is None or not isinstance(negotiation_roles, list) or len(negotiation_roles) == 0:
                    negotiation_roles = [1] * len(completions)
                current_role = negotiation_roles[i] if negotiation_roles[i] is not None else 1

                evaluation_model = OpenAIModel(model_provider="openai", model_name="gpt-4o-mini")
                evaluator = Evaluator(model=evaluation_model, game=game, game_type=self.game_type)

                evaluation = evaluator.evaluate(
                    messages, 
                    starting_agent=starting_agent, 
                    get_payoffs=True
                )

                # Compute max metrics for this game (needed for both success and failure cases)
                max_metrics = self.compute_max_metrics(game, current_role)

                if evaluation is None:
                    print(f"Warning: Evaluation returned None for sample {i}, assigning reward 0.0")
                    rewards.append(0.0)
                    batch_U_A.append(0.0)
                    batch_U_B.append(0.0)
                    batch_ratio_self.append(0.0)
                    batch_ratio_welfare.append(0.0)
                    batch_ratio_nash.append(0.0)
                    batch_ratio_rcoop.append(0.0)
                    batch_max_rcoop.append(max_metrics["max_r_coop"])
                    batch_archetypes.append(self.get_archetype_from_game(game))
                    batch_agreed.append(False)
                    if get_full_info:
                        evaluations.append(None)
                    continue

                if "payoffs" not in evaluation:
                    print(f"Warning: No payoffs in evaluation for sample {i}, assigning reward 0.0")
                    rewards.append(0.0)
                    batch_U_A.append(0.0)
                    batch_U_B.append(0.0)
                    batch_ratio_self.append(0.0)
                    batch_ratio_welfare.append(0.0)
                    batch_ratio_nash.append(0.0)
                    batch_ratio_rcoop.append(0.0)
                    batch_max_rcoop.append(max_metrics["max_r_coop"])
                    batch_archetypes.append(self.get_archetype_from_game(game))
                    batch_agreed.append(False)
                    if get_full_info:
                        evaluations.append(None)
                    continue
                
                payoff_agent1 = evaluation["payoffs"]["Agent 1"]
                payoff_agent2 = evaluation["payoffs"]["Agent 2"]
                
                if current_role == 1:
                    U_A = payoff_agent1
                    U_B = payoff_agent2
                else:
                    U_A = payoff_agent2
                    U_B = payoff_agent1

                social_welfare = U_A + U_B
                nash_product = U_A * U_B
                nash_product_normalized = nash_product / 100.0

                R_coop = (
                    lambda_self * U_A
                    + lambda_welfare * social_welfare
                    + lambda_fair * nash_product_normalized
                )

                ratio_self = U_A / max_metrics["max_U_A"] if max_metrics["max_U_A"] > 0 else 0.0
                ratio_welfare = social_welfare / max_metrics["max_social_welfare"] if max_metrics["max_social_welfare"] > 0 else 0.0
                ratio_nash = nash_product / max_metrics["max_nash_product"] if max_metrics["max_nash_product"] > 0 else 0.0
                ratio_rcoop = R_coop / max_metrics["max_r_coop"] if max_metrics["max_r_coop"] > 0 else 0.0

                rewards.append(R_coop)
                batch_U_A.append(U_A)
                batch_U_B.append(U_B)
                batch_ratio_self.append(ratio_self)
                batch_ratio_welfare.append(ratio_welfare)
                batch_ratio_nash.append(ratio_nash)
                batch_ratio_rcoop.append(ratio_rcoop)
                batch_max_rcoop.append(max_metrics["max_r_coop"])
                batch_archetypes.append(self.get_archetype_from_game(game))
                batch_agreed.append(U_A > 0 or U_B > 0)

                if get_full_info:
                    evaluation["R_coop"] = R_coop
                    evaluation["U_A"] = U_A
                    evaluation["U_B"] = U_B
                    evaluation["social_welfare"] = social_welfare
                    evaluation["nash_product"] = nash_product
                    evaluation["ratio_self"] = ratio_self
                    evaluation["ratio_welfare"] = ratio_welfare
                    evaluation["ratio_nash"] = ratio_nash
                    evaluation["ratio_rcoop"] = ratio_rcoop
                    evaluations.append(evaluation)

            # Accumulate metrics for averaged wandb logging
            is_training = torch.is_grad_enabled()
            try:
                import wandb
                if wandb.run is not None and not is_training:
                    # Eval mode: log immediately with eval/ prefix
                    n = len(batch_U_A) or 1
                    sw = [a + b for a, b in zip(batch_U_A, batch_U_B)]
                    agreements = sum(batch_agreed)
                    eval_metrics = {
                        "eval/negotiation/U_A_mean": sum(batch_U_A) / n,
                        "eval/negotiation/U_B_mean": sum(batch_U_B) / n,
                        "eval/negotiation/social_welfare_mean": sum(sw) / n,
                        "eval/negotiation/agreement_rate": agreements / n,
                        "eval/negotiation/ratio_self_mean": sum(batch_ratio_self) / n,
                        "eval/negotiation/ratio_welfare_mean": sum(batch_ratio_welfare) / n,
                        "eval/negotiation/ratio_nash_mean": sum(batch_ratio_nash) / n,
                        "eval/negotiation/ratio_rcoop_mean": sum(batch_ratio_rcoop) / n,
                    }
                    if agreements > 0:
                        agreed_ratio_rcoop = [v for v, a in zip(batch_ratio_rcoop, batch_agreed) if a]
                        agreed_ratio_self = [v for v, a in zip(batch_ratio_self, batch_agreed) if a]
                        agreed_ratio_welfare = [v for v, a in zip(batch_ratio_welfare, batch_agreed) if a]
                        eval_metrics["eval/negotiation/agreed/ratio_rcoop_mean"] = sum(agreed_ratio_rcoop) / len(agreed_ratio_rcoop)
                        eval_metrics["eval/negotiation/agreed/ratio_self_mean"] = sum(agreed_ratio_self) / len(agreed_ratio_self)
                        eval_metrics["eval/negotiation/agreed/ratio_welfare_mean"] = sum(agreed_ratio_welfare) / len(agreed_ratio_welfare)
                    wandb.log(eval_metrics, commit=False)

                if wandb.run is not None and is_training:
                    step_metrics = {
                        "U_A": list(batch_U_A),
                        "U_B": list(batch_U_B),
                        "ratio_self": list(batch_ratio_self),
                        "ratio_welfare": list(batch_ratio_welfare),
                        "ratio_nash": list(batch_ratio_nash),
                        "ratio_rcoop": list(batch_ratio_rcoop),
                        "max_rcoop": list(batch_max_rcoop),
                        "rewards": list(rewards),
                        "archetypes": list(batch_archetypes),
                        "agreed": list(batch_agreed),
                    }
                    env_self._metrics_accumulator.append(step_metrics)
                    env_self._reward_call_count += 1

                    if env_self._reward_call_count >= env_self.logging_steps:
                        # Flatten all accumulated batches
                        all_U_A = [v for s in env_self._metrics_accumulator for v in s["U_A"]]
                        all_U_B = [v for s in env_self._metrics_accumulator for v in s["U_B"]]
                        all_ratio_self = [v for s in env_self._metrics_accumulator for v in s["ratio_self"]]
                        all_ratio_welfare = [v for s in env_self._metrics_accumulator for v in s["ratio_welfare"]]
                        all_ratio_nash = [v for s in env_self._metrics_accumulator for v in s["ratio_nash"]]
                        all_ratio_rcoop = [v for s in env_self._metrics_accumulator for v in s["ratio_rcoop"]]
                        all_max_rcoop = [v for s in env_self._metrics_accumulator for v in s["max_rcoop"]]
                        all_rewards = [v for s in env_self._metrics_accumulator for v in s["rewards"]]
                        all_archetypes = [v for s in env_self._metrics_accumulator for v in s["archetypes"]]
                        all_agreed = [v for s in env_self._metrics_accumulator for v in s["agreed"]]

                        n = len(all_U_A) or 1
                        sw = [a + b for a, b in zip(all_U_A, all_U_B)]
                        np_vals = [a * b for a, b in zip(all_U_A, all_U_B)]
                        agreements = sum(all_agreed)

                        metrics = {
                            "negotiation/U_A_mean": sum(all_U_A) / n,
                            "negotiation/U_B_mean": sum(all_U_B) / n,
                            "negotiation/social_welfare_mean": sum(sw) / n,
                            "negotiation/nash_product_mean": sum(np_vals) / n,
                            "negotiation/agreement_rate": agreements / n,
                            "negotiation/ratio_self_mean": sum(all_ratio_self) / n,
                            "negotiation/ratio_welfare_mean": sum(all_ratio_welfare) / n,
                            "negotiation/ratio_nash_mean": sum(all_ratio_nash) / n,
                            "negotiation/ratio_rcoop_mean": sum(all_ratio_rcoop) / n,
                            "negotiation/max_rcoop_mean": sum(all_max_rcoop) / n,
                            "negotiation/rcoop_mean": sum(all_rewards) / n,
                        }

                        # Agreed-only metrics (quality of successful negotiations)
                        if agreements > 0:
                            agreed_U_A = [v for v, a in zip(all_U_A, all_agreed) if a]
                            agreed_U_B = [v for v, a in zip(all_U_B, all_agreed) if a]
                            agreed_ratio_self = [v for v, a in zip(all_ratio_self, all_agreed) if a]
                            agreed_ratio_welfare = [v for v, a in zip(all_ratio_welfare, all_agreed) if a]
                            agreed_ratio_nash = [v for v, a in zip(all_ratio_nash, all_agreed) if a]
                            agreed_ratio_rcoop = [v for v, a in zip(all_ratio_rcoop, all_agreed) if a]
                            m_a = len(agreed_U_A)
                            metrics["negotiation/agreed/U_A_mean"] = sum(agreed_U_A) / m_a
                            metrics["negotiation/agreed/U_B_mean"] = sum(agreed_U_B) / m_a
                            metrics["negotiation/agreed/social_welfare_mean"] = sum(a + b for a, b in zip(agreed_U_A, agreed_U_B)) / m_a
                            metrics["negotiation/agreed/ratio_self_mean"] = sum(agreed_ratio_self) / m_a
                            metrics["negotiation/agreed/ratio_welfare_mean"] = sum(agreed_ratio_welfare) / m_a
                            metrics["negotiation/agreed/ratio_nash_mean"] = sum(agreed_ratio_nash) / m_a
                            metrics["negotiation/agreed/ratio_rcoop_mean"] = sum(agreed_ratio_rcoop) / m_a

                        # Per-archetype metrics
                        from collections import defaultdict
                        arch_data = defaultdict(lambda: {"U_A": [], "U_B": [], "ratio_self": [], "ratio_welfare": [], "ratio_nash": [], "ratio_rcoop": []})
                        for idx, arch in enumerate(all_archetypes):
                            arch_data[arch]["U_A"].append(all_U_A[idx])
                            arch_data[arch]["U_B"].append(all_U_B[idx])
                            arch_data[arch]["ratio_self"].append(all_ratio_self[idx])
                            arch_data[arch]["ratio_welfare"].append(all_ratio_welfare[idx])
                            arch_data[arch]["ratio_nash"].append(all_ratio_nash[idx])
                            arch_data[arch]["ratio_rcoop"].append(all_ratio_rcoop[idx])

                        for arch, vals in arch_data.items():
                            m = len(vals["U_A"])
                            metrics[f"negotiation/{arch}/U_A_mean"] = sum(vals["U_A"]) / m
                            metrics[f"negotiation/{arch}/U_B_mean"] = sum(vals["U_B"]) / m
                            metrics[f"negotiation/{arch}/ratio_self_mean"] = sum(vals["ratio_self"]) / m
                            metrics[f"negotiation/{arch}/ratio_welfare_mean"] = sum(vals["ratio_welfare"]) / m
                            metrics[f"negotiation/{arch}/ratio_nash_mean"] = sum(vals["ratio_nash"]) / m
                            metrics[f"negotiation/{arch}/ratio_rcoop_mean"] = sum(vals["ratio_rcoop"]) / m
                            metrics[f"negotiation/{arch}/count"] = m

                        wandb.log(metrics, commit=False)

                        # Reset accumulator
                        env_self._metrics_accumulator = []
                        env_self._reward_call_count = 0
            except Exception:
                pass
            
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