# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


def load_novel_solution_generation_prompt(problem, solutions, k):
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the difference between two mathematical solutions include:
i). If the methods used to arrive at the solutions are fundamentally different, such as algebraic manipulation versus geometric reasoning, they can be considered distinct;
ii). Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;
iii). If two solutions rely on different assumptions or conditions, they are likely to be distinct;
iv). A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
v). If one solution is significantly simpler or more complex than the other, they can be regarded as essentially different, even if they lead to the same result. 

Given the following mathematical problem:
{problem} 

And some typical solutions:
{reference_solutions} 

Please output a novel solution distinct from the given ones for this math problem."""

    return prompt


def load_correctness_evaluation_prompt(problem, solutions, new_solution):
    # Provide two reference solutions if number of solutions more than one.
    if len(solutions) == 1:
        reference_solutions = f"Solution 1:\n{solutions[0]}"
    else:
        reference_solutions = "\n\n".join(
            [
                f"Solution {i + 1}:\n{solution}"
                for i, solution in enumerate(solutions[:2])
            ]
        )

    prompt = f"""Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please output "YES" if the new solution leads to the same result as the reference solutions; otherwise, output "NO".
YES or NO?"""

    return prompt


def load_coarse_grained_novelty_evaluation_prompt(problem, solutions, k, new_solution):
    # Provide the first k reference solutions
    first_k_solutions = solutions[:k]
    reference_solutions = "\n\n".join(
        [
            f"Solution {i + 1}:\n{solution}"
            for i, solution in enumerate(first_k_solutions)
        ]
    )

    prompt = f"""Criteria for evaluating the novelty of a new mathematical solution include:
1. If the new solution used to arrive at the solutions is fundamentally different from reference solutions, such as algebraic manipulation versus geometric reasoning, it can be considered novel;
2. Even if the final results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the new solution can be considered novel;
3. If the new solution relies on different assumptions or conditions, it should be considered novel;
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;
5. If the new solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem:
{problem}

Reference solutions:
{reference_solutions}

New solution:
{new_solution}

Please output "YES" if the new solution is a novel solution; otherwise, output "NO".
YES or NO?"""

    return prompt
